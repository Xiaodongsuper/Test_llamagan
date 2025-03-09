# Modified from:
#   gpt-fast: https://github.com/pytorch-labs/gpt-fast/blob/main/generate.py
#   DiT:      https://github.com/facebookresearch/DiT/blob/main/models.py
import torch
import torch.nn as nn
from torch.nn import functional as F
import torch._dynamo.config
import torch._inductor.config
import copy
# torch._inductor.config.coordinate_descent_tuning = True
# torch._inductor.config.triton.unique_kernel_names = True
# torch._inductor.config.fx_graph_cache = True # Experimental feature to reduce compilation times, will be on by default in future


### from https://huggingface.co/transformers/v3.2.0/_modules/transformers/generation_utils.html
def top_k_top_p_filtering(
    logits,
    top_k: int = 0,
    top_p: float = 1.0,
    filter_value: float = -float("Inf"),
    min_tokens_to_keep: int = 1,
):
    """Filter a distribution of logits using top-k and/or nucleus (top-p) filtering.

    Args:
        logits: Logits distribution of shape (batch_size, vocabulary_size)
        top_k: If > 0, keep only the top k tokens with highest probability (top-k filtering)
        top_p: If < 1.0, keep tokens with cumulative probability >= top_p (nucleus filtering)
        filter_value: Value to assign to filtered tokens
        min_tokens_to_keep: Minimum number of tokens to keep per batch example

    Returns:
        torch.Tensor: Filtered logits distribution
    """
    if top_k > 0:
        top_k = min(max(top_k, min_tokens_to_keep), logits.size(-1))  # Safety check
        # Remove all tokens with a probability less than the last token of the top-k
        indices_to_remove = logits < torch.topk(logits, top_k)[0][..., -1, None]
        logits[indices_to_remove] = filter_value

    if top_p < 1.0:
        sorted_logits, sorted_indices = torch.sort(logits, descending=True)
        cumulative_probs = torch.cumsum(F.softmax(sorted_logits, dim=-1), dim=-1)

        # Remove tokens with cumulative probability above the threshold (token with 0 are kept)
        sorted_indices_to_remove = cumulative_probs > top_p
        if min_tokens_to_keep > 1:
            # Keep at least min_tokens_to_keep (set to min_tokens_to_keep-1 because we add the first one below)
            sorted_indices_to_remove[..., :min_tokens_to_keep] = 0
        # Shift the indices to the right to keep also the first token above the threshold
        sorted_indices_to_remove[..., 1:] = sorted_indices_to_remove[..., :-1].clone()
        sorted_indices_to_remove[..., 0] = 0

        # scatter sorted tensors to original indexing
        indices_to_remove = sorted_indices_to_remove.scatter(1, sorted_indices, sorted_indices_to_remove)
        logits[indices_to_remove] = filter_value
    return logits


def sample(logits, temperature: float=1.0, top_k: int=0, top_p: float=1.0, sample_logits=True):        
    """Sample next token from processed logits distribution.

    Args:
        logits: Model output logits
        temperature: Temperature parameter to scale logits
        top_k: Parameter for top-k filtering
        top_p: Parameter for nucleus sampling
        sample_logits: If True, sample from distribution; if False, take argmax

    Returns:
        tuple: (sampled token indices, probability distribution)
    """
    logits = logits[:, -1, :] / max(temperature, 1e-5)
    if top_k > 0 or top_p < 1.0:
        logits = top_k_top_p_filtering(logits, top_k=top_k, top_p=top_p)
    probs = F.softmax(logits, dim=-1)
    if sample_logits:
        idx = torch.multinomial(probs, num_samples=1)
    else:
        _, idx = torch.topk(probs, k=1, dim=-1)
    return idx, probs


def logits_to_probs(logits, temperature: float = 1.0, top_p: float=1.0, top_k: int = None, **kwargs):
    logits = logits / max(temperature, 1e-5)
    if top_k > 0 or top_p < 1.0:
        logits = top_k_top_p_filtering(logits, top_k=top_k, top_p=top_p)
    probs = torch.nn.functional.softmax(logits, dim=-1)
    return probs


def prefill(model, cond_idx: torch.Tensor, input_pos: torch.Tensor, cfg_scale: float, **sampling_kwargs):
    if cfg_scale > 1.0:
        logits, _ = model(None, cond_idx, input_pos)
        logits_combined = logits
        cond_logits, uncond_logits = torch.split(logits_combined, len(logits_combined) // 2, dim=0)
        logits = uncond_logits + (cond_logits - uncond_logits) * cfg_scale
    else:
        logits, _ = model(None, cond_idx, input_pos)

    return sample(logits, **sampling_kwargs)[0]


def decode_one_token(model, x: torch.Tensor, input_pos: torch.Tensor, cfg_scale: float, cfg_flag: bool, **sampling_kwargs):
    """Generate a single next token using the model.

    Args:
        model: The language model
        x: Current token sequence
        input_pos: Position of input in sequence
        cfg_scale: Classifier-free guidance scale
        cfg_flag: Whether to apply CFG
        **sampling_kwargs: Additional sampling parameters

    Returns:
        tuple: (next token, token probabilities)
    """
    assert input_pos.shape[-1] == 1
    if cfg_scale > 1.0:
        x_combined = torch.cat([x, x])
        logits, _ = model(x_combined, cond_idx=None, input_pos=input_pos)
        logits_combined = logits
        cond_logits, uncond_logits = torch.split(logits_combined, len(logits_combined) // 2, dim=0) 
        if cfg_flag:
            logits = uncond_logits + (cond_logits - uncond_logits) * cfg_scale
        else:
            logits = cond_logits
    else:
        logits, _ = model(x, cond_idx=None, input_pos=input_pos)
    return sample(logits, **sampling_kwargs)


def decode_n_tokens(
    model, cur_token: torch.Tensor, input_pos: torch.Tensor, num_new_tokens: int, 
    cfg_scale: float, cfg_interval: int,
    **sampling_kwargs):
    new_tokens, new_probs = [], []
    cfg_flag = True
    for i in range(num_new_tokens):
        with torch.backends.cuda.sdp_kernel(enable_flash=False, enable_mem_efficient=False, enable_math=True): # Actually better for Inductor to codegen attention here
            if cfg_interval > -1 and i > cfg_interval:
                cfg_flag = False
            next_token, next_prob = decode_one_token(
                model, cur_token, input_pos, cfg_scale, cfg_flag, **sampling_kwargs
            )
            input_pos += 1
            new_tokens.append(next_token.clone())
            new_probs.append(next_prob.clone())
            cur_token = next_token.view(-1, 1)
    
    return new_tokens, new_probs


@torch.no_grad()
def generate(model, cond, max_new_tokens, emb_masks=None, cfg_scale=1.0, cfg_interval=-1, **sampling_kwargs):
    """Generate a sequence of tokens using the model.

    Args:
        model: The language model
        cond: Conditioning input
        max_new_tokens: Maximum number of tokens to generate
        emb_masks: Optional embedding masks
        cfg_scale: Classifier-free guidance scale
        cfg_interval: Interval for applying CFG
        **sampling_kwargs: Additional sampling parameters

    Returns:
        torch.Tensor: Generated token sequence
    """
    if model.model_type == 'c2i':
        if cfg_scale > 1.0:
            cond_null = torch.ones_like(cond) * model.num_classes
            cond_combined = torch.cat([cond, cond_null])
        else:
            cond_combined = cond
        T = 1
    elif model.model_type == 't2i':
        if cfg_scale > 1.0:
            cond_null = torch.zeros_like(cond) + model.cls_embedding.uncond_embedding
            cond_combined = torch.cat([cond, cond_null])
        else:
            cond_combined = cond
        T = cond.shape[1]      
    else:
        raise Exception("please check model type")

    T_new = T + max_new_tokens
    max_seq_length = T_new
    max_batch_size = cond.shape[0]

    device = cond.device
    with torch.device(device):
        max_batch_size_cfg = max_batch_size * 2 if cfg_scale > 1.0 else max_batch_size
        model.setup_caches(max_batch_size=max_batch_size_cfg, max_seq_length=max_seq_length, dtype=model.tok_embeddings.weight.dtype)
    
    if emb_masks is not None:
        assert emb_masks.shape[0] == max_batch_size
        assert emb_masks.shape[-1] == T
        if cfg_scale > 1.0:
            model.causal_mask[:, :, :T] = model.causal_mask[:, :, :T] * torch.cat([emb_masks, emb_masks]).unsqueeze(1)
        else:
            model.causal_mask[:, :, :T] = model.causal_mask[:, :, :T] * emb_masks.unsqueeze(1)

        eye_matrix = torch.eye(model.causal_mask.size(1), model.causal_mask.size(2), device=device)
        model.causal_mask[:] = model.causal_mask * (1 - eye_matrix) + eye_matrix
    
    # create an empty tensor of the expected final shape and fill in the current tokens
    seq = torch.empty((max_batch_size, T_new), dtype=torch.int, device=device)

    input_pos = torch.arange(0, T, device=device)
    next_token = prefill(model, cond_combined, input_pos, cfg_scale, **sampling_kwargs)
    seq[:, T:T+1] = next_token

    input_pos = torch.tensor([T], device=device, dtype=torch.int)
    generated_tokens, _ = decode_n_tokens(model, next_token, input_pos, max_new_tokens-1, cfg_scale, cfg_interval, **sampling_kwargs)
    seq[:, T+1:] = torch.cat(generated_tokens, dim=1)

    return seq[:, T:]


class LabelEmbedder(nn.Module):
    """
    Embeds class labels into vector representations. Also handles label dropout for classifier-free guidance.
    
    Args:
        num_classes (int): Number of classes
        hidden_size (int): Hidden dimension size
        dropout_prob (float): Dropout probability
    """

class CaptionEmbedder(nn.Module):
    """
    Embeds text captions into vector representations. Also handles label dropout for classifier-free guidance.
    
    Args:
        in_channels (int): Input channel dimension
        hidden_size (int): Hidden dimension size 
        uncond_prob (float): Unconditional probability
        token_num (int): Number of tokens
    """

class MLP(nn.Module):
    """
    Multi-layer perceptron module.
    
    Args:
        in_features (int): Input feature dimension
        hidden_features (int): Hidden layer dimension
        out_features (int): Output feature dimension
    """

class RMSNorm(torch.nn.Module):
    """
    Root Mean Square normalization layer.
    
    Args:
        dim (int): Input dimension
        eps (float): Small constant for numerical stability
    """

class FeedForward(nn.Module):
    """
    Feed-forward neural network module.
    
    Args:
        config (ModelArgs): Model configuration
    """

class KVCache(nn.Module):
    """
    Key-Value cache module for accelerating autoregressive generation.
    
    Args:
        max_batch_size (int): Maximum batch size
        max_seq_length (int): Maximum sequence length
        n_head (int): Number of attention heads
        head_dim (int): Dimension of each head
        dtype: Data type
    """

class Attention(nn.Module):
    """
    Multi-head self-attention module.
    
    Args:
        config (ModelArgs): Model configuration
    """

class TransformerBlock(nn.Module):
    """
    Transformer block containing self-attention and feed-forward network.
    
    Args:
        config (ModelArgs): Model configuration
        drop_path (float): Drop path probability
    """

class Transformer(nn.Module):
    """
    Complete Transformer model.
    
    Args:
        config (ModelArgs): Model configuration
    """

def precompute_freqs_cis(seq_len: int, n_elem: int, base: int = 10000, cls_token_num=120):
    """
    Precompute frequencies for rotary position embeddings.
    
    Args:
        seq_len (int): Sequence length
        n_elem (int): Number of elements
        base (int): Base for frequency computation
        cls_token_num (int): Number of class tokens
        
    Returns:
        torch.Tensor: Precomputed frequency cache
    """

def precompute_freqs_cis_2d(grid_size: int, n_elem: int, base: int = 10000, cls_token_num=120):
    """
    Precompute frequencies for 2D rotary position embeddings.
    
    Args:
        grid_size (int): Size of 2D grid
        n_elem (int): Number of elements
        base (int): Base for frequency computation
        cls_token_num (int): Number of class tokens
        
    Returns:
        torch.Tensor: Precomputed 2D frequency cache
    """

def apply_rotary_emb(x: torch.Tensor, freqs_cis: torch.Tensor):
    """
    Apply rotary position embeddings to input tensor.
    
    Args:
        x (torch.Tensor): Input tensor (bs, seq_len, n_head, head_dim)
        freqs_cis (torch.Tensor): Precomputed frequencies (seq_len, head_dim//2, 2)
        
    Returns:
        torch.Tensor: Tensor with rotary embeddings applied
    """
