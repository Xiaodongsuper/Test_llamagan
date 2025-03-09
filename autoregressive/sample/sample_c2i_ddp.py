# Modified from:
#   DiT:  https://github.com/facebookresearch/DiT/blob/main/sample_ddp.py
import torch
torch.backends.cuda.matmul.allow_tf32 = True
torch.backends.cudnn.allow_tf32 = True
import torch.nn.functional as F
import torch.distributed as dist

from tqdm import tqdm
import os
from PIL import Image
import numpy as np
import math
import argparse

from tokenizer.tokenizer_image.vq_model import VQ_models
from autoregressive.models.gpt import GPT_models
from autoregressive.models.generate import generate
import pdb

def create_npz_from_sample_folder(sample_dir, num=50_000):
    """
    Converts a folder of PNG image samples into a single NPZ file.
    
    Args:
        sample_dir (str): Directory path containing PNG image samples
        num (int): Number of samples to process (default: 50,000)
    
    Returns:
        str: Path to the created NPZ file
        
    The function:
    1. Loads PNG images sequentially from the sample directory
    2. Converts them to numpy arrays
    3. Stacks them into a single array
    4. Saves the combined array as a compressed NPZ file
    """
    samples = []
    for i in tqdm(range(num), desc="Building .npz file from samples"):
        sample_pil = Image.open(f"{sample_dir}/{i:06d}.png")
        sample_np = np.asarray(sample_pil).astype(np.uint8)
        samples.append(sample_np)
    samples = np.stack(samples)
    assert samples.shape == (num, samples.shape[1], samples.shape[2], 3)
    npz_path = f"{sample_dir}.npz"
    np.savez(npz_path, arr_0=samples)
    print(f"Saved .npz file to {npz_path} [shape={samples.shape}].")
    return npz_path


def main(args):
    """
    Main function for distributed image generation using a trained GPT model.
    
    Args:
        args: Command line arguments containing model and sampling configurations
        
    The function:
    1. Initializes distributed training environment
    2. Sets up VQ (Vector Quantization) model:
       - Loads model architecture and weights
       - Moves model to appropriate device
    3. Sets up GPT model:
       - Configures model precision (fp32/bf16/fp16)
       - Loads model architecture and weights
       - Optionally compiles model for performance
    4. Generates images in batches:
       - Samples class indices
       - Generates token sequences using the GPT model
       - Decodes tokens to images using the VQ model
       - Saves generated images as PNG files
    5. Converts generated PNGs to a single NPZ file
    """
    # Setup PyTorch:
    assert torch.cuda.is_available(), "Sampling with DDP requires at least one GPU. sample.py supports CPU-only usage"
    torch.set_grad_enabled(False)

    # Setup DDP:
    dist.init_process_group("nccl")
    rank = dist.get_rank()
    device = rank % torch.cuda.device_count()
    seed = args.global_seed * dist.get_world_size() + rank
    torch.manual_seed(seed)
    torch.cuda.set_device(device)
    print(f"Starting rank={rank}, seed={seed}, world_size={dist.get_world_size()}.")

    # create and load model
    vq_model = VQ_models[args.vq_model](
        codebook_size=args.codebook_size,
        codebook_embed_dim=args.codebook_embed_dim)
    vq_model.to(device)
    vq_model.eval()
    checkpoint = torch.load(args.vq_ckpt, map_location="cpu")
    vq_model.load_state_dict(checkpoint["model"])
    del checkpoint

    # create and load gpt model
    precision = {'none': torch.float32, 'bf16': torch.bfloat16, 'fp16': torch.float16}[args.precision]
    latent_size = args.image_size // args.downsample_size
    gpt_model = GPT_models[args.gpt_model](
        vocab_size=args.codebook_size,
        block_size=latent_size ** 2,
        num_classes=args.num_classes,
        cls_token_num=args.cls_token_num,
        model_type=args.gpt_type,
    ).to(device=device, dtype=precision)
    print(f"gpt_model: {args.gpt_ckpt}")
    checkpoint = torch.load(args.gpt_ckpt, map_location="cpu")
    if args.from_fsdp: # fsdp
        model_weight = checkpoint
    elif "model" in checkpoint:  # ddp
        model_weight = checkpoint["model"]
    elif "module" in checkpoint: # deepspeed
        model_weight = checkpoint["module"]
    elif "state_dict" in checkpoint:
        model_weight = checkpoint["state_dict"]
    else:
        raise Exception("please check model weight, maybe add --from-fsdp to run command")
    # if 'freqs_cis' in model_weight:
    #     model_weight.pop('freqs_cis')
    
    gpt_model.load_state_dict(model_weight, strict=False)
    gpt_model.eval()
    del checkpoint

    if args.compile:
        print(f"compiling the model...")
        gpt_model = torch.compile(
            gpt_model,
            mode="reduce-overhead",
            fullgraph=True
        ) # requires PyTorch 2.0 (optional)
    else:
        print(f"no model compile") 

    # Create folder to save samples:
    model_string_name = args.gpt_model.replace("/", "-")
    if args.from_fsdp:
        ckpt_string_name = args.gpt_ckpt.split('/')[-2]
    else:
        ckpt_string_name = os.path.basename(args.gpt_ckpt).replace(".pth", "").replace(".pt", "")
    folder_name = f"{model_string_name}-{ckpt_string_name}-size-{args.image_size}-size-{args.image_size_eval}-{args.vq_model}-" \
                  f"topk-{args.top_k}-topp-{args.top_p}-temperature-{args.temperature}-" \
                  f"cfg-{args.cfg_scale}-seed-{args.global_seed}"
    sample_folder_dir = f"{args.sample_dir}/{folder_name}"
    if rank == 0:
        os.makedirs(sample_folder_dir, exist_ok=True)
        print(f"Saving .png samples at {sample_folder_dir}")
    dist.barrier()

    # Figure out how many samples we need to generate on each GPU and how many iterations we need to run:
    n = args.per_proc_batch_size
    global_batch_size = n * dist.get_world_size()
    # To make things evenly-divisible, we'll sample a bit more than we need and then discard the extra samples:
    total_samples = int(math.ceil(args.num_fid_samples / global_batch_size) * global_batch_size)
    if rank == 0:
        print(f"Total number of images that will be sampled: {total_samples}")
    assert total_samples % dist.get_world_size() == 0, "total_samples must be divisible by world_size"
    samples_needed_this_gpu = int(total_samples // dist.get_world_size())
    assert samples_needed_this_gpu % n == 0, "samples_needed_this_gpu must be divisible by the per-GPU batch size"
    iterations = int(samples_needed_this_gpu // n)
    pbar = range(iterations)
    pbar = tqdm(pbar) if rank == 0 else pbar
    total = 0
    for _ in pbar:
        # Sample inputs:
        c_indices = torch.randint(0, args.num_classes, (n,), device=device)
        qzshape = [len(c_indices), args.codebook_embed_dim, latent_size, latent_size]

        index_sample = generate(
            gpt_model, c_indices, latent_size ** 2,
            cfg_scale=args.cfg_scale, cfg_interval=args.cfg_interval,
            temperature=args.temperature, top_k=args.top_k,
            top_p=args.top_p, sample_logits=True, 
            )
        
        samples = vq_model.decode_code(index_sample, qzshape) # output value is between [-1, 1]
        if args.image_size_eval != args.image_size:
            samples = F.interpolate(samples, size=(args.image_size_eval, args.image_size_eval), mode='bicubic')
        samples = torch.clamp(127.5 * samples + 128.0, 0, 255).permute(0, 2, 3, 1).to("cpu", dtype=torch.uint8).numpy()
        
        # Save samples to disk as individual .png files
        for i, sample in enumerate(samples):
            index = i * dist.get_world_size() + rank + total
            Image.fromarray(sample).save(f"{sample_folder_dir}/{index:06d}.png")
        total += global_batch_size

    # Make sure all processes have finished saving their samples before attempting to convert to .npz
    dist.barrier()
    if rank == 0:
        create_npz_from_sample_folder(sample_folder_dir, args.num_fid_samples)
        print("Done.")
    dist.barrier()
    dist.destroy_process_group()



if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--gpt_model", type=str, choices=list(GPT_models.keys()), default="GPT-B")
    parser.add_argument("--gpt_ckpt", type=str, default='')
    parser.add_argument("--gpt_type", type=str, choices=['c2i', 't2i'], default="c2i", help="class-conditional or text-conditional")
    parser.add_argument("--from_fsdp", action='store_true')
    parser.add_argument("--cls_token_num", type=int, default=1, help="max token number of condition input")
    parser.add_argument("--precision", type=str, default='bf16', choices=["none", "fp16", "bf16"]) 
    parser.add_argument("--compile", action='store_true', default=True)
    parser.add_argument("--vq_model", type=str, choices=list(VQ_models.keys()), default="VQ-16")
    parser.add_argument("--vq_ckpt", type=str, default=None, help="ckpt path for vq model")
    parser.add_argument("--codebook_size", type=int, default=16384, help="codebook size for vector quantization")
    parser.add_argument("--codebook_embed_dim", type=int, default=8, help="codebook dimension for vector quantization")
    parser.add_argument("--image_size", type=int, choices=[256, 384, 512], default=384)
    parser.add_argument("--image_size_eval", type=int, choices=[256, 384, 512], default=256)
    parser.add_argument("--downsample_size", type=int, choices=[8, 16], default=16)
    parser.add_argument("--num_classes", type=int, default='200')
    parser.add_argument("--cfg_scale",  type=float, default=2.0)
    parser.add_argument("--cfg_interval", type=float, default=-1)
    parser.add_argument("--sample_dir", type=str, default="samples")
    parser.add_argument("--per_proc_batch_size", type=int, default=32)
    parser.add_argument("--num_fid_samples", type=int, default=50000)
    parser.add_argument("--global_seed", type=int, default=0)
    parser.add_argument("--top_k", type=int, default=0,help="top-k value to sample with")
    parser.add_argument("--temperature", type=float, default=1.0, help="temperature value to sample with")
    parser.add_argument("--top_p", type=float, default=1.0, help="top-p value to sample with")
    args = parser.parse_args()
    main(args)