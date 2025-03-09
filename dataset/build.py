from dataset.imagenet import build_imagenet, build_imagenet_code
from dataset.coco import build_coco
from dataset.cifar10 import build_cifar10
from dataset.cifar10 import build_cifar10_code
from dataset.openimage import build_openimage
from dataset.cub200 import build_cub200
from dataset.cub200 import build_cub200_code
from dataset.pexels import build_pexels
from dataset.t2i import build_t2i, build_t2i_code, build_t2i_image


def build_dataset(args, **kwargs):
    # images
    if args.dataset == 'imagenet':
        return build_imagenet(args, **kwargs)
    if args.dataset == 'cub200':
        return build_cub200(args, **kwargs)
    if args.dataset == 'cub200_code':
        return build_cub200_code(args, **kwargs)
    if args.dataset == 'cifar10':
        return build_cifar10(args, **kwargs)
    if args.dataset == 'cifar10_code':
        return build_cifar10_code(args, **kwargs)
    if args.dataset == 'imagenet_code':
        return build_imagenet_code(args, **kwargs)
    if args.dataset == 'coco':
        return build_coco(args, **kwargs)
    if args.dataset == 'openimage':
        return build_openimage(args, **kwargs)
    if args.dataset == 'pexels':
        return build_pexels(args, **kwargs)
    if args.dataset == 't2i_image':
        return build_t2i_image(args, **kwargs)
    if args.dataset == 't2i':
        return build_t2i(args, **kwargs)
    if args.dataset == 't2i_code':
        return build_t2i_code(args, **kwargs)
    
    raise ValueError(f'dataset {args.dataset} is not supported')