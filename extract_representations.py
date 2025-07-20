import numpy as np
import torch
from PIL import Image
import os.path
import argparse
from pathlib import Path

from torch.utils.data import DataLoader
import tqdm
from utils.factory import create_model_and_transforms, get_tokenizer
from utils.binary_waterbirds import BinaryWaterbirds
from prs_hook import hook_prs_logger
from torchvision.datasets import CIFAR100, CIFAR10, ImageNet, ImageFolder
from utils.bd_dataset import get_test_dataloader
import pickle
from einops import rearrange, reduce, repeat
from torch.nn import functional as F

from timm.utils import AverageMeter


def get_args_parser():
    parser = argparse.ArgumentParser("Project Residual Stream", add_help=False)
    parser.add_argument("--batch_size", default=128, type=int, help="Batch size")
    # Model parameters
    parser.add_argument(
        "--model",
        default="ViT-B-32",
        type=str,
        metavar="MODEL",
        help="Name of model to use",
    )
    parser.add_argument("--pretrained", default="./model_path/accordion_blended_vitB32_cleanclip.pt", type=str) 
    # Dataset parameters
    parser.add_argument(
        "--data_path", default="/datasets/caltech-101", type=str, help="dataset path"
    )
    parser.add_argument(
        "--dataset", type=str, default="imagenet", help="imagenet, caltech101 or oxford_pets"
    )
    parser.add_argument("--num_workers", default=8, type=int)
    parser.add_argument(
        "--output_dir", default="./output_dir", help="path where to save"
    )
    parser.add_argument("--device", default="cuda:0", help="device to use for testing")

    # backdoor parameters
    parser.add_argument("--bd_model", default=True, action="store_true", help="backdoored model")
    parser.add_argument("--cleanclip", default=False, action="store_true", help="cleanclip model")
    parser.add_argument("--add_backdoor", default=False, action = "store_true", help = "add backdoor or not")
    parser.add_argument("--backdoor_type", default='badnet', type=str, help="backdoor attack")
    parser.add_argument("--patch_type", default = 'random', type = str, help = "patch type of backdoor")
    parser.add_argument("--patch_location", default = 'random', type = str, help = "patch location of backdoor")
    parser.add_argument("--patch_size", default = 16, type = int, help = "patch size of backdoor")
    parser.add_argument("--target_label", default = 954, type = int, help = "target label")



    return parser


def main(args):
    """Calculates the projected residual stream for a dataset."""
    model, _, preprocess = create_model_and_transforms(
        args.model, pretrained=args.pretrained
    )
    model.to(args.device)
    model.eval()
    context_length = model.context_length
    vocab_size = model.vocab_size

    print(
        "Model parameters:",
        f"{np.sum([int(np.prod(p.shape)) for p in model.parameters()]):,}",
    )
    print("Context length:", context_length)
    print("Vocab size:", vocab_size)
    print("Len of res:", len(model.visual.transformer.resblocks))

    prs = hook_prs_logger(model, args.device)


    attention_results = []
    mlp_results = []
    cls_to_cls_results = []


    suffix = "bdModel" if args.bd_model else "cleanModel"
    if args.cleanclip: suffix += '_cleanclip'
    suffix += f"_{args.backdoor_type}_{args.target_label}"
    if not args.add_backdoor: suffix += '_clean'

    dis_avg = [AverageMeter() for _ in range(12)]
    nmi_avg = [AverageMeter() for _ in range(12)]

    for i, (image, label) in enumerate(tqdm.tqdm(dataloader)):
        with torch.no_grad():
            prs.reinit()
            representation = model.encode_image(
                image.to(args.device), attn_method="head", normalize=False
            )

            attentions, mlps, token_maps = prs.finalize(representation)
            attentions = attentions.detach().cpu().numpy()  # [b, l, n, h, d]

            mlps = mlps.detach().cpu().numpy()  # [b, l+1, d]
            attention_results.append(
                np.sum(attentions, axis=2)
            )  # Reduce the spatial dimension
            mlp_results.append(mlps)
            cls_to_cls_results.append(
                np.sum(attentions[:, :, 0], axis=2)
            )  # Store the cls->cls attention, reduce the heads

    with open(
        os.path.join(args.output_dir, f"{args.dataset}_attn_{args.model}_{suffix}.npy"), "wb"
    ) as f:
        np.save(f, np.concatenate(attention_results, axis=0))
    with open(
        os.path.join(args.output_dir, f"{args.dataset}_mlp_{args.model}_{suffix}.npy"), "wb"
    ) as f:
        np.save(f, np.concatenate(mlp_results, axis=0))
    with open(
        os.path.join(args.output_dir, f"{args.dataset}_cls_attn_{args.model}_{suffix}.npy"), "wb"
    ) as f:
        np.save(f, np.concatenate(cls_to_cls_results, axis=0))


if __name__ == "__main__":
    args = get_args_parser()
    args = args.parse_args()
    print(args)
    if args.output_dir:
        Path(args.output_dir).mkdir(parents=True, exist_ok=True)
    main(args)
