"""Encoders. Every encoder maps a batch of images to one embedding vector per image."""

import torch
import torch.nn as nn


def conv_block(in_ch: int, out_ch: int, pool: bool = True) -> nn.Sequential:
    layers = [nn.Conv2d(in_ch, out_ch, kernel_size=3, padding=1), nn.BatchNorm2d(out_ch), nn.ReLU(inplace=True)]
    if pool:
        layers.append(nn.MaxPool2d(2))
    return nn.Sequential(*layers)


class ConvEncoder(nn.Module):
    """Plain stack of 3x3 conv blocks with four 2x2 poolings (input side / 16).

    depth 4 is the usual Conv-4; depth 6 adds two blocks without pooling.
    With `global_pool` the output is always 64-d; otherwise the final feature
    map is flattened and the embedding grows with the input size.
    """

    def __init__(self, depth: int = 4, width: int = 64, global_pool: bool = False):
        super().__init__()
        pools = {4: [True] * 4, 6: [True, False, True, False, True, True]}[depth]
        blocks, in_ch = [], 3
        for pool in pools:
            blocks.append(conv_block(in_ch, width, pool))
            in_ch = width
        self.encoder = nn.Sequential(*blocks)
        self.pool = nn.AdaptiveAvgPool2d(1) if global_pool else nn.Identity()

    def forward(self, x):
        return self.pool(self.encoder(x)).flatten(1)


class Dinov2(nn.Module):
    """DINOv2 ViT-S/14 from Hugging Face; the pooled CLS token is the embedding."""

    def __init__(self):
        super().__init__()
        from transformers import AutoModel

        self.model = AutoModel.from_pretrained("facebook/dinov2-small")

    def forward(self, x):
        return self.model(pixel_values=x).pooler_output


def build_encoder(name: str) -> nn.Module:
    from torchvision import models as tv

    if name == "conv4":
        return ConvEncoder(depth=4)
    if name == "conv6":
        return ConvEncoder(depth=6)
    if name == "conv6_gap":
        return ConvEncoder(depth=6, global_pool=True)
    if name == "resnet12":
        from easyfsl.modules import resnet12

        return resnet12()
    if name == "mobilenet_v2":
        net = tv.mobilenet_v2(weights=None)
        net.classifier = nn.Identity()
        return net
    if name == "shufflenet_v2":
        net = tv.shufflenet_v2_x1_0(weights=None)
        net.fc = nn.Identity()
        return net
    # encoders that start from ImageNet weights
    if name == "resnet18_imagenet":
        net = tv.resnet18(weights=tv.ResNet18_Weights.IMAGENET1K_V1)
        net.fc = nn.Identity()
        return net
    if name == "resnet50_imagenet":
        net = tv.resnet50(weights=tv.ResNet50_Weights.IMAGENET1K_V2)
        net.fc = nn.Identity()
        return net
    if name == "dinov2_small":
        return Dinov2()
    raise ValueError(f"unknown encoder '{name}'")


# Encoders used frozen expect ImageNet statistics; ours are trained on [-1, 1] inputs.
IMAGENET_NORMALIZED = {"resnet50_imagenet", "dinov2_small"}


def count_parameters(model: nn.Module) -> int:
    return sum(p.numel() for p in model.parameters())


@torch.no_grad()
def embed(model: nn.Module, images: torch.Tensor, device, imagenet_norm: bool = False):
    """Embed uint8 images [N, H, W] in eval mode; returns float32 features on the CPU."""
    from .data import to_input

    model.eval()
    batch = 256 if images.shape[-1] <= 84 else 64   # keeps large inputs within 4 GB of GPU memory
    out = []
    for i in range(0, len(images), batch):
        x = to_input(images[i:i + batch].to(device))
        if imagenet_norm:
            x = x * 0.5 + 0.5
            mean = torch.tensor([0.485, 0.456, 0.406], device=device).view(1, 3, 1, 1)
            std = torch.tensor([0.229, 0.224, 0.225], device=device).view(1, 3, 1, 1)
            x = (x - mean) / std
        out.append(model(x).float().cpu())
    return torch.cat(out)
