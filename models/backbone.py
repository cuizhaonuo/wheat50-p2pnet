# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved
"""
Backbone modules.
"""
import re
from collections import OrderedDict

import torch
import torch.nn.functional as F
import torchvision
from torch import nn

import models.vgg_ as models
from models.convnext_ import ConvNeXt, convnext_sizes

class BackboneBase_VGG(nn.Module):
    def __init__(self, backbone: nn.Module, num_channels: int, name: str, return_interm_layers: bool):
        super().__init__()
        features = list(backbone.features.children())
        if return_interm_layers:
            if name == 'vgg16_bn':
                self.body1 = nn.Sequential(*features[:13])
                self.body2 = nn.Sequential(*features[13:23])
                self.body3 = nn.Sequential(*features[23:33])
                self.body4 = nn.Sequential(*features[33:43])
            else:
                self.body1 = nn.Sequential(*features[:9])
                self.body2 = nn.Sequential(*features[9:16])
                self.body3 = nn.Sequential(*features[16:23])
                self.body4 = nn.Sequential(*features[23:30])
        else:
            if name == 'vgg16_bn':
                self.body = nn.Sequential(*features[:44])  # 16x down-sample
            elif name == 'vgg16':
                self.body = nn.Sequential(*features[:30])  # 16x down-sample
        self.num_channels = num_channels
        self.return_interm_layers = return_interm_layers

    def forward(self, tensor_list):
        out = []

        if self.return_interm_layers:
            xs = tensor_list
            for _, layer in enumerate([self.body1, self.body2, self.body3, self.body4]):
                xs = layer(xs)
                out.append(xs)

        else:
            xs = self.body(tensor_list)
            out.append(xs)
        return out


class Backbone_VGG(BackboneBase_VGG):
    """ResNet backbone with frozen BatchNorm."""
    def __init__(self, name: str, return_interm_layers: bool):
        if name == 'vgg16_bn':
            backbone = models.vgg16_bn(pretrained=True)
        elif name == 'vgg16':
            backbone = models.vgg16(pretrained=True)
        num_channels = 256
        super().__init__(backbone, num_channels, name, return_interm_layers)
        # channels of the last three maps, which are the decoder inputs
        self.fpn_channels = (256, 512, 512)


# the DINOv3 ConvNeXt sizes distilled from ViT-7B
CONVNEXT_ARCHS = {
    'dinov3_convnext_tiny': 'tiny',
    'dinov3_convnext_small': 'small',
    'dinov3_convnext_base': 'base',
    'dinov3_convnext_large': 'large',
}


def remap_convnext_state_dict(state_dict):
    """Rename Hugging Face DINOv3ConvNextModel keys to the upstream ConvNeXt ones.

    The tensors are identical, only the module paths differ. Keys that already
    use the upstream names are passed through unchanged.
    """
    remapped = {}
    for key, value in state_dict.items():
        match = re.match(r'stages\.(\d+)\.downsample_layers\.(\d+)\.(.+)$', key)
        if match:
            remapped['downsample_layers.{}.{}.{}'.format(*match.groups())] = value
            continue
        match = re.match(r'stages\.(\d+)\.layers\.(\d+)\.(.+)$', key)
        if match:
            stage, block, suffix = match.groups()
            for hf_name, upstream_name in (
                ('depthwise_conv', 'dwconv'),
                ('pointwise_conv1', 'pwconv1'),
                ('pointwise_conv2', 'pwconv2'),
                ('layer_norm', 'norm'),
            ):
                if suffix.startswith(hf_name):
                    suffix = upstream_name + suffix[len(hf_name):]
                    break
            remapped['stages.{}.{}.{}'.format(stage, block, suffix)] = value
            continue
        match = re.match(r'layer_norm\.(weight|bias)$', key)
        if match:
            # norm and norms[3] are the same module registered under two paths
            remapped['norm.{}'.format(match.group(1))] = value
            remapped['norms.3.{}'.format(match.group(1))] = value
            continue
        remapped[key] = value
    return remapped


def load_convnext_state_dict(weights_path):
    """Read DINOv3 ConvNeXt weights from a Hugging Face or upstream checkpoint."""
    if weights_path.endswith('.safetensors'):
        from safetensors.torch import load_file

        state_dict = load_file(weights_path)
    else:
        state_dict = torch.load(weights_path, map_location='cpu')
        state_dict = state_dict.get('model', state_dict)
    return remap_convnext_state_dict(state_dict)


class Backbone_ConvNeXt(nn.Module):
    """DINOv3 ConvNeXt backbone exposing the 4x, 8x and 16x stages.

    Unlike the ViT variants of DINOv3, the stages are already a spatial pyramid,
    so the decoder consumes them directly and no token reshaping is needed.
    """

    def __init__(self, name: str, weights_path: str = '', freeze: bool = False):
        super().__init__()
        size = convnext_sizes[CONVNEXT_ARCHS[name]]
        body = ConvNeXt(depths=size['depths'], dims=size['dims'])
        if weights_path:
            body.load_state_dict(load_convnext_state_dict(weights_path), strict=True)
        else:
            body.init_weights()
        # the 32x stage never reaches the decoder, so drop it after loading to
        # keep the checkpoints and the optimizer state smaller
        del body.stages[3]
        del body.downsample_layers[3]
        del body.norms
        del body.norm
        self.body = body
        self.num_channels = 256
        self.fpn_channels = tuple(size['dims'][:3])
        self.freeze = freeze
        if freeze:
            for parameter in self.body.parameters():
                parameter.requires_grad_(False)

    def train(self, mode: bool = True):
        super().train(mode)
        if self.freeze:
            self.body.eval()
        return self

    def forward(self, tensor_list):
        out = []
        xs = tensor_list
        for downsample, stage in zip(self.body.downsample_layers, self.body.stages):
            xs = downsample(xs)
            xs = stage(xs)
            out.append(xs)
        return out


def build_backbone(args):
    if args.backbone in CONVNEXT_ARCHS:
        return Backbone_ConvNeXt(
            args.backbone,
            weights_path=getattr(args, 'backbone_weights', ''),
            freeze=getattr(args, 'freeze_backbone', False),
        )
    backbone = Backbone_VGG(args.backbone, True)
    return backbone

if __name__ == '__main__':
    Backbone_VGG('vgg16', True)