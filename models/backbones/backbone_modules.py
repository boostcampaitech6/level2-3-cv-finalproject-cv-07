"""
Backbone modules
"""

from collections import OrderedDict
from torchvision import models
import torch
import torch.nn.functional as F
import torchvision
from torch import nn

from util.misc import NestedTensor
from .vgg import *

# 추가 새로운 백본 import
from .mobilenetv3 import mobilenetv3
from ..position_encoding import build_position_encoding


class FeatsFusion(nn.Module):
    def __init__(
        self, C3_size, C4_size, C5_size, hidden_size=256, out_size=256, out_kernel=3
    ):
        super(FeatsFusion, self).__init__()
        self.P5_1 = nn.Conv2d(C5_size, hidden_size, kernel_size=1, stride=1, padding=0)
        self.P5_2 = nn.Conv2d(
            hidden_size,
            out_size,
            kernel_size=out_kernel,
            stride=1,
            padding=out_kernel // 2,
        )

        self.P4_1 = nn.Conv2d(C4_size, hidden_size, kernel_size=1, stride=1, padding=0)
        self.P4_2 = nn.Conv2d(
            hidden_size,
            out_size,
            kernel_size=out_kernel,
            stride=1,
            padding=out_kernel // 2,
        )

        self.P3_1 = nn.Conv2d(C3_size, hidden_size, kernel_size=1, stride=1, padding=0)
        self.P3_2 = nn.Conv2d(
            hidden_size,
            out_size,
            kernel_size=out_kernel,
            stride=1,
            padding=out_kernel // 2,
        )

    def forward(self, inputs):
        C3, C4, C5 = inputs
        C3_shape, C4_shape, C5_shape = C3.shape[-2:], C4.shape[-2:], C5.shape[-2:]

        P5_x = self.P5_1(C5)
        P5_upsampled_x = F.interpolate(P5_x, C4_shape)
        P5_x = self.P5_2(P5_x)

        P4_x = self.P4_1(C4)
        P4_x = P5_upsampled_x + P4_x
        P4_upsampled_x = F.interpolate(P4_x, C3_shape)
        P4_x = self.P4_2(P4_x)

        P3_x = self.P3_1(C3)
        P3_x = P3_x + P4_upsampled_x
        P3_x = self.P3_2(P3_x)

        return [P3_x, P4_x, P5_x]


class BackboneBase_VGG(nn.Module):
    def __init__(
        self,
        backbone: nn.Module,
        num_channels: int,
        name: str,
        return_interm_layers: bool,
    ):
        super().__init__()
        features = list(backbone.features.children())
        if return_interm_layers:
            if name == "vgg16_bn":
                self.body1 = nn.Sequential(*features[:13])
                self.body2 = nn.Sequential(*features[13:23])
                self.body3 = nn.Sequential(*features[23:33])
                self.body4 = nn.Sequential(*features[33:43])
        else:
            if name == "vgg16_bn":
                self.body = nn.Sequential(*features[:44])  # 16x down-sample
        self.num_channels = num_channels
        self.return_interm_layers = return_interm_layers
        self.fpn = FeatsFusion(
            256, 512, 512, hidden_size=num_channels, out_size=num_channels, out_kernel=3
        )

    def forward(self, tensor_list: NestedTensor):
        feats = []
        if self.return_interm_layers:
            xs = tensor_list.tensors
            for idx, layer in enumerate(
                [self.body1, self.body2, self.body3, self.body4]
            ):
                xs = layer(xs)
                feats.append(xs)

            # feature fusion
            features_fpn = self.fpn([feats[1], feats[2], feats[3]])
            features_fpn_4x = features_fpn[0]
            features_fpn_8x = features_fpn[1]

            # get tensor mask
            m = tensor_list.mask
            assert m is not None
            mask_4x = F.interpolate(
                m[None].float(), size=features_fpn_4x.shape[-2:]
            ).to(torch.bool)[0]
            mask_8x = F.interpolate(
                m[None].float(), size=features_fpn_8x.shape[-2:]
            ).to(torch.bool)[0]

            out: Dict[str, NestedTensor] = {}
            out["4x"] = NestedTensor(features_fpn_4x, mask_4x)
            out["8x"] = NestedTensor(features_fpn_8x, mask_8x)
        else:
            xs = self.body(tensor_list)
            out.append(xs)

        return out


class Backbone_VGG(BackboneBase_VGG):
    """
    VGG backbone
    """

    def __init__(self, name: str, return_interm_layers: bool):
        if name == "vgg16_bn":
            backbone = vgg16_bn(pretrained=True)
        num_channels = 256
        super().__init__(backbone, num_channels, name, return_interm_layers)


# 추가
##################################################################3
class BackboneBase_MobileNetv3(nn.Module):
    def __init__(
        self,
        backbone: nn.Module,
        num_channels: int,
        name: str,
        return_interm_layers: bool,
    ):
        super().__init__()
        features = list(backbone.features.children())
        if return_interm_layers:
            if name == "mobilenetv3":

                self.body1 = nn.Sequential(*features[:1])
                self.body2 = nn.Sequential(*features[1:3])
                self.body3 = nn.Sequential(*features[3:5])
                self.body4 = nn.Sequential(*features[5:7])
        else:
            if name == "mobilenetv3":
                self.body = nn.Sequential(*features[:7])
        self.num_channels = num_channels
        self.return_interm_layers = return_interm_layers
        self.fpn = FeatsFusion(
            24, 40, 40, hidden_size=num_channels, out_size=num_channels, out_kernel=3
        )

    def forward(self, tensor_list: NestedTensor):
        feats = []
        if self.return_interm_layers:
            xs = tensor_list.tensors
            for idx, layer in enumerate(
                [self.body1, self.body2, self.body3, self.body4]
            ):
                xs = layer(xs)
                feats.append(xs)

            # feature fusion
            features_fpn = self.fpn([feats[1], feats[2], feats[3]])
            features_fpn_4x = features_fpn[0]
            features_fpn_8x = features_fpn[1]

            # get tensor mask
            m = tensor_list.mask
            assert m is not None
            mask_4x = F.interpolate(
                m[None].float(), size=features_fpn_4x.shape[-2:]
            ).to(torch.bool)[0]
            mask_8x = F.interpolate(
                m[None].float(), size=features_fpn_8x.shape[-2:]
            ).to(torch.bool)[0]

            out: Dict[str, NestedTensor] = {}
            out["4x"] = NestedTensor(features_fpn_4x, mask_4x)
            out["8x"] = NestedTensor(features_fpn_8x, mask_8x)
        else:
            xs = self.body(tensor_list)
            out.append(xs)

        return out


class Backbone_MobileNetv3(BackboneBase_MobileNetv3):
    """
    MobileNetv3 backbone
    """

    def __init__(self, name: str, return_interm_layers: bool):
        if name == "mobilenetv3":
            backbone = mobilenetv3(pretrained=True)
        num_channels = 256
        super().__init__(backbone, num_channels, name, return_interm_layers)


#########################################################################################


class Joiner(nn.Sequential):
    def __init__(self, backbone, position_embedding):
        super().__init__(backbone, position_embedding)

    def forward(self, tensor_list: NestedTensor):
        xs = self[0](tensor_list)
        out: Dict[NestedTensor] = {}
        pos = {}
        for name, x in xs.items():
            out[name] = x
            # position encoding
            pos[name] = self[1](x).to(x.tensors.dtype)
        return out, pos


def build_backbone_vgg(args):
    position_embedding = build_position_encoding(args)
    backbone = Backbone_VGG(args.backbone, True)
    model = Joiner(backbone, position_embedding)
    model.num_channels = backbone.num_channels
    return model


def build_backbone_mobilenet(args):
    position_embedding = build_position_encoding(args)
    backbone = Backbone_MobileNetv3(args.backbone, True)
    model = Joiner(backbone, position_embedding)
    model.num_channels = backbone.num_channels
    return model


# 백본을 선택할 수 있기 때문에 build_backbone 메소드를 만들어 args.backbone를 확인한다.
def build_backbone(args):

    if args.backbone == "mobilenetv3":
        backbone = build_backbone_mobilenet(args)
    elif args.backbone == "vgg16_bn":
        backbone = build_backbone_vgg(args)
    else:
        raise ValueError("Unsupported backbone type")
    return backbone


if __name__ == "__main__":
    Backbone_VGG("vgg16_bn", True)
    Backbone_MobileNetv3("mobilenetv3", True)
