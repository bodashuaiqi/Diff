import math
from modulefinder import Module
import warnings
import copy
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from mmcv.cnn import (Conv2d, build_activation_layer, build_norm_layer,
                      constant_init, normal_init, trunc_normal_init,build_conv_layer)
from mmcv.cnn.bricks.drop import build_dropout
from mmcv.cnn.bricks.transformer import MultiheadAttention
from mmcv.runner import (BaseModule, ModuleList, Sequential, _load_checkpoint,
                         load_state_dict)
from torch.nn.modules.utils import _pair as to_2tuple

# from pretraining.mmpose.utils import get_root_logger
# from pretraining.mmpose.models.builder import BACKBONES
# from pretraining.mmpose.models.utils import PatchEmbed_new, nchw_to_nlc, nlc_to_nchw, pvt_convert
# from pretraining.mmpose.models.backbones.utils import load_checkpoint
from collections import OrderedDict
from functools import partial
from typing import Optional, Union

import torch
import torch.nn as nn
from einops.layers.torch import Rearrange
# from fairscale.nn.checkpoint import checkpoint_wrapper
from timm.models import register_model
from timm.models.layers import DropPath, to_2tuple, trunc_normal_
from timm.models.vision_transformer import _cfg
from torch.nn import init

# from .bra_legacy import BiLevelRoutingAttention
# from .common import Attention, AttentionLePE, DWConv


def get_pe_layer(emb_dim, pe_dim=None, name='none'):
    if name == 'none':
        return nn.Identity()
    # if name == 'sum':
    #     return Summer(PositionalEncodingPermute2D(emb_dim))
    # elif name == 'npe.sin':
    #     return NeuralPE(emb_dim=emb_dim, pe_dim=pe_dim, mode='sin')
    # elif name == 'npe.coord':
    #     return NeuralPE(emb_dim=emb_dim, pe_dim=pe_dim, mode='coord')
    # elif name == 'hpe.conv':
    #     return HybridPE(emb_dim=emb_dim, pe_dim=pe_dim, mode='conv', res_shortcut=True)
    # elif name == 'hpe.dsconv':
    #     return HybridPE(emb_dim=emb_dim, pe_dim=pe_dim, mode='dsconv', res_shortcut=True)
    # elif name == 'hpe.pointconv':
    #     return HybridPE(emb_dim=emb_dim, pe_dim=pe_dim, mode='pointconv', res_shortcut=True)
    else:
        raise ValueError(f'PE name {name} is not surpported!')
    
class ChannelAttention(nn.Module):
    def __init__(self, in_planes, ratio=16):
        super(ChannelAttention, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.max_pool = nn.AdaptiveMaxPool2d(1)
        self.mlp=nn.Sequential(
            nn.Conv2d(in_planes, in_planes // ratio, 1, bias=False),
            nn.ReLU(),
            nn.Conv2d(in_planes // ratio, in_planes, 1, bias=False)
        )
        self.sigmoid = nn.Sigmoid()
 
    def forward(self, x):
        avg_out = self.mlp(self.avg_pool(x))  
        max_out = self.mlp(self.max_pool(x))  
        out = avg_out + max_out
        return self.sigmoid(out)
 
 
class SpatialAttention(nn.Module):
    def __init__(self, kernel_size=7):
        super(SpatialAttention, self).__init__()
 
        assert kernel_size in (3, 7), 'kernel size must be 3 or 7'
        padding = 3 if kernel_size == 7 else 1
        self.conv1 = nn.Conv2d(2, 1, kernel_size, padding=padding, bias=False)
        self.sigmoid = nn.Sigmoid()
 
    def forward(self, x):
        avg_out = torch.mean(x, dim=1, keepdim=True)    
        max_out, _ = torch.max(x, dim=1, keepdim=True)  
        x = torch.cat([avg_out, max_out], dim=1)     
        x = self.conv1(x)                         
        return self.sigmoid(x)
 
class CBAM(nn.Module):
    def __init__(self, in_planes, ratio=16, kernel_size=7):
        super(CBAM, self).__init__()
        self.ca = ChannelAttention(in_planes, ratio)
        self.sa = SpatialAttention(kernel_size)
 
    def init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                init.kaiming_normal_(m.weight, mode='fan_out')
                if m.bias is not None:
                    init.constant_(m.bias, 0)
            elif isinstance(m, nn.BatchNorm2d):
                init.constant_(m.weight, 1)
                init.constant_(m.bias, 0)
            elif isinstance(m, nn.Linear):
                init.normal_(m.weight, std=0.001)
                if m.bias is not None:
                    init.constant_(m.bias, 0)
 
    def forward(self, x):
        out = x * self.ca(x)                
        result = out * self.sa(out)          
        return result
class BasicBlock(nn.Module):
    """BasicBlock for ResNet.

    Args:
        in_channels (int): Input channels of this block.
        out_channels (int): Output channels of this block.
        expansion (int): The ratio of ``out_channels/mid_channels`` where
            ``mid_channels`` is the output channels of conv1. This is a
            reserved argument in BasicBlock and should always be 1. Default: 1.
        stride (int): stride of the block. Default: 1
        dilation (int): dilation of convolution. Default: 1
        downsample (nn.Module): downsample operation on identity branch.
            Default: None.
        style (str): `pytorch` or `caffe`. It is unused and reserved for
            unified API with Bottleneck.
        with_cp (bool): Use checkpoint or not. Using checkpoint will save some
            memory while slowing down the training speed.
        conv_cfg (dict): dictionary to construct and config conv layer.
            Default: None
        norm_cfg (dict): dictionary to construct and config norm layer.
            Default: dict(type='BN')
    """

    def __init__(self,
                 in_channels,
                 out_channels,
                 expansion=1,
                 stride=1,
                 dilation=1,
                 downsample=None,
                 style='pytorch',
                 with_cp=False,
                 conv_cfg=None,
                 norm_cfg=dict(type='BN')):
        # Protect mutable default arguments
        norm_cfg = copy.deepcopy(norm_cfg)
        super().__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.expansion = expansion
        assert self.expansion == 1
        assert out_channels % expansion == 0
        self.mid_channels = out_channels // expansion
        self.stride = stride
        self.dilation = dilation
        self.style = style
        self.with_cp = with_cp
        self.conv_cfg = conv_cfg
        self.norm_cfg = norm_cfg
        self.CBAM = CBAM(in_planes=in_channels)

        self.norm1_name, norm1 = build_norm_layer(
            norm_cfg, self.mid_channels, postfix=1)
        self.norm2_name, norm2 = build_norm_layer(
            norm_cfg, out_channels, postfix=2)

        self.conv1 = build_conv_layer(
            conv_cfg,
            in_channels,
            self.mid_channels,
            3,
            stride=1,
            padding=1,
            dilation=1,
            bias=False)
        self.add_module(self.norm1_name, norm1)
        self.conv2 = build_conv_layer(
            conv_cfg,
            self.mid_channels,
            out_channels,
            kernel_size=3,  
            stride=1,
            padding=2,
            dilation=2,
            bias=False)
        self.add_module(self.norm2_name, norm2)
        self.conv4 = build_conv_layer(
            conv_cfg,
            in_channels,
            out_channels,
            kernel_size=3,
            stride=1,
            padding=1,
            dilation=1,
            bias=False)

        self.relu = nn.ReLU(inplace=True)
        self.downsample = downsample

    @property
    def norm1(self):
        """nn.Module: the normalization layer named "norm1" """
        return getattr(self, self.norm1_name)

    @property
    def norm2(self):
        """nn.Module: the normalization layer named "norm2" """
        return getattr(self, self.norm2_name)

    def forward(self, x):
        """Forward function."""

        def _inner_forward(x):
            identity = self.conv4(x)

            out = self.conv1(x)
            out = self.norm1(out)
            out = self.relu(out)

            out = self.conv2(out)
            out = self.norm2(out)
            #CBAM Block
            out = self.CBAM(out)          
            out += identity
            
            return out

        if self.with_cp and x.requires_grad:
            pass
            # out = cp.checkpoint(_inner_forward, x)
        else:
            out = _inner_forward(x)

        out = self.relu(out)

        return out


# class DWConv(nn.Module):
#     def __init__(self, dim=768):
#         super(DWConv, self).__init__()
#         self.dwconv = nn.Conv2d(dim, dim, 3, 1, 1, bias=True, groups=dim)

#     def forward(self, x):
#         """
#         x: NHWC tensor
#         """
#         x = x.permute(0, 3, 1, 2) #NCHW
#         x = self.dwconv(x)
#         x = x.permute(0, 2, 3, 1) #NHWC

#         return x

# class Attention(nn.Module):
#     """
#     vanilla attention
#     """
#     def __init__(self, dim, num_heads=8, qkv_bias=False, qk_scale=None, attn_drop=0., proj_drop=0.):
#         super().__init__()
#         self.num_heads = num_heads
#         head_dim = dim // num_heads
#         # NOTE scale factor was wrong in my original version, can set manually to be compat with prev weights
#         self.scale = qk_scale or head_dim ** -0.5

#         self.qkv = nn.Linear(dim, dim * 3, bias=qkv_bias)
#         self.attn_drop = nn.Dropout(attn_drop)
#         self.proj = nn.Linear(dim, dim)
#         self.proj_drop = nn.Dropout(proj_drop)

#     def forward(self, x):
#         """
#         args:
#             x: NHWC tensor
#         return:
#             NHWC tensor
#         """
#         _, H, W, _ = x.size()
#         x = rearrange(x, 'n h w c -> n (h w) c')
        
#         #######################################
#         B, N, C = x.shape        
#         qkv = self.qkv(x).reshape(B, N, 3, self.num_heads, C // self.num_heads).permute(2, 0, 3, 1, 4)
#         q, k, v = qkv[0], qkv[1], qkv[2]   # make torchscript happy (cannot use tensor as tuple)

#         attn = (q @ k.transpose(-2, -1)) * self.scale
#         attn = attn.softmax(dim=-1)
#         attn = self.attn_drop(attn)

#         x = (attn @ v).transpose(1, 2).reshape(B, N, C)
#         x = self.proj(x)
#         x = self.proj_drop(x)
#         #######################################

#         x = rearrange(x, 'n (h w) c -> n h w c', h=H, w=W)
#         return x

# class AttentionLePE(nn.Module):
#     """
#     vanilla attention
#     """
#     def __init__(self, dim, num_heads=8, qkv_bias=False, qk_scale=None, attn_drop=0., proj_drop=0., side_dwconv=5):
#         super().__init__()
#         self.num_heads = num_heads
#         head_dim = dim // num_heads
#         # NOTE scale factor was wrong in my original version, can set manually to be compat with prev weights
#         self.scale = qk_scale or head_dim ** -0.5

#         self.qkv = nn.Linear(dim, dim * 3, bias=qkv_bias)
#         self.attn_drop = nn.Dropout(attn_drop)
#         self.proj = nn.Linear(dim, dim)
#         self.proj_drop = nn.Dropout(proj_drop)
#         self.lepe = nn.Conv2d(dim, dim, kernel_size=side_dwconv, stride=1, padding=side_dwconv//2, groups=dim) if side_dwconv > 0 else \
#                     lambda x: torch.zeros_like(x)

#     def forward(self, x):
#         """
#         args:
#             x: NHWC tensor
#         return:
#             NHWC tensor
#         """
#         _, H, W, _ = x.size()
#         x = rearrange(x, 'n h w c -> n (h w) c')
        
#         #######################################
#         B, N, C = x.shape        
#         qkv = self.qkv(x).reshape(B, N, 3, self.num_heads, C // self.num_heads).permute(2, 0, 3, 1, 4)
#         q, k, v = qkv[0], qkv[1], qkv[2]   # make torchscript happy (cannot use tensor as tuple)

#         lepe = self.lepe(rearrange(x, 'n (h w) c -> n c h w', h=H, w=W))
#         lepe = rearrange(lepe, 'n c h w -> n (h w) c')

#         attn = (q @ k.transpose(-2, -1)) * self.scale
#         attn = attn.softmax(dim=-1)
#         attn = self.attn_drop(attn)

#         x = (attn @ v).transpose(1, 2).reshape(B, N, C)
#         x = x + lepe

#         x = self.proj(x)
#         x = self.proj_drop(x)
#         #######################################

#         x = rearrange(x, 'n (h w) c -> n h w c', h=H, w=W)
#         return x



# class nchwAttentionLePE(nn.Module):
#     """
#     Attention with LePE, takes nchw input
#     """
#     def __init__(self, dim, num_heads=8, qkv_bias=False, qk_scale=None, attn_drop=0., proj_drop=0., side_dwconv=5):
#         super().__init__()
#         self.num_heads = num_heads
#         self.head_dim = dim // num_heads
#         self.scale = qk_scale or self.head_dim ** -0.5

#         self.qkv = nn.Conv2d(dim, dim*3, kernel_size=1, bias=qkv_bias)
#         self.attn_drop = nn.Dropout(attn_drop)
#         self.proj = nn.Conv2d(dim, dim, kernel_size=1)
#         self.proj_drop = nn.Dropout(proj_drop)
#         self.lepe = nn.Conv2d(dim, dim, kernel_size=side_dwconv, stride=1, padding=side_dwconv//2, groups=dim) if side_dwconv > 0 else \
#                     lambda x: torch.zeros_like(x)

#     def forward(self, x:torch.Tensor):
#         """
#         args:
#             x: NCHW tensor
#         return:
#             NCHW tensor
#         """
#         B, C, H, W = x.size()
#         q, k, v = self.qkv.forward(x).chunk(3, dim=1) # B, C, H, W

#         attn = q.view(B, self.num_heads, self.head_dim, H*W).transpose(-1, -2) @ \
#                k.view(B, self.num_heads, self.head_dim, H*W)
#         attn = torch.softmax(attn*self.scale, dim=-1)
#         attn = self.attn_drop(attn)

#         # (B, nhead, HW, HW) @ (B, nhead, HW, head_dim) -> (B, nhead, HW, head_dim)
#         output:torch.Tensor = attn @ v.view(B, self.num_heads, self.head_dim, H*W).transpose(-1, -2)
#         output = output.permute(0, 1, 3, 2).reshape(B, C, H, W)
#         output = output + self.lepe(v)

#         output = self.proj_drop(self.proj(output))

#         return output
