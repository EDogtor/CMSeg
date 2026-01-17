import torch
import torch.nn as nn
import torch.nn.functional as F
import matplotlib.pyplot as plt
from functools import partial
from .local_vmamba.region_mamba import *
from timm.models.layers import DropPath, to_2tuple, trunc_normal_
import math
import time
from utils.logger import get_logger
from .vmamba import Backbone_VSSM, CrossMambaFusionBlock, ConcatMambaFusionBlock
from .mamba_net_utils import ChannelRectifyModule
logger = get_logger()


class AdaptiveFusionModule(nn.Module):
    """自适应模态融合模块 - 学习CT和PET的最优融合权重"""
    def __init__(self, dim):
        super().__init__()
        # 学习融合权重
        self.weight_net = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(dim * 2, max(dim // 4, 16), 1),
            nn.ReLU(),
            nn.Conv2d(max(dim // 4, 16), 2, 1),
            nn.Softmax(dim=1)
        )
    
    def forward(self, feat_rgb, feat_x):
        """
        Args:
            feat_rgb: CT特征 [B, C, H, W]
            feat_x: PET特征 [B, C, H, W]
        Returns:
            融合后的特征 [B, C, H, W]
        """
        concat_feat = torch.cat([feat_rgb, feat_x], dim=1)  # [B, 2C, H, W]
        weights = self.weight_net(concat_feat)  # [B, 2, 1, 1]
        w_rgb, w_x = weights[:, 0:1], weights[:, 1:2]
        # 自适应加权融合
        fused = w_rgb * feat_rgb + w_x * feat_x
        return fused


class RGBXTransformer(nn.Module):
    def __init__(self,
                 num_classes=1000,
                 norm_layer=nn.LayerNorm,
                 depths=[2,2,27,2], # [2,2,27,2] for vmamba small
                 dims=96,
                 pretrained=None,
                 mlp_ratio=4.0,
                 downsample_version='v1',
                 ape=False,
                 img_size=[512, 512],
                 patch_size=4,
                 drop_path_rate=0.2,
                 **kwargs):
        super().__init__()

        self.ape = ape
        #33M左右
        # 共享权重版本: self.vssm = Backbone_VSSM(...)
        
        # 不共享权重版本：CT和PET各一个Backbone
        # 使用3通道输入以充分利用预训练权重（与CIPA一致）
        self.vssm_rgb = Backbone_VSSM(
            in_chans=3, # CT channel (will repeat 1ch to 3ch)
            pretrained=pretrained,
            norm_layer=norm_layer,
            num_classes=num_classes,
            depths=depths,
            dims=dims,
            mlp_ratio=mlp_ratio,
            downsample_version=downsample_version,
            drop_path_rate=drop_path_rate,
        )
        
        self.vssm_x = Backbone_VSSM(
            in_chans=3, # PET channel (will repeat 1ch to 3ch)
            pretrained=pretrained,
            norm_layer=norm_layer,
            num_classes=num_classes,
            depths=depths,
            dims=dims,
            mlp_ratio=mlp_ratio,
            downsample_version=downsample_version,
            drop_path_rate=drop_path_rate,
        )

        self.cross_mamba=nn.ModuleList(
            ChannelRectifyModule(dim=dims * (2 ** i),HW=(128*128/((2 ** i)*(2 ** i))),reduction=16)   #(128*128/((2 ** i)*(2 ** i)))
            for i in range(4)
            )
        self.channel_attn_mamba = nn.ModuleList(
            Region_global_Block(
                outer_dim=dims * (2 ** i), inner_dim=dims * (2 ** i)
                ,num_words=16,drop_path=0)for i in range(4)  #num_words:4*4 small region
        )
        self.region_patch = nn.ModuleList(
            Stem(inner_dim=dims * (2 ** i),outer_dim=dims * (2 ** i))
            for i in range(4)
        )
        # 添加自适应融合模块
        self.adaptive_fusion = nn.ModuleList(
            AdaptiveFusionModule(dim=dims * (2 ** i))
            for i in range(4)
        )
        if self.ape:
            self.patches_resolution = [img_size[0] // patch_size, img_size[1] // patch_size]
            self.absolute_pos_embed = []
            self.absolute_pos_embed_x = []
            for i_layer in range(len(depths)):
                input_resolution=(self.patches_resolution[0] // (2 ** i_layer),
                                      self.patches_resolution[1] // (2 ** i_layer))
                dim=int(dims * (2 ** i_layer))
                absolute_pos_embed = nn.Parameter(torch.zeros(1, dim, input_resolution[0], input_resolution[1]))
                trunc_normal_(absolute_pos_embed, std=.02)
                absolute_pos_embed_x = nn.Parameter(torch.zeros(1, dim, input_resolution[0], input_resolution[1]))
                trunc_normal_(absolute_pos_embed_x, std=.02)

                self.absolute_pos_embed.append(absolute_pos_embed)
                self.absolute_pos_embed_x.append(absolute_pos_embed_x)

    def forward_features(self, x_rgb, x_e):
        """
        x_rgb: B x 1 x H x W  #ct (单通道)
        x_e: B x 1 x H x W    #pet (单通道)
        """
        B = x_rgb.shape[0]
        outs_fused = []

        # 将单通道输入复制到3通道（与CIPA一致，充分利用预训练权重）
        if x_rgb.shape[1] == 1:
            x_rgb = x_rgb.repeat(1, 3, 1, 1)  # [B,1,H,W] -> [B,3,H,W]
        if x_e.shape[1] == 1:
            x_e = x_e.repeat(1, 3, 1, 1)      # [B,1,H,W] -> [B,3,H,W]

        # 不共享权重：分别使用各自的backbone
        outs_rgb = self.vssm_rgb(x_rgb) # B x C x H x W
        outs_x = self.vssm_x(x_e) # B x C x H x W

        for i in range(4):
            if self.ape:
                # this has been discarded
                out_rgb = self.absolute_pos_embed[i].to(outs_rgb[i].device) + outs_rgb[i]
                out_x = self.absolute_pos_embed_x[i].to(outs_x[i].device) + outs_x[i]
            else:
                out_rgb = outs_rgb[i]
                out_x = outs_x[i]
            #CRM
            CRM = True
            #DCIM
            DCIM = True
            if CRM and DCIM:
                cross_rgb, cross_x = self.cross_mamba[i](out_rgb,out_x)
                cross_rgb,cross_x,(H_out, W_out), (H_in, W_in) = self.region_patch[i](cross_rgb,cross_x)
                # 使用自适应融合替代简单相加
                attn_output = self.channel_attn_mamba[i](cross_rgb.contiguous(), cross_x.contiguous(),H_out, W_out,H_in, W_in).permute(0, 3, 1, 2).contiguous()
                adaptive_fused = self.adaptive_fusion[i](out_rgb, out_x)
                x_fuse = adaptive_fused + attn_output
            elif not DCIM and CRM:
                out_rgb,  out_x = self.cross_mamba[i](out_rgb, out_x)
                x_fuse = (out_rgb + out_x)
            elif DCIM and not CRM:
                cross_rgb = out_rgb
                cross_x = out_x
                cross_rgb, cross_x, (H_out, W_out), (H_in, W_in) = self.region_patch[i](cross_rgb, cross_x)
                x_fuse = out_rgb+out_x+self.channel_attn_mamba[i](cross_rgb.contiguous(), cross_x.contiguous(),
                                                                    H_out, W_out, H_in, W_in).permute(0, 3, 1,2).contiguous()

            elif not DCIM and not CRM:
                x_fuse = (out_rgb + out_x)
            outs_fused.append(x_fuse)
        return outs_fused

    def forward(self, x_rgb, x_e):
        out = self.forward_features(x_rgb, x_e)
        return out

    def init_weights(self, pretrained=None):
        if pretrained is not None:
            self.vssm_rgb.load_pretrained(pretrained)
            self.vssm_x.load_pretrained(pretrained)

class vssm_tiny(RGBXTransformer):
    def __init__(self, fuse_cfg=None, **kwargs):
        super(vssm_tiny, self).__init__(
            depths=[2, 2, 9, 2],
            dims=96,
            pretrained='./pretrained/vmamba/vssmtiny_dp01_ckpt_epoch_292.pth',
            mlp_ratio=0.0,
            downsample_version='v1',
            drop_path_rate=0,
        )

class vssm_small(RGBXTransformer):
    def __init__(self, fuse_cfg=None, **kwargs):
        super(vssm_small, self).__init__(
            depths=[2, 2, 27, 2],
            dims=96,
            pretrained='pretrained/vmamba/vssmsmall_dp03_ckpt_epoch_238.pth',
            mlp_ratio=0.0,
            downsample_version='v1',
            drop_path_rate=0.3,
        )

class vssm_base(RGBXTransformer):
    def __init__(self, fuse_cfg=None, **kwargs):
        super(vssm_base, self).__init__(
            depths=[2, 2, 27, 2],
            dims=128,
            pretrained='pretrained/vmamba/vssmbase_dp06_ckpt_epoch_241.pth',
            mlp_ratio=0.0,
            downsample_version='v1',
            drop_path_rate=0.6, # VMamba-B with droppath 0.5 + no ema. VMamba-B* represents for VMamba-B with droppath 0.6 + ema
        )