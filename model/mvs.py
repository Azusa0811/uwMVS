import torch
import torch.nn as nn
import torch.nn.functional as F
from lib.config import cfg
from .utils import *
from .feature_net import Unet
from .util.encodings import SHEncoding
# from .util.mlp import MLP
import cv2
import os
import numpy as np

class MVS(nn.Module):
    def __init__(self, hid_n=64, feat_ch=16+3, i=1):
        """
        """
        super(MVS, self).__init__()
        self.hid_n = hid_n
        self.agg = Agg(feat_ch)
        self.head_dim = 24
        self.Unet = Unet(self.head_dim, 16)
        self.level = i
        self.color = nn.Sequential(
            nn.Linear(feat_ch+self.head_dim+4, self.head_dim),
            nn.ReLU(),
            nn.Linear(self.head_dim, 1),
            nn.ReLU())
        self.direction_encoding = SHEncoding(levels=2, implementation="torch")
        self.colour_activation = nn.Sigmoid()
        self.sigma_activation = nn.Softplus()
        self.medium_head = nn.Sequential(
            nn.Linear(self.direction_encoding.get_out_dim(), 128),
            nn.ReLU(),
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Linear(64, 9),
        )
        self.register_buffer('depth_map', torch.zeros(50,cfg.train_dataset.input_h_w[0]*cfg.train_dataset.input_h_w[1],1))
        if cfg.mvs.wave_weight:
            self.wavelength_weights = nn.Parameter(
                torch.tensor([0.6, 0.8, 1.0]),
                requires_grad=True
            )

    def constrain_weights(self):
        weights = F.softplus(self.wavelength_weights)
        weights = weights / torch.max(weights)
        return weights

    def forward(self, vox_feat, img_feat_rgb_dir, z_vals, batch, size, level, src_dirs, tar_dir):
        H,W = size
        B, N_points, N_views = img_feat_rgb_dir.shape[:-1]
        S = img_feat_rgb_dir.shape[2]
        img_feat = self.agg(img_feat_rgb_dir)
        x = torch.cat((vox_feat, img_feat), dim=-1)

        d = z_vals.shape[-1]
        z_vals = z_vals.reshape(B,H,W,d)
        z_vals = torch.clamp_min(z_vals, 1e-6)
        if cfg.mvs.cas_config.depth_inv[level]:
            z_vals = 1./torch.clamp_min(z_vals, 1e-6)
        depth_medium = z_vals.reshape(B,H*W,1) / cfg.mvsg.scale_factor
        self.depth_map[batch['tar_id']] = depth_medium.detach()
        depth_medium = ((depth_medium - depth_medium.min()) / (depth_medium.max() - depth_medium.min() + 1e-20))
        depth_medium = depth_medium.reshape(B,H,W,1)

        raw2alpha = lambda raw: 1.-torch.exp(-raw)

        # radiance head
        x0 = x.unsqueeze(2).repeat(1,1,S,1)
        x0 = torch.cat((x0, img_feat_rgb_dir), dim=-1)
        color_weight = F.softmax(self.color(x0), dim=-2)
        radiance = torch.sum((img_feat_rgb_dir[..., -7:-4] * color_weight), dim=-2)
        if cfg.mvsgs.wave_weight:
            atten_weights = self.constrain_weights()
        src_clear_rgbs = []
        for i in range(S):
            src_dir_flat = src_dirs[:,i].view(-1,3)
            src_dir_enc = self.direction_encoding(src_dir_flat)
            src_depth = self.depth_map[batch['src_ids'][i]]
            src_medium_base_out = self.medium_head(src_dir_enc.float()).to(x.device)
            src_medium_rgb = (
                self.colour_activation(src_medium_base_out[..., :3])
                .view(B,H*W,3)
                .to(x.device)
            )
            src_medium_bs = (
                self.sigma_activation(src_medium_base_out[..., 3:6])
                .view(B,H*W,3)
                .to(x.device)
            )
            src_medium_atten = (
                self.sigma_activation(src_medium_base_out[..., 6:])
                .view(B,H*W,3)
                .to(x.device)
            )
            if cfg.mvs.wave_weight:
                src_atten = torch.exp(-src_medium_atten * src_depth) * atten_weights.view(1, 1, 3)
            else:
                src_atten = torch.exp(-src_medium_atten * src_depth)
            src_alpha_bs = raw2alpha(src_medium_bs * src_depth)
            src_rgb_bs =   src_medium_rgb * src_alpha_bs
            src_clear_rgb = (img_feat_rgb_dir[..., i,-7:-4]  - src_rgb_bs) / src_atten
            src_clear_rgb = torch.clamp(src_clear_rgb, 0., 1.)
            src_clear_rgbs.append(src_clear_rgb)
        src_clear_rgbs = torch.stack(src_clear_rgbs, dim=-2)
        rgb_obj = torch.sum((src_clear_rgbs * color_weight), dim=-2)
        tar_dir_flat = tar_dir.view(-1, 3)
        tar_dir_enc = self.direction_encoding(tar_dir_flat)
        tar_depth = self.depth_map[batch['tar_id']]
        medium_base_out = self.medium_head(tar_dir_enc.float()).to(x.device)
        medium_rgb = (
            self.colour_activation(medium_base_out[..., :3])
            .view(B,H*W,3)
            .to(x.device)
        )
        medium_bs = (
            self.sigma_activation(medium_base_out[..., 3:6])
            .view(B,H*W,3)
            .to(x.device)
        )
        medium_atten = (
            self.sigma_activation(medium_base_out[..., 6:])
            .view(B,H*W,3)
            .to(x.device)
        )
        if cfg.mvs.wave_weight:
            atten = torch.exp(-medium_atten * tar_depth) * atten_weights.view(1, 1, 3)
        else:
            atten = torch.exp(-medium_atten * tar_depth)
        alpha_bs = raw2alpha(medium_bs * tar_depth)
        rgb_bs =  medium_rgb * alpha_bs
        rgb_atten = rgb_obj * atten
        rgb_vr = rgb_atten + rgb_bs
        rgb_vr = torch.clamp(rgb_vr, 0., 1.)
        
        rgb_obj = rgb_obj.reshape(B,H,W,3).permute(0,3,1,2)
        rgb_atten = rgb_atten.reshape(B,H,W,3).permute(0,3,1,2)
        rgb_bs = rgb_bs.reshape(B,H,W,3).permute(0,3,1,2)
        rgb_vr = rgb_vr.reshape(B,H,W,3).permute(0,3,1,2)

        return rgb_vr, rgb_bs, rgb_obj, rgb_atten, depth_medium



class Agg(nn.Module):
    def __init__(self, feat_ch):
        """
        """
        super(Agg, self).__init__()
        self.feat_ch = feat_ch
        if cfg.mvsgs.viewdir_agg:
            self.view_fc = nn.Sequential(
                    nn.Linear(4, feat_ch),
                    nn.ReLU(),
                    )
            self.view_fc.apply(weights_init)
        self.global_fc = nn.Sequential(
                nn.Linear(feat_ch*3, 32),
                nn.ReLU(),
                )

        self.agg_w_fc = nn.Sequential(
                nn.Linear(32, 1),
                nn.ReLU(),
                )
        self.fc = nn.Sequential(
                nn.Linear(32, 16),
                nn.ReLU(),
                )
        self.global_fc.apply(weights_init)
        self.agg_w_fc.apply(weights_init)
        self.fc.apply(weights_init)

    def forward(self, img_feat_rgb_dir):
        B, S = len(img_feat_rgb_dir), img_feat_rgb_dir.shape[-2]
        if cfg.mvsgs.viewdir_agg:
            view_feat = self.view_fc(img_feat_rgb_dir[..., -4:])
            img_feat_rgb =  img_feat_rgb_dir[..., :-4] + view_feat
        else:
            img_feat_rgb =  img_feat_rgb_dir[..., :-4]

        var_feat = torch.var(img_feat_rgb, dim=-2).view(B, -1, 1, self.feat_ch).repeat(1, 1, S, 1)
        avg_feat = torch.mean(img_feat_rgb, dim=-2).view(B, -1, 1, self.feat_ch).repeat(1, 1, S, 1)

        feat = torch.cat([img_feat_rgb, var_feat, avg_feat], dim=-1)
        global_feat = self.global_fc(feat)
        agg_w = F.softmax(self.agg_w_fc(global_feat), dim=-2)
        im_feat = (global_feat * agg_w).sum(dim=-2)
        return self.fc(im_feat)



def weights_init(m):
    if isinstance(m, nn.Linear):
        nn.init.kaiming_normal_(m.weight.data)
        if m.bias is not None:
            nn.init.zeros_(m.bias.data)
