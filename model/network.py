import os

import cv2
import imageio
import numpy as np
import PIL
import torch
import torch.nn as nn
from torch.nn import functional as F

from lib.config import cfg

from . import utils
from .cost_reg_net import CostRegNet, MinCostRegNet, DSCVolumeReg
from .feature_net import FeatureNet
from .mvs import MVS
from .utils import save_pfm, visualize_depth, write_cam


class Network(nn.Module):
    
    def __init__(self,):
        super(Network, self).__init__()
        self.feature_net = FeatureNet()
        for i in range(cfg.mvs.cas_config.num):
            if i == 0:
                cost_reg_l = MinCostRegNet(int(32 * (2**(-i))*2))
            else:
                cost_reg_l = CostRegNet(int(32 * (2**(-i)*2)))
            setattr(self, f'cost_reg_{i}', cost_reg_l)
            mvs_l = MVS(feat_ch=cfg.mvs.cas_config.gs_model_feat_ch[i]+3, i=i)
            setattr(self, f'mvs_{i}', mvs_l)

    def render_rays(self, rays, **kwargs):
        level, batch, im_feat, feat_volume, gs_model, size = kwargs['level'], kwargs['batch'], kwargs['im_feat'], kwargs['feature_volume'], kwargs['gs_model'], kwargs['size']
        world_xyz, uvd, z_vals = utils.sample_along_depth(rays, N_samples=cfg.mvs.cas_config.num_samples[level], level=level)
        B, N_rays, N_samples = world_xyz.shape[:3]
        rgbs = utils.unpreprocess(batch['src_inps'], render_scale=cfg.mvs.cas_config.render_scale[level])
        up_feat_scale = cfg.mvs.cas_config.render_scale[level] / cfg.mvs.cas_config.im_ibr_scale[level]
        if up_feat_scale != 1.:
            B, S, C, H, W = im_feat.shape
            im_feat = F.interpolate(im_feat.reshape(B*S, C, H, W), None, scale_factor=up_feat_scale, align_corners=True, mode='bilinear').view(B, S, C, int(H*up_feat_scale), int(W*up_feat_scale))

        img_feat_rgb = torch.cat((im_feat, rgbs), dim=2)
        H_O, W_O = kwargs['batch']['src_inps'].shape[-2:]
        B, H, W = len(uvd), int(H_O * cfg.mvs.cas_config.render_scale[level]), int(W_O * cfg.mvs.cas_config.render_scale[level])
        uvd[..., 0], uvd[..., 1] = (uvd[..., 0]) / (W-1), (uvd[..., 1]) / (H-1)
        vox_feat = utils.get_vox_feat(uvd.reshape(B, -1, 3), feat_volume)
        img_feat_rgb_dir, src_dirs, tar_dir = utils.get_img_feat(world_xyz, img_feat_rgb, batch, self.training, level)
        net_output = gs_model(vox_feat, img_feat_rgb_dir, z_vals, batch, size, level, src_dirs, tar_dir) 
        return net_output


    def batchify_rays(self, rays, **kwargs):
        ret = self.render_rays(rays, **kwargs)
        return ret


    def forward_feat(self, x):
        B, S, C, H, W = x.shape
        x = x.view(B*S, C, H, W)
        feat2, feat1, feat0 = self.feature_net(x)
        feats = {
                'level_2': feat0.reshape((B, S, feat0.shape[1], H, W)),
                'level_1': feat1.reshape((B, S, feat1.shape[1], H//2, W//2)),
                'level_0': feat2.reshape((B, S, feat2.shape[1], H//4, W//4)),
                }
        return feats

    def forward_render(self, ret, batch):
        B, _, _, H, W = batch['src_inps'].shape
        rgb = ret['rgb'].reshape(B, H, W, 3).permute(0, 3, 1, 2)
        rgb = self.cnn_renderer(rgb)
        ret['rgb'] = rgb.permute(0, 2, 3, 1).reshape(B, H*W, 3)


    def forward(self, batch):
        B, _, _, H_img, W_img = batch['src_inps'].shape
        if not cfg.save_video:
            feats = self.forward_feat(batch['src_inps'])
            ret = {}
            depth, std, near_far = None, None, None
            for i in range(cfg.mvs.cas_config.num):
                H, W = int(H_img*cfg.mvs.cas_config.render_scale[i]), int(W_img*cfg.mvs.cas_config.render_scale[i])
                feature_volume, depth_values, near_far = utils.build_feature_volume( 
                        feats[f'level_{i}'],
                        batch,
                        D=cfg.mvs.cas_config.volume_planes[i],
                        depth=depth,
                        std=std,
                        near_far=near_far,
                        level=i)
                feature_volume, depth_prob = getattr(self, f'cost_reg_{i}')(feature_volume)
                depth, std = utils.depth_regression(depth_prob, depth_values, i, batch)
                if not cfg.mvs.cas_config.render_if[i]:
                    continue
                rays = utils.build_rays(depth, std, batch, self.training, near_far, i)
                im_feat_level = cfg.mvs.cas_config.render_im_feat_level[i]
                output = self.batchify_rays(
                        rays=rays,
                        feature_volume=feature_volume,
                        batch=batch,
                        im_feat=feats[f'level_{im_feat_level}'],
                        gs_model=getattr(self, f'mvs_{i}'),
                        level=i,
                        size=(H,W)
                        )
                ret_i = {}
                rgb_vr, rgb_bs, rgb_obj, rgb_atten, depth_medium = output
                render_novel = []
                render_atten = []
                render_bs = []
                render_obj = []
                render_depth = []
                for b_i in range(B):
                    if cfg.mvs.reweighting: 
                        render_novel_i = rgb_vr[b_i]
                    else:
                        render_novel_i = rgb_vr[b_i]
                    render_novel.append(render_novel_i)
                    render_atten.append(rgb_atten[b_i])
                    render_bs.append(rgb_bs[b_i])
                    render_obj.append(rgb_obj[b_i])
                    render_depth.append(depth_medium[b_i])
                render_novel = torch.stack(render_novel)
                render_atten = torch.stack(render_atten)    
                render_bs = torch.stack(render_bs)
                render_obj = torch.stack(render_obj)
                render_depth = torch.stack(render_depth)
                ret_i.update({'rgb': render_novel.flatten(2).permute(0,2,1)})
                ret_i.update({'rgb_atten': render_atten.flatten(2).permute(0,2,1)})
                ret_i.update({'rgb_bs': render_bs.flatten(2).permute(0,2,1)})
                ret_i.update({'rgb_obj': render_obj.flatten(2).permute(0,2,1)})
                ret_i.update({'rgb_depth': render_depth})
                if cfg.mvs.cas_config.depth_inv[i]:
                    ret_i.update({'depth_mvs': 1./depth})
                else:
                    ret_i.update({'depth_mvs': depth})
                ret_i.update({'std': std})
                if ret_i['rgb'].isnan().any():
                    __import__('ipdb').set_trace()
                ret.update({key+f'_level{i}': ret_i[key] for key in ret_i})
                
        else:
            pred_rgb_nb_list = []
            pred_obj_list = []
            for v_i, meta in enumerate(batch['rendering_video_meta']):
                batch['tar_ext'][:,:3] = meta['tar_ext'][:,:3]
                batch['rays_0'] = meta['rays_0']
                batch['rays_1'] = meta['rays_1']
                feats = self.forward_feat(batch['src_inps'])
                depth, std, near_far = None, None, None
                for i in range(cfg.mvs.cas_config.num):
                    H, W = int(H_img*cfg.mvs.cas_config.render_scale[i]), int(W_img*cfg.mvs.cas_config.render_scale[i])
                    feature_volume, depth_values, near_far = utils.build_feature_volume(
                            feats[f'level_{i}'],
                            batch,
                            D=cfg.mvs.cas_config.volume_planes[i],
                            depth=depth,
                            std=std,
                            near_far=near_far,
                            level=i)
                    feature_volume, depth_prob = getattr(self, f'cost_reg_{i}')(feature_volume)
                    depth, std = utils.depth_regression(depth_prob, depth_values, i, batch)
                    if not cfg.mvs.cas_config.render_if[i]:
                        continue
                    rays = utils.build_rays(depth, std, batch, self.training, near_far, i)
                    im_feat_level = cfg.mvs.cas_config.render_im_feat_level[i]
                    output = self.batchify_rays(
                            rays=rays,
                            feature_volume=feature_volume,
                            batch=batch,
                            im_feat=feats[f'level_{im_feat_level}'],
                            gs_model=getattr(self, f'gs_{i}'),
                            level=i,
                            size=(H,W)
                            )
                    if i == cfg.mvs.cas_config.num-1:
                        rgb_vr, rgb_bs, rgb_obj, rgb_atten, depth_medium = output
                        for b_i in range(B):
                            if cfg.mvs.reweighting: 
                                render_novel_i = rgb_vr
                            else:
                                render_novel_i = rgb_vr
                            render_novel_i = render_novel_i[b_i].permute(1,2,0)
                            render_obj = rgb_obj[b_i].permute(1,2,0)
                            if cfg.mvs.eval_center:
                                H_crop, W_crop = int(H_img*0.1), int(W_img*0.1)
                                render_novel_i = render_novel_i[H_crop:-H_crop, W_crop:-W_crop,:]
                                render_obj = render_obj[H_crop:-H_crop, W_crop:-W_crop,:]
                            if v_i == 0:
                                pred_rgb_nb_list.append([(render_novel_i.data.cpu().numpy()*255).astype(np.uint8)])
                                pred_obj_list.append([(render_obj.data.cpu().numpy()*255).astype(np.uint8)])
                            else:
                                pred_rgb_nb_list[b_i].append((render_novel_i.data.cpu().numpy()*255).astype(np.uint8))
                                pred_obj_list[b_i].append((render_obj.data.cpu().numpy()*255).astype(np.uint8))
                            img_dir = os.path.join(cfg.result_dir, '{}_{}'.format(batch['meta']['scene'][b_i], batch['meta']['tar_view'][b_i].item()), 'render')
                            obj_dir = os.path.join(cfg.result_dir, '{}_{}'.format(batch['meta']['scene'][b_i], batch['meta']['tar_view'][b_i].item()), 'obj')
                            os.makedirs(img_dir,exist_ok=True)
                            os.makedirs(obj_dir,exist_ok=True)
                            save_path = os.path.join(img_dir,f'{len(pred_rgb_nb_list[b_i])}.png')
                            obj_path = os.path.join(obj_dir,f'{len(pred_rgb_nb_list[b_i])}.png')
                            PIL.Image.fromarray((render_novel_i.data.cpu().numpy()*255).astype(np.uint8)).save(save_path)
                            PIL.Image.fromarray((render_obj.data.cpu().numpy()*255).astype(np.uint8)).save(obj_path)
            for b_i in range(B):
                video_path = os.path.join(cfg.result_dir, '{}_{}_{}.mp4'.format(batch['meta']['scene'][b_i], batch['meta']['tar_view'][b_i].item(), batch['meta']['frame_id'][b_i].item()))
                objv_path = os.path.join(cfg.result_dir, '{}_{}_{}_obj.mp4'.format(batch['meta']['scene'][b_i], batch['meta']['tar_view'][b_i].item(), batch['meta']['frame_id'][b_i].item()))
                imageio.mimwrite(video_path, np.stack(pred_rgb_nb_list[b_i]), fps=10, quality=10)
                imageio.mimwrite(objv_path, np.stack(pred_obj_list[b_i]), fps=10, quality=10)