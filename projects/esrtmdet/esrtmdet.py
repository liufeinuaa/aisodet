import torch
import math
import numpy as np
import cv2
import mmcv
from mmengine.dist import get_world_size
from mmengine.logging import print_log
from torch import Tensor
import torch.nn.functional as F
from mmdet.structures import OptSampleList, SampleList
from mmdet.models.detectors.base import BaseDetector
from mmdet.structures.bbox import BaseBoxes
from mmdet.models.data_preprocessors import DetDataPreprocessor
from mmdet.utils import ConfigType, OptConfigType, OptMultiConfig
from typing import List, Tuple, Union
import torch.nn as nn

from aisodet.registry import MODELS
from .sr_modules import make_layer, PixelUpSample, DeconvUpSample, ConvUpSample, ResidualBlockNoBN
from .fa_loss import calcu_pixels_similarity, fa_loss
import pdb



@MODELS.register_module()
class ESRTMDet(BaseDetector):
    """
    esrtmdet 检测器的具体实现

    """

    def __init__(self,
                 backbone: ConfigType,
                 neck: ConfigType,
                 bbox_head: ConfigType,
                 
                 det_up_sample_blocks: ConfigType, # 共享上采样超分模块的cfg
                 sr_branch=None,

                 train_cfg: OptConfigType = None,
                 test_cfg: OptConfigType = None,
                 data_preprocessor: OptConfigType = None,
                 init_cfg: OptMultiConfig = None,
                 
                 use_syncbn: bool = True) -> None:
        super().__init__(
            data_preprocessor=data_preprocessor,
            init_cfg=init_cfg)

        self.data_norm_cfg = data_preprocessor

        self.backbone = MODELS.build(backbone)
        
        self.neck_lvlchns = neck.in_channels
        if neck is not None:
            self.neck = MODELS.build(neck)
        
        if train_cfg is not None:
            bbox_head.update(train_cfg=train_cfg['det']) # 将det相关的训练cfg 放在单独的key中
        bbox_head.update(test_cfg=test_cfg)
        self.bbox_head = MODELS.build(bbox_head)

        self.train_cfg = train_cfg
        self.test_cfg = test_cfg

        # TODO： Waiting for mmengine support, 默认都是启用的
        if use_syncbn and get_world_size() > 1:
            torch.nn.SyncBatchNorm.convert_sync_batchnorm(self)
            print_log('Using SyncBatchNorm()', 'current')


        # ——————————平行于det网络的 sr分支网络参数及初始化————————————————
        if sr_branch is not None:
            self.sr_branch_cfg = sr_branch
            if self.sr_branch_cfg.use_encoder:
                self.pre_encoder = MODELS.build(self.sr_branch_cfg.pre_encoder)

            self.sr_net = MODELS.build(self.sr_branch_cfg.sr_arch)
            self.sr_loss = MODELS.build(self.sr_branch_cfg.sr_loss)

            if self.sr_branch_cfg.align_feat:
                self.align_loss = MODELS.build(self.sr_branch_cfg.align_loss)
            
            # gt hr img 需要的正则化参数
            self.mean = torch.tensor(self.data_norm_cfg.mean).view(-1, 1, 1)
            self.std = torch.tensor(self.data_norm_cfg.std).view(-1, 1, 1)

            if self.sr_branch_cfg.use_affinity: # 在 det 和 sr 任务间的 feat 上增加一个 小的用于调整特征的仿射conv
                if self.sr_branch_cfg.align_one_feat:
                    self.affin_conv = nn.Conv2d(64, 64, 1)
                else:
                    self.affin_conv = nn.ModuleList()
                    for i in range(3):
                        self.affin_conv.append(nn.Conv2d(64, 64, 1))
        else:
            self.sr_branch_cfg = None

        # ———————— 插入det网络中的 上采样模块的参数与初始化的实现——————————————
        if det_up_sample_blocks is not None:
            self.det_upSBs = MODELS.build(det_up_sample_blocks)
            self.det_up_sample_cfg = det_up_sample_blocks


    def loss(self, 
            batch_inputs: Tensor, # 这里所谓的inputs就是指的图像数据
            batch_data_samples: SampleList #这里主要就是标签数据
            ) -> Union[dict, tuple]:
        """
        用于train中的方法
        计算一个batch输入的loss
        """
        x = self.extract_feat(batch_inputs)
        """
        x[0].shape
        torch.Size([8, 24, 256, 256])
        (Pdb) x[1].shape
        torch.Size([8, 48, 128, 128])
        (Pdb) x[2].shape
        torch.Size([8, 96, 64, 64])
        (Pdb) x[3].shape
        torch.Size([8, 192, 32, 32])
        (Pdb) x[4].shape
        torch.Size([8, 384, 16, 16])
        """

        losses = dict()


        if self.det_up_sample_cfg.insert_position == 'after_backbone':
            (up_x, resrep_x) = self.det_upSBs(x[-3:])
            neck_x = self.neck(up_x)
            det_losses = self.bbox_head.loss(neck_x, batch_data_samples)
            cls_scores = det_losses.pop('cls_scores') # 将多的cls_scores 推出来
            cls_mask = self.get_cls_mask(cls_scores)

        elif self.det_up_sample_cfg.insert_position == 'after_neck':
            neck_x = self.neck(x[-3:])
            (up_x, resrep_x) = self.det_upSBs(neck_x)
            det_losses = self.bbox_head.loss(up_x, batch_data_samples)
            cls_scores = det_losses.pop('cls_scores') # 将多的cls_scores 推出来
            cls_mask = self.get_cls_mask(cls_scores)

        losses.update(det_losses)
        
        if self.sr_branch_cfg is not None:
            gt_hr_shape = [shape * 2 for shape in batch_inputs.shape[-2:]]

            try:
                gt_hr_imgs = torch.stack([gt.gt_hr_img.hr_img for gt in batch_data_samples]) 
                # 之前忘了增加这个操作了。。。导致sr loss 巨大，。。。tmd
                gt_hr_imgs = self._norm_gt(gt_hr_imgs)
            except:
                pdb.set_trace()
                gt_hr_imgs = [self._norm_gt(gt.gt_hr_img.hr_img) for gt in batch_data_samples]
                gt_hr_imgs = self.pad_hr_img(gt_hr_imgs, gt_hr_shape)
            # pdb.set_trace()
            gt_hr_imgs = gt_hr_imgs.float()
            
            if self.sr_branch_cfg.use_encoder:
                # 新增加的 sr net 中的 encoder部分
                if self.sr_branch_cfg.use_fg_mask: 
# 在这里引入 gt rbbox 生成的 mask，遮蔽背景部分的信息，让背景不参与实际的超分优化
                    fg_mask, hr_fg_mask = self.get_foreground_mask(batch_data_samples)
                    fg_mask = torch.stack(fg_mask).float().to(batch_inputs.device)
                    hr_fg_mask = torch.stack(hr_fg_mask).float().to(batch_inputs.device)
                    # pdb.set_trace()
                    fg_inputs = fg_mask * batch_inputs
                    gt_hr_imgs = gt_hr_imgs * hr_fg_mask 
                    # pdb.set_trace()
                    sr_encoder_x = self.pre_encoder(fg_inputs)

                elif self.sr_branch_cfg.use_cls_mask:
# 因为都是在cuda上进行计算的，不会带来太高计算时间的开销
                    sr_encoder_x = self.pre_encoder(batch_inputs)
                    # pdb.set_trace()
                    sr_encoder_x = sr_encoder_x * cls_mask
                    gt_hr_imgs = gt_hr_imgs * F.interpolate(cls_mask, scale_factor=4)

                else:
                    sr_encoder_x = self.pre_encoder(batch_inputs)
            else:
                sr_encoder_x= x[0]

            # sr 分支的训练部分
            # pdb.set_trace()
            (sr_imgs, sr_res_feat) = self.sr_net(sr_encoder_x)
            
            # pdb.set_trace()
            if self.sr_branch_cfg.mask_as_weight:
                cls_mask_weight = F.interpolate(cls_mask, scale_factor=4)
                cls_mask_weight = cls_mask_weight*9 + 1
                cls_mask_weight = cls_mask_weight.repeat_interleave(3, dim=1)
                sr_losses =  self.sr_loss(sr_imgs, gt_hr_imgs, cls_mask_weight) 
            else:
                sr_losses =  self.sr_loss(sr_imgs, gt_hr_imgs) 
            
            losses.update(sr_losses=sr_losses)

            if self.sr_branch_cfg.align_feat: # 对齐 det net 中的上采样 feat 与 sr res feat
                # 手动关闭amp，开能避免自编loss的nan问题不啊
                with torch.cuda.amp.autocast(enabled=False):
                    
                    if self.sr_branch_cfg.align_encoder_feat and self.sr_branch_cfg.use_encoder:
                        stem_feat = x[0].float()
                        sr_encoder_x = sr_encoder_x.float()
                        stem_loss = self.align_loss(stem_feat, sr_encoder_x)
                        losses.update(stem_align_loss=stem_loss)

                    align_losses = []
                    sr_feat = []
                    if self.sr_branch_cfg.align_with_resrepfeat:
                        # 具体对齐 det 上采样模块中的 res 特征
                        sr_res_feat = sr_res_feat.float()

                        if self.sr_branch_cfg.align_one_feat:
                            # 尝试仅对齐 最底下的一层 feat
                            feat = resrep_x[0].float()
                            sr_feat = F.interpolate(sr_res_feat, scale_factor=1/4)
                            
                            if self.sr_branch_cfg.use_affinity:
                                sr_feat = self.affin_conv(sr_feat)
                            
                            if self.sr_branch_cfg.mask_as_weight:
                                cls_mask_weight = F.interpolate(cls_mask, scale_factor=1/4)
                                cls_mask_weight = cls_mask_weight*9 + 1
                                cls_mask_weight = cls_mask_weight.repeat_interleave(64, dim=1)
                                align_losses = self.align_loss(sr_feat, feat, weight=cls_mask_weight)
                            else:
                                align_losses = self.align_loss(sr_feat, feat)

                        else: # 默认对齐 3层 feat
                            for i, feat in enumerate(resrep_x):
                                feat = feat.float()
                                sr_feat.append(F.interpolate(sr_res_feat, scale_factor=1/2**(i+2))) # 针对不同的resrep feat 使用不同的下采样倍数，使得 feat size 相一致

                                if self.sr_branch_cfg.use_affinity:
                                    sr_feat[i] = self.affin_conv[i](sr_feat[i])
                                
                                # pdb.set_trace()
                                # 计算各个feat上与sr的 align loss
                                if self.sr_branch_cfg.mask_as_weight:
                                    cls_mask_weight = F.interpolate(cls_mask, scale_factor=1/2**(i+2))
                                    cls_mask_weight = cls_mask_weight*9 + 1
                                    cls_mask_weight = cls_mask_weight.repeat_interleave(64, dim=1)
                                    align_losses.append(self.align_loss(sr_feat[i], feat, weight=cls_mask_weight))
                                else:
                                    align_losses.append(self.align_loss(sr_feat[i], feat))
                        
                            align_losses = torch.sum(torch.stack(align_losses))

                        if torch.isnan(align_losses): #增加 loss nan 检测
                            pdb.set_trace()
                    else:
                        pass
                    
                    losses.update(align_losses=align_losses)
            else:
                pass 

        return losses



    def predict(self, batch_inputs: Tensor, 
                batch_data_samples: SampleList,
                rescale: bool = True) -> SampleList:
        """
        用于测试的方法
        主要用于结合模型的前向推理与后处理方法
        """
        x = self.extract_feat(batch_inputs)
        
        if self.det_up_sample_cfg.insert_position == 'after_backbone':
            up_x, _, = self.det_upSBs(x[-3:])
            neck_x = self.neck(up_x)
            results_list = self.bbox_head.predict(
                neck_x, batch_data_samples, rescale=rescale)
            
        elif self.det_up_sample_cfg.insert_position == 'after_neck':
            neck_x = self.neck(x[-3:])
            up_x, _, = self.det_upSBs(neck_x)
            results_list = self.bbox_head.predict(
                up_x, batch_data_samples, rescale=rescale) 
        
        batch_data_samples = self.add_pred_to_datasample(
            batch_data_samples, results_list
        )
        return batch_data_samples



    def _forward(self, batch_inputs: Tensor, 
                batch_data_samples: OptSampleList = None):
        """Network forward process. Usually includes backbone, neck and head
        forward without any post-processing.
        """
        x = self.extract_feat(batch_inputs) 

        if self.det_up_sample_cfg.insert_position == 'after_backbone':
            up_x, _, = self.det_upSBs(x[-3:])
            neck_x = self.neck(up_x)
            det_results = self.bbox_head.forward(neck_x)

        elif self.det_up_sample_cfg.insert_position == 'after_neck':
            neck_x = self.neck(x[-3:])
            up_x, _, = self.det_upSBs(neck_x)
            det_results = self.bbox_head.forward(up_x)

        return det_results # 这边随便用个list封装起来

        
    def extract_feat(self, 
                    batch_inputs: Tensor # 就是输入的图像数据
                    ):
        """
        提取图像特征（常规的one-stage模型中这里就是指backbone+neck后出来的特征， 
            而我这里因为要引入sr网络，因此这边提取执行backbone后的特征图）

        """
        x = self.backbone(batch_inputs)
        # pdb.set_trace()
        return x

    def _norm_gt(self, gt_imgs):
        """
        增加对hr imgs 的规范化
        """
        if self.mean is not None and self.std is not None:

            """
            这里 gt_imgs.dim() == 4 or 3 都是可以的
            """
            mean = self.mean.to(gt_imgs.device)
            std = self.std.to(gt_imgs.device)

            batch_gt_imgs = gt_imgs.float()
            norm_gt_imgs = (batch_gt_imgs - mean) / std
            # pdb.set_trace()
            return norm_gt_imgs
        else:
            return gt_imgs
    

# 效果不行。。。 废弃的代码
    def get_foreground_mask(self, batch_data_samples):
        """
        这部分代码还是插入到 data pipeline 应该更好 （能加速训练）

        由 gt rbbox 的信息 生成 前景信息 mask，让 sr net 专注于 前景信息的超分，背景超分被抑制
        """
        # pdb.set_trace()
        fg_mask = []
        fg_mask_s2x = []
        for data_sample in batch_data_samples:

            gt_rbbox = data_sample.gt_instances.bboxes # 得到 RotatedBoxes 的实例
            # pdb.set_trace()
            gt_poly = gt_rbbox.rbox2corner(gt_rbbox.tensor) # 得到 poly 形式的 rbbox gt
            
            # pdb.set_trace()
            nums = gt_poly.size(0)
            gt_poly_np = gt_poly.cpu().numpy()
            gt_poly_cv = gt_poly_np.reshape(nums, 1, 4, 2).astype(np.int32)

            mask = np.full((512, 512, 3), 0, np.uint8)
            for poly_cv in gt_poly_cv:
                # pdb.set_trace()
                cv2.fillPoly(mask, poly_cv, (1, 1, 1))
                # cv2.fillPoly(mask, poly_cv, 255) # 用于可视化时候的参数
                # mmcv.imwrite(mask, './ts_sr20_gt_rbbox_mask.jpg')
            
            # pdb.set_trace()
            mask_s2x = mmcv.imrescale(mask, scale=2)
            fg_mask.append(torch.tensor(mask).permute(2, 0, 1))
            # pdb.set_trace()
            fg_mask_s2x.append(torch.tensor(mask_s2x).permute(2, 0, 1))

        return fg_mask, fg_mask_s2x

    def get_cls_mask(self, cls_scores):
        # 根据 cls scores 得到 能用于 sr net 中的 mask
        # 因为都是在cuda上进行计算的，不会带来太高计算时间的开销
        masks = []
        for i, cls_s in enumerate(cls_scores):
            max_cls, _ = cls_s.max(dim=1)
            mean = max_cls.mean()
            std = max_cls.std()
            mask = torch.unsqueeze(max_cls, dim=1)
            mask = (mask >= (mean + std)).float()
            # pdb.set_trace()
            masks.append(F.interpolate(mask, scale_factor=2**(i+1)))
            if i == 1:
                cls_mask = torch.logical_or(masks[0], masks[1])
            if i > 1:
                cls_mask = torch.logical_or(cls_mask, masks[i])
        
        del masks, mask
        cls_mask = cls_mask.float()

        return cls_mask 
    

    def pad_hr_img(self, hr_imgs, gt_hr_shape, pad_size_divisor=1):
        """
        将 hr img 填充到指定的尺寸，主要用于 hrsc 这类 输入 大小会变化的数据集中。。。因为在预处理过程中会把 batch inputs的size 都统一起来。。。 因此这边也得做这个填充操作。。。
        """
        self.pad_size_divisor = pad_size_divisor
        self.pad_value = 0
        
        pad_imgs = []
        for idx, img in enumerate(hr_imgs):
            # pdb.set_trace()
            h, w = img.shape[-2:]
            target_h = math.ceil(
                gt_hr_shape[0] / self.pad_size_divisor) * self.pad_size_divisor
            target_w = math.ceil(
                gt_hr_shape[1] / self.pad_size_divisor) * self.pad_size_divisor
            pad_h = target_h - h
            pad_w = target_w - w
            img = F.pad(img, (0, pad_w, 0, pad_h),
                                 'constant', self.pad_value)
            # img = img.float()
            pad_imgs.append(img)

        # pdb.set_trace()
        return torch.stack(pad_imgs)

    def show_lr_hr_imgs(self, batch_inputs, gt_hr_imgs):
        """
        检查做的处理的究竟对不对。。。
        """
        # pdb.set_trace()
        mean = self.mean.to(gt_hr_imgs.device)
        std = self.std.to(gt_hr_imgs.device)

        lr = batch_inputs[0]
        hr = gt_hr_imgs[0]

        lr = lr * std + mean
        hr = hr * std + mean

        lr = lr.permute(1, 2, 0)
        hr = hr.permute(1, 2, 0)

        hr_np = hr.cpu().numpy()
        lr_np = lr.cpu().numpy()

        lr_np = lr_np.astype(np.uint8)
        hr_np = hr_np.astype(np.uint8)

       
        # pdb.set_trace()
        cv2.imwrite('show_lr.jpg', lr_np)
        cv2.imwrite('show_hr.jpg', hr_np)

        # pass

