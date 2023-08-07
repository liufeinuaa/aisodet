import torch
import torch.nn as nn
import torch.nn.functional as F
from mmengine.model import BaseModule
from mmdet.utils import ConfigType, OptConfigType, OptMultiConfig
from mmcv.cnn import ConvModule, DepthwiseSeparableConvModule
from torch import Tensor

from aisodet.registry import MODELS
from .sr_modules import make_layer, PixelUpSample, ResidualBlockNoBN
import pdb

@MODELS.register_module()
class upSBs(BaseModule):
    """
    使用深度可分离卷积及大核卷积 (就是类似rtmdet中使用的卷积)
    来构建resblk

    """
    def __init__(self, 
                 insert_position,
                 feat_chn,
                 use_pixelshuffle=True,
                 shard_upSBs=False,
                 use_res_represent=False,
                 num_resrep_resblks=1,
                 resrep_chns=64,
                 use_rfa=False, 
                 use_depthwise_3xconv=False,
                 use_cspxt_blks=False,
                 init_cfg = None):
        super().__init__(init_cfg)

        self.insert_position = insert_position
        self.feat_chn = feat_chn
        self.use_pixelshuffle = use_pixelshuffle
        self.shard_upSBs = shard_upSBs
        self.use_res_represent = use_res_represent
        self.num_resrep_resblks = num_resrep_resblks
        self.resrep_chns = resrep_chns
        self.use_rfa=use_rfa # 是否使用 rfa （残差特征聚合方法）---- 实验结果表明这种复杂的密集连接带不来精度的提升。。。
        self.use_cspxt_blks=use_cspxt_blks # 是否 启用 cspxt blks 模块，就是使用大核深度可分离卷积作为主要resblk模块
        self.use_depthwise_3xconv=use_depthwise_3xconv # 是否使用 深度可分离卷积来替代原始的第一层的3x3卷积


        if self.use_pixelshuffle:
            if self.insert_position == 'after_neck':
                self.upsample = PixelUpSample(scale=2, mid_channels=self.feat_chn)
            
            elif self.insert_position == 'after_backbone':
                self.upsample_list = nn.ModuleList()
                for chn in self.feat_chn:
                    upsample = PixelUpSample(scale=2, mid_channels=chn)
                    self.upsample_list.append(upsample)
            else:
                pass
        
        self.upSBs = nn.ModuleList()
        
        if self.insert_position == 'after_neck':
            for _ in range(3):
                self.upSBs.append(self.upsample)

            """
            新的参数共享操作
            """
            if self.shard_upSBs: # 使用和 head 中实现的 shard 方法一样
                for i in range(len(self.upSBs)):
                    for j, layer in enumerate(self.upSBs[0]):
                        self.upSBs[i][j] = layer # 实现 亚像素上采样方法的共享

        elif self.insert_position == 'after_backbone':
            """
            这里可以参考unet做一些特征融合操作
            后面在添加
            """
            for chn, upsample in zip(self.feat_chn, self.upsample_list): # 这里 每次的 upsample 中的 chn 都不一样
                self.upSBs.append(upsample)

        # 残差表示模块
        if self.use_res_represent:
            # pdb.set_trace()
            if type(self.feat_chn) == int: # 就是在neck后插入时生效
                self.feat_chn = [self.feat_chn for _ in range(3)] # 在backbone 后插入 self.feat_chn就为list
            
            if self.use_rfa: # 启用 rfa 结构 ---- 没啥用就----。。。。
                self.down_chn_conv = nn.ModuleList()
                
                self.resrep_resblks = nn.ModuleList() # 全部输入 feats 上的
                self.after_resblks = nn.ModuleList()
                self.res_feat_aggr = nn.ModuleList()
                self.up_chn_conv = nn.ModuleList()

                for chn in self.feat_chn:
                    self.down_chn_conv.append(nn.Conv2d(chn, self.resrep_chns, 3, padding=1)) # 使用 3x3 conv

                    self.resblks_list = nn.ModuleList() # 单个 feat 上的 resblks
                    for i in range(num_resrep_resblks):
                        self.resblks_list.append(ResidualBlockNoBN(mid_channels=self.resrep_chns, res_scale=1))
                    self.resrep_resblks.append(self.resblks_list)

                    self.after_resblks.append(nn.Conv2d(self.resrep_chns, self.resrep_chns, 3, padding=1))
                    self.res_feat_aggr.append(nn.Conv2d(self.resrep_chns, self.resrep_chns, 1))
                    self.up_chn_conv.append(nn.Conv2d(self.resrep_chns, chn, 1))

                if self.shard_upSBs:
                    # 只能在 neck 后插入时启用
                    for lvl in range(len(self.down_chn_conv)):
                        self.down_chn_conv[lvl] = self.down_chn_conv[0]
                        self.after_resblks[lvl] = self.after_resblks[0]
                        self.res_feat_aggr[lvl] = self.res_feat_aggr[0]
                        self.up_chn_conv[lvl] = self.up_chn_conv[0]

                        for indx, layer in enumerate(self.resrep_resblks[0]):
                            # pdb.set_trace()
                            self.resrep_resblks[lvl][indx] = layer

            elif self.use_cspxt_blks:
                self.resrep_resblks = nn.ModuleList()
                self.resrep_upchn = nn.ModuleList()

                for chn in self.feat_chn:
                    conv_blks=CSPNeXtBlockNoBN(in_channels=chn, out_channels=self.resrep_chns, use_depthwise=self.use_depthwise_3xconv)
                    self.resrep_resblks.append(conv_blks)
                    self.resrep_upchn.append(nn.Conv2d(self.resrep_chns, chn, 1)) # 使用 1x1conv 在放大 chn 回到 backbone 对应的输出 chn上
                
                # pdb.set_trace()
                if self.shard_upSBs:
                    """
                    使用新的参数共享操作方法
                    """
                    for lvl in range(len(self.resrep_resblks)):
                        for indx, layer in enumerate(self.resrep_resblks[0]):
                            # pdb.set_trace()
                            self.resrep_resblks[lvl][indx] = layer
                    
                    for lvl in range(len(self.resrep_upchn)):
                        self.resrep_upchn[lvl] = self.resrep_upchn[0]
            else:
                # 原本的 方法
                self.resrep_resblks = nn.ModuleList()
                self.resrep_upchn = nn.ModuleList()
                for chn in self.feat_chn:
                    layers = []
                    layers.append(nn.Conv2d(chn, self.resrep_chns, 1)) # 先降低backbone出来的chn到 resblk需要的chn 使用 1x1conv
                    layers.append(make_layer(ResidualBlockNoBN, self.
                                                num_resrep_resblks, 
                                                mid_channels=self.resrep_chns,
                                                res_scale=1))
                    self.resrep_resblks.append(nn.Sequential(*layers))
                    self.resrep_upchn.append(nn.Conv2d(self.resrep_chns, chn, 1)) # 使用 1x1conv
                
                if self.shard_upSBs:
                    """
                    使用新的参数共享操作方法
                    """
                    for lvl in range(len(self.resrep_resblks)):
                        for indx, layer in enumerate(self.resrep_resblks[0]):
                            # pdb.set_trace()
                            self.resrep_resblks[lvl][indx] = layer

                    for lvl in range(len(self.resrep_upchn)):
                        self.resrep_upchn[lvl] = self.resrep_upchn[0]

    def forward(self, x):
        up_x= []
        resrep_x = []
        res_x = []
        if self.use_rfa:
            for i, feat in enumerate(x):
                # pdb.set_trace()
                res_feats = []
                resfeat = self.down_chn_conv[i](feat)
                res_feats.append(resfeat)
                for blk in self.resrep_resblks[i]:
                    # pdb.set_trace()
                    resfeat = blk(resfeat)
                    res_feats.append(resfeat)
                
                resfeat = self.after_resblks[i](resfeat)
                res_feats.append(resfeat)
                sum_res_feat = torch.sum(torch.stack(res_feats), dim=0)
                resrep_x.append(self.res_feat_aggr[i](sum_res_feat))
                # pdb.set_trace()
                res_x.append(self.up_chn_conv[i](resrep_x[i]) + feat)
                up_x.append(self.upSBs[i](res_x[i]))
        else:
            for i, feat in enumerate(x):
                resrep_x.append(self.resrep_resblks[i](feat))
                # pdb.set_trace()
                res_x.append(self.resrep_upchn[i](resrep_x[i]) + feat)
                up_x.append(self.upSBs[i](res_x[i])) 
        # pdb.set_trace()
        return up_x, resrep_x
    


class CSPNeXtBlockNoBN(BaseModule):
    """"
    在 CSPNetXt 上 去除 BN 与 identity 作为替换 原本插入 det net 中的 上采样模块中的基础res模块
    
    """
    def __init__(self, 
                in_channels: int,
                out_channels: int,
                expansion: float = 0.5,
                use_depthwise: bool = False,
                kernel_size: int = 5,
                conv_cfg: OptConfigType = None,
                norm_cfg = None, # 不启用bn
                act_cfg: ConfigType = dict(type='SiLU'),
                init_cfg = None):
        super().__init__(init_cfg=init_cfg)
        hidden_channels = int(out_channels * expansion)
        conv = DepthwiseSeparableConvModule if use_depthwise else ConvModule
        self.conv1 = conv(
            in_channels,
            hidden_channels,
            3,
            stride=1,
            padding=1,
            norm_cfg=norm_cfg,
            act_cfg=act_cfg)
        self.conv2 = DepthwiseSeparableConvModule(
            hidden_channels,
            out_channels,
            kernel_size,
            stride=1,
            padding=kernel_size // 2,
            conv_cfg=conv_cfg,
            norm_cfg=norm_cfg,
            act_cfg=act_cfg)

    def forward(self, x: Tensor) -> Tensor:
        """Forward function."""
        identity = x
        out = self.conv1(x)
        out = self.conv2(out)

        return out
    