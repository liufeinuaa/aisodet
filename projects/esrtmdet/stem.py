import torch
import torch.nn as nn
import torch.nn.functional as F
from mmengine.model import BaseModule
from typing import Iterable, List, Optional, Union
from mmdet.utils import ConfigType, OptConfigType, OptMultiConfig
from mmcv.cnn import ConvModule

from aisodet.registry import MODELS
import pdb


@MODELS.register_module()
class CSPNetXt_stem(BaseModule):
    """
    将 rtmdet 中的 stem 作为 sr net 的 encoder

    """
    def __init__(self, 
                 begin_chn,
                 widen_factor,
                 norm_cfg: ConfigType = dict(type='SyncBN', momentum=0.03, eps=0.001),
                 act_cfg: ConfigType = dict(type='SiLU'),
                 init_cfg: Union[dict, List[dict], None] = None):
        super().__init__(init_cfg)
        
        self.stem = nn.Sequential(
            ConvModule(
                3,
                int(begin_chn * widen_factor // 2),
                3,
                padding=1,
                stride=2, # 这边就实现了2倍下采样
                norm_cfg=norm_cfg,
                act_cfg=act_cfg 
            ),
            ConvModule(
                int(begin_chn * widen_factor // 2),
                int(begin_chn * widen_factor // 2),
                3,
                padding=1,
                stride=1,
                norm_cfg=norm_cfg,
                act_cfg=act_cfg
            ),
            ConvModule(
                int(begin_chn * widen_factor // 2),
                int(begin_chn * widen_factor),
                3,
                padding=1,
                stride=1,
                norm_cfg=norm_cfg,
                act_cfg=act_cfg
            )
        )
    
    def forward(self, x):
        return self.stem(x)