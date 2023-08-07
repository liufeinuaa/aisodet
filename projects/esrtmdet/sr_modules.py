import torch
import torch.nn as nn
import torch.nn.functional as F
from mmengine.model.weight_init import (constant_init, kaiming_init,
                                        normal_init, xavier_init)
import math
import pdb
from aisodet.registry import MODELS
from mmengine.model import BaseModule

"""
会使用到的超分辨率模型中的辅助模块

"""

def default_init_weights(module, scale=1):
    """Initialize network weights.

    Args:
        modules (nn.Module): Modules to be initialized.
        scale (float): Scale initialized weights, especially for residual
            blocks. Default: 1.
    """
    for m in module.modules():
        if isinstance(m, nn.Conv2d):
            kaiming_init(m, a=0, mode='fan_in', bias=0)
            m.weight.data *= scale
        elif isinstance(m, nn.Linear):
            kaiming_init(m, a=0, mode='fan_in', bias=0)
            m.weight.data *= scale


def make_layer(block, num_blocks, **kwarg):
    """Make layers by stacking the same blocks.

    Args:
        block (nn.module): nn.module class for basic block.
        num_blocks (int): number of blocks.

    Returns:
        nn.Sequential: Stacked blocks in nn.Sequential.
    """
    layers = []
    for _ in range(num_blocks):
        layers.append(block(**kwarg))
    return nn.Sequential(*layers)



"""
———————————————————— 一些插件模块 ——————————————————————————————
"""

class ResidualBlockNoBN(nn.Module):
    """Residual block without BN.
    主要用于EDSR中

    It has a style of:

    ::

        ---Conv-ReLU-Conv-+-
         |________________|

    Args:
        mid_channels (int): Channel number of intermediate features.
            Default: 64.
        res_scale (float): Used to scale the residual before addition.
            Default: 1.0. 但是在常用的edsr模型中这里的scale应该为0.1
    """
    def __init__(self, mid_channels=64, res_scale=1.0):
        super().__init__()
        self.res_scale = res_scale #  就是残差模块的输出比例
        self.conv1 = nn.Conv2d(mid_channels, mid_channels, 3, 1, 1, bias=True)
        self.conv2 = nn.Conv2d(mid_channels, mid_channels, 3, 1, 1, bias=True)

        self.relu = nn.ReLU(inplace=True)

        # if res_scale < 1.0, use the default initialization, as in EDSR.
        # if res_scale = 1.0, use scaled kaiming_init, as in MSRResNet.
        if res_scale == 1.0:
            self.init_weights()

    def init_weights(self):
        """Initialize weights for ResidualBlockNoBN.

        Initialization methods like `kaiming_init` are for VGG-style modules.
        For modules with residual paths, using smaller std is better for
        stability and performance. We empirically use 0.1. See more details in
        "ESRGAN: Enhanced Super-Resolution Generative Adversarial Networks"
        """

        for m in [self.conv1, self.conv2]:
            default_init_weights(m, 0.1)
    
    def forward(self, x):
        """Forward function.

        Args:
            x (Tensor): Input tensor with shape (n, c, h, w).

        Returns:
            Tensor: Forward results.
        """
        identity = x
        out = self.conv2(self.relu(self.conv1(x)))
        return identity + out * self.res_scale



class decode_ResBlock(nn.Module):
    """
    不知道用在那边的了。。。(应该是增加上sr分支后的解码部分)
    """
    def __init__(self, in_chn, out_chn, \
            dropout=0, use_conv_in_skip=False) -> None:
        super().__init__()

        self.in_conv = nn.Sequential(
            nn.SiLU(),
            nn.Conv2d(in_chn, out_chn, kernel_size=3, padding=1)
        )

        self.out_conv = nn.Sequential(
            nn.SiLU(),
            nn.Dropout(p=dropout),
            nn.Conv2d(out_chn, out_chn, kernel_size=3, padding=1)
        )

        if in_chn != out_chn and not use_conv_in_skip:
            self.skip_conn = nn.Conv2d(in_chn, out_chn, kernel_size=1)
        elif use_conv_in_skip:
            self.skip_conn = nn.Conv2d(in_chn, out_chn, kernel_size=3, padding=1)
        else:
            self.skip_conn = nn.Identity()
    
    def forward(self, x):
        h = self.in_conv(x)
        h = self.out_conv(h)

        return h + self.skip_conn(x)

"""
使用unet12调试出来的结构，方便并入sr7的结构中
"""
class unetResBlock(nn.Module):
    def __init__(self, 
                in_chn, out_chn, 
                dropout=0, 
                use_conv_in_skip=False,
                use_norm=False):
        super().__init__()

        self.in_conv = nn.Sequential(nn.Conv2d(in_chn, out_chn, 3, padding=1),
                                    nn.ReLU())
        self.out_conv = nn.Sequential(nn.Dropout(p=dropout),
                                    nn.Conv2d(out_chn, out_chn, 3, padding=1),
                                    nn.ReLU())

        if in_chn != out_chn and not use_conv_in_skip:
            self.skip_conn = nn.Conv2d(in_chn, out_chn, kernel_size=1)
        elif use_conv_in_skip:
            self.skip_conn = nn.Conv2d(in_chn, out_chn, 3)
        else:
            self.skip_conn = nn.Identity()

    def forward(self, x):
        h = self.in_conv(x)
        h = self.out_conv(h)
        return h + self.skip_conn(x)


"""
—————————————————————————————— end ———————————————————————————————————
"""








"""
—————————————————————————————— 自编的上采样方法模块 ——————————————————————————
"""
class PixelUpSample(nn.Sequential):
    """Upsample module used in EDSR.
    原名UpSampleModul, 重命名为更准确意思的 PixelUpSample

    采用亚像素上采样方法（PixelShuffle）来实现图像尺寸的放大， 在执行上采样前接一个3x3conv2d来放大通道维度
    主要用于EDSR中
    
    Args:
        scale (int): Scale factor. Supported scales: 2^n and 3.
        mid_channels (int): Channel number of intermediate features.
    """
    def __init__(self, scale, mid_channels):
        modules = []
        if (scale & (scale - 1)) == 0: # scale = 2^n ???
            for _ in range(int(math.log(scale, 2))):
                # 每次执行一次上采样率为2的 的 亚像素重组上采样方法
                modules.append(
                    nn.Conv2d( 
                        mid_channels, 
                        mid_channels * 4, # 2*2
                        3,
                        padding=1))
                modules.append(
                    nn.PixelShuffle(2)
                )
        elif scale == 3:
            modules.append(nn.Conv2d(
                mid_channels,
                mid_channels * 9, # 3*3
                3,
                padding=1
            ))
            modules.append(nn.PixelShuffle(3))
        else:
            raise ValueError(f'scale {scale} is not supported. '
                             'Supported scales: 2^n and 3.')

        super().__init__(*modules)

        # 对模块进行指定的初始化
        # pdb.set_trace()
        for i in range(len(modules)): # 这块代码有必要嘛？？？感觉可以取消掉啊
            default_init_weights(modules[i], 1)


class DeconvUpSample(nn.Sequential):
    """
    新增的使用反卷积来实现上采样的方法
    """
    def __init__(self, 
                scale, chn, use_act=False, use_BN=False, **kwagrs):
        modules = []
        if (scale & (scale - 1)) == 0: # scale = 2^n ???
            for _ in range(int(math.log(scale, 2))):
                # 每次执行一次上采样率为2的deconv
                modules.append(
                    nn.ConvTranspose2d(chn, chn, 3, stride=2, padding=1, output_padding=1)
                )
                if use_BN:
                    modules.append(
                        nn.BatchNorm2d(chn)
                    )
                if use_act:
                    modules.append(
                        nn.PReLU() # 这边具体使用的激活函数方法可以修改
                    )
        else:
            raise ValueError(f'scale {scale} is not supported. '
                             'Supported scales: 2^n.')

        super().__init__(*modules)


class ConvUpSample(nn.Module):
    """
    这里就是原始unet网络中的上采样方法（直接2倍插值+conv）   其实也就是SRCNN方法的思路先插值放大再接conv 
    原本名称:unetUpsample , 这里重命名为: ConvUpSample

    An upsampling layer with an optional convolution.

    :param channels: channels in the inputs and outputs.
    :param use_conv: a bool determining if a convolution is applied.
    :param dims: determines if the signal is 1D, 2D, or 3D. If 3D, then upsampling occurs in the inner-two dimensions.
    
    interpolate_mode:  
            mode (str): algorithm used for upsampling:
            ``'nearest'`` | ``'linear'`` | ``'bilinear'`` | ``'bicubic'`` |
            ``'trilinear'`` | ``'area'`` | ``'nearest-exact'``. Default: ``'nearest'``
    
    """
    def __init__(self, channels, up_factor, use_conv, 
                 interpolate_mode='nearest', align_corners=False,
                 **kwargs) -> None:
        super().__init__()
        self.channels = channels
        self.use_conv = use_conv
        self.up_factor = up_factor
        self.interpolate_mode = interpolate_mode
        self.align_corners = align_corners

        # if self.interpolate_mode != 'nearest':
        #     self.align_corners = True
        # else:
        #     self.align_corners = False

        # pdb.set_trace()
        if self.use_conv:
            self.conv = nn.Conv2d(channels, channels, kernel_size=3, padding=1) # 保持featmap不变
    
    def forward(self, x):
        assert x.shape[1] == self.channels
        # 真实上采样的代码
        # x = F.interpolate(x, scale_factor=self.up_factor, mode='nearest')

        # pdb.set_trace()
        
        x = F.interpolate(x, scale_factor=self.up_factor, mode=self.interpolate_mode, align_corners=self.align_corners)

        # if self.interpolate_mode == 'bicubic': # x 做了归一化的，不在0-255之间了
        #     x = torch.clamp(min=0, max=255)

        # pdb.set_trace()
        if self.use_conv:
            x = self.conv(x) # 用卷积处理下上采样后的feature
        return x

"""
—————————————————————————————— end ———————————————————————————————————
"""







"""
——————————————自编的用于sr分支中的feat 编码模块，通常采用edsr作为decoder（解码）模块 ————————————
"""

@MODELS.register_module()
# class preEncoder(nn.Module):
class preEncoder(BaseModule):
    """
    就是superyolo中输入的两个特征（backbone 的低级特征和对应的高级特征）做预处理，使得他们的通道维度和特征尺寸一致，再输入edsr超分辨网络中
    """
    def __init__(self, 
                h_chn, # high-level featmap 对应的通道
                l_chn, # low-level featmap 对应的通道
                up_factor=1, # featmap 放大的倍数, encode上，上采样放大的倍数（默认不对l feat不放大，将h feat 放大到l feat 的大小上）
                ) -> None:
        super().__init__()
        self.up_factor = up_factor
        self.conv_h = nn.Conv2d(h_chn, h_chn//2, 1, bias=False) # 1x1 conv 并且输入通道减半
        self.conv_l = nn.Conv2d(l_chn, l_chn//2, 1, bias=False)
        self.relu = nn.ReLU()

        self.last_conv = nn.Sequential(
            nn.Conv2d((h_chn + l_chn) // 2, 256, \
                3, padding=1, bias=False),
            nn.ReLU(),
            nn.Conv2d(256, 128, 3, padding=1, bias=False),
            nn.ReLU(),
            nn.Conv2d(128, 64, 1), # 用1x1conv将通道维度降到edsr的输入通道维度
        )
    
    def forward(self, 
                h_feat, # high-level featmap 
                l_feat, # low-level featmap 
                ):
        l_feat = self.conv_l(l_feat)
        l_feat = self.relu(l_feat)

        h_feat = self.conv_h(h_feat)
        h_feat = self.relu(h_feat)
        
        # 这里采用双线性插值来直接放大h_feat特征图的尺寸....这个方法应该能修改
        # pdb.set_trace()
        h_feat = F.interpolate(h_feat, size=[s*self.up_factor for s in l_feat.shape[2:]], mode='bilinear', align_corners=True)
        
        if self.up_factor > 1:
            l_feat = F.interpolate(l_feat, size=[s*self.up_factor for s in l_feat.shape[2:]], mode='bilinear', align_corners=True)
        
        # pdb.set_trace()
        x = torch.cat([h_feat, l_feat], dim=1)
        x = self.last_conv(x)
        return x
    


@MODELS.register_module()
class unetEncoder(BaseModule):
    def __init__(self, 
                feat_chn=[96, 192, 384], 
                up_factor=2, # encoder 部分预先的放大倍数
                # use_shared_upblocks=False,
                conv_resample=True, # 就是是否使用卷积来实现上下采样
                init_cfg=None):
        super().__init__(init_cfg)

        self.lvl_resbls = nn.ModuleList()
        self.up_blocks = nn.ModuleList()
        self.out_res = unetResBlock(feat_chn[0]//2, 64)

        # 初始化unetEncoder模块
        for i, chn in list(enumerate(feat_chn))[::-1]:
            # print(i, chn)
            if i == 2:
                self.lvl_resbls.append(unetResBlock(chn, chn//2))
            else:
                self.lvl_resbls.append(unetResBlock(chn*2, chn//2))
            
            self.up_blocks.append(ConvUpSample(chn//2, up_factor, use_conv=conv_resample))

        # pdb.set_trace()

    
    def forward(self, x):
        feats = [x[i] for i in range(len(x))[::-1]] # 对原本输入顺序为lvl2-0的顺序到了一下
        lvl = range(len(x))
        for i, res_bls, up_bls, feat in zip(lvl, self.lvl_resbls, self.up_blocks, feats):
            # pdb.set_trace()
            if i==0:
                cat_feat = feat
            else:
                cat_feat = torch.cat([cat_feat, feat], dim=1)
            # pdb.set_trace() 

            cat_feat = res_bls(cat_feat)
            cat_feat = up_bls(cat_feat)
        
        # pdb.set_trace()
        return self.out_res(cat_feat)



@MODELS.register_module()
class unetEncoder2(BaseModule):
    """
    使用更大的feat size 输入（即更底层的lvl feat），
    """
    def __init__(self, 
                feat_chn=[48, 96, 192, 384], 
                up_factor=2, # encoder 部分预先的放大倍数
                # use_shared_upblocks=False,
                conv_resample=True, # 就是是否使用卷积来实现上下采样
                init_cfg=None):
        super().__init__(init_cfg)

        self.lvl_resbls = nn.ModuleList()
        self.up_blocks = nn.ModuleList()
        self.out_res = unetResBlock(feat_chn[0]//2, 64)
        
        # pdb.set_trace()
        # 初始化unetEncoder模块
        for i, chn in list(enumerate(feat_chn))[::-1]:
            # print(i, chn)
            if i == (len(feat_chn) - 1): # 对最上层的feat（chn最大的feat）做不同的处理
                self.lvl_resbls.append(unetResBlock(chn, chn//2))
            else:
                self.lvl_resbls.append(unetResBlock(chn*2, chn//2))
            
            self.up_blocks.append(unetUpsample(chn//2, up_factor, use_conv=conv_resample))

        # pdb.set_trace()

    
    def forward(self, x):
        feats = [x[i] for i in range(len(x))[::-1]] # 对原本输入顺序为lvl2-0的顺序到了一下
        lvl = range(len(x))
        # pdb.set_trace()
        for i, res_bls, up_bls, feat in zip(lvl, self.lvl_resbls, self.up_blocks, feats):
            # pdb.set_trace()
            if i==0:
                cat_feat = feat
            else:
                cat_feat = torch.cat([cat_feat, feat], dim=1)
            # pdb.set_trace() 

            cat_feat = res_bls(cat_feat)
            cat_feat = up_bls(cat_feat)
        
        # pdb.set_trace()
        return self.out_res(cat_feat)

"""
—————————————————————————————— end ———————————————————————————————————
"""



@MODELS.register_module()
class EDSRnet(BaseModule):
    """
    esrtmdet 中使用到的 edsr 模型

    EDSR network structure.

    Paper: Enhanced Deep Residual Networks for Single Image Super-Resolution.
    Ref repo: https://github.com/thstkdgus35/EDSR-PyTorch

    Args:
        in_channels (int): Channel number of inputs.
        out_channels (int): Channel number of outputs.
        mid_channels (int): Channel number of intermediate features.
            Default: 64.
        num_blocks (int): Block number in the trunk network. Default: 16.
        upscale_factor (int): Upsampling factor. Support 2^n and 3.
            Default: 4.
        res_scale (float): Used to scale the residual in residual block.
            Default: 1.
        rgb_mean (list[float]): Image mean in RGB orders.
            Default: [0.4488, 0.4371, 0.4040], calculated from DIV2K dataset.
        rgb_std (list[float]): Image std in RGB orders. In EDSR, it uses
            [1.0, 1.0, 1.0]. Default: [1.0, 1.0, 1.0].
    """
    def __init__(self, 
                in_channels,
                out_channels=3, # 默认为3
                mid_channels=64,
                num_blocks=16,
                upscale_factor=2,
                res_scale=1,
                init_cfg=None):
        super().__init__(init_cfg)
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.mid_channels = mid_channels
        self.num_blocks = num_blocks
        self.upscale_factor = upscale_factor

        # 第一个conv用于预处理featmap
        self.conv_first = nn.Conv2d(in_channels, mid_channels, 3, padding=1)
        # 模型的主干部分：多层堆叠的残差模块
        self.body = make_layer(
            ResidualBlockNoBN,
            num_blocks,
            # 传入ResidualBlockNoBN中的参数(必须得使用=号)
            mid_channels=mid_channels,
            res_scale=res_scale
        )
        # 多层堆叠的残差模块后的conv
        self.conv_after_body = nn.Conv2d(mid_channels, mid_channels, 3, padding=1)
        # 模型上采样模块，扩大featmap size
        self.upsample = PixelUpSample(upscale_factor, mid_channels)
        # 最后一个conv
        self.conv_last = nn.Conv2d(mid_channels, out_channels, 3, padding=1)

    def forward(self, x):
        """Forward function.

        Args:
            x (Tensor): Input tensor with shape (n, c, h, w).

        Returns:
            Tensor: Forward results.
        """
        # pdb.set_trace()
        x = self.conv_first(x)
        res_feat = self.conv_after_body(self.body(x))
        res_feat += x # 整体模型上的残差连接

        # 模型输出的结果
        x = self.conv_last(self.upsample(res_feat))

        return x, res_feat



