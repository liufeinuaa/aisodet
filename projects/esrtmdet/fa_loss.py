import torch
from torch import nn
import torch.nn.functional as F
from mmdet.models.losses.utils import weighted_loss
from mmdet.models.losses import L1Loss
from aisodet.registry import MODELS
import pdb


def calcu_pixels_similarity(feat, use_subsample=True):
    """
    计算输入特征内部像素间的相似性，得到feat的像素间相似性矩阵 （来自 DSRL 文章中）

    本质上就是 计算各个pixel（shape=1x1xc）之间的点积相似度
    """
    # pdb.set_trace()
     
    if use_subsample: # 控制是否做下采样
        # 先对feat做1/8降采样, 要不然后面计算不了就。。。
        feat = F.interpolate(feat, scale_factor=1/8) # 就使用默认的nearest 方法, 并且默认下采样8倍

    Fi = torch.reshape(feat, (feat.shape[0], feat.shape[1], -1))
    Fj = torch.clone(Fi).permute(0, 2, 1)

    SM = torch.bmm(Fj, Fi) # 需要的显存太大了。。。计算不出来？？？()
    
    Fi_p2 = Fi.norm(dim=1, p=2).unsqueeze(1) # 分别计算chn维度上的p2范数
    Fj_p2 = Fj.norm(dim=2, p=2).unsqueeze(-1) # 分别计算chn维度上的p2范数
    norm_M = torch.bmm(Fj_p2, Fi_p2) # 归一化因子矩阵

    # pdb.set_trace()
    if torch.isnan(norm_M).sum() > 0:
        pdb.set_trace

    if torch.isinf(norm_M).sum() > 0:
        pdb.set_trace()

    # pdb.set_trace()
    SM_norm = torch.mul(SM, 1/norm_M) # 得到最终的归一化后的相似度矩阵
    
    return SM_norm

@weighted_loss
def fa_loss(det, sr, use_subsample=True):
    """
    Feature Affinity loss 的实现
    """
    SM_norm_det = calcu_pixels_similarity(det, use_subsample)
    SM_norm_sr = calcu_pixels_similarity(sr, use_subsample)
    
    # pdb.set_trace()
    l1 = nn.L1Loss() # 默认的l1 loss 就是做了求平均操作的，因此所谓的fa loss 就是feat的 similarity matirx 间的求 l1 loss
    # l1 = L1Loss()  
    loss = l1(SM_norm_det, SM_norm_sr)
    
    return loss


@MODELS.register_module()
class FALoss(nn.Module):
    """
    Feature Affinity loss 的实现
    第一版 使用 DSRL 文章中的feat像素间相似性矩阵 
    """
    def __init__(self, 
                reduction='mean',
                loss_weight=1.0,
                **kwargs) -> None:
        super(FALoss, self).__init__()
        assert reduction in ['none', 'sum', 'mean']
        self.reduction = reduction
        self.loss_weight = loss_weight
        

    def forward(self, det, sr,
                use_subsample=True,
                weight=None,
                avg_factor=None,
                reduction_override=None,
                **kwargs):
        """Forward function.

        Args:
            det (torch.Tensor): det net Predicted convexes.
            sr (torch.Tensor): sr net Predicted convexes.
            weight (torch.Tensor, optional): The weight of loss for each
                prediction. Defaults to None.
            avg_factor (int, optional): Average factor that is used to average
                the loss. Defaults to None.
            pred_decode (torch.Tensor): Predicted decode bboxes.
            targets_decode (torch.Tensor): Corresponding gt decode bboxes.
            reduction_override (str, optional): The reduction method used to
               override the original reduction method of the loss.
               Defaults to None.

        Returns:
            loss (torch.Tensor)
        """
        assert reduction_override in (None, 'none', 'mean', 'sum')
        reduction = (
            reduction_override if reduction_override else self.reduction)
        if (weight is not None) and (not torch.any(weight > 0)) and (
                reduction != 'none'):
            return (det * weight).sum()
        if weight is not None and weight.dim() > 1:
            assert weight.shape == det.shape
            weight = weight.mean(-1)

        return fa_loss(
            det,
            sr,
            use_subsample=use_subsample, # 控制是否做下采样
            weight=weight,
            avg_factor=avg_factor,
            reduction=reduction,
            **kwargs) * self.loss_weight
    
#————————————————————————————————————————————————————————————————————————————

# gram 矩阵 广泛用于风格迁移中
def gram_matrix(y):
    """ Returns the gram matrix of y (used to compute style loss) """
    (b, c, h, w) = y.size()
    features = y.view(b, c, w * h)
    features_t = features.transpose(1, 2)   #C和w*h转置
    gram = features.bmm(features_t) / (c * h * w)   #bmm 将features与features_t相乘
    return gram


def norm_gramM(feat, use_subsample):
    """
    计算输入特征内部像素间的相似性，得到feat的像素间相似性矩阵 （来自 DSRL 文章中）

    本质上就是 计算各个pixel（shape=1x1xc）之间的点积相似度, 
    其和gram 矩阵 广泛用于风格迁移中，和这边的相似度矩阵的计算很像（本质上就是差不多的东西），
    但是 gram 矩阵出来的 size 是 chn x chn 而计算 DSRL中的相似性是 HW x HW （大得多）
    """
    if use_subsample: # 控制是否做下采样
        # 先对feat做1/8降采样, 要不然后面计算不了就。。。
        feat = F.interpolate(feat, scale_factor=1/8) # 就使用默认的nearest 方法, 并且默认下采样8倍

    (B, C, H, W) = feat.size()

    Feat = feat.view(B, C, -1)
    Feat_t = Feat.transpose(1, 2)
    
    gram = Feat.bmm(Feat_t) # gram 矩阵 得到的输入为 （B, C, C), 将HW维度上的信息合并到通道维度上了

    norm_F = Feat.norm(dim=2, p=2).unsqueeze(1) # 计算了HW 维度上的 2 范数

    norm_F_t = norm_F.transpose(1, 2)
    norm_M = norm_F_t.bmm(norm_F)


    if torch.isnan(norm_M).sum() > 0:
        pdb.set_trace

    if torch.isinf(norm_M).sum() > 0: # norm_M 中有 inf; gram 中也有 inf 导致 计算loss 会nan .... 好像是因为 feat 为 float16 引起的
        pdb.set_trace()

    # pdb.set_trace()
    norm_gramM = gram / norm_M
    
    # pdb.set_trace()
    return norm_gramM

    

@weighted_loss
def fa_loss_v2(det, sr, use_subsample, use_l2):
    """
    Feature Affinity loss 的实现
    """
    # pdb.set_trace()
    SM_norm_det = norm_gramM(det, use_subsample)
    SM_norm_sr = norm_gramM(sr, use_subsample)
    
    if use_l2:
        l2 = nn.MSELoss()
        loss = l2(SM_norm_det, SM_norm_sr)
    else:
        # pdb.set_trace()
        l1 = nn.L1Loss() # 默认的l1 loss 就是做了求平均操作的，因此所谓的fa loss 就是feat的 similarity matirx 间的求 l1 loss
        # l1 = L1Loss()  
        loss = l1(SM_norm_det, SM_norm_sr) # 因为做了归一化并且，l1 loss 又是求mean的，所以缩小了feat size 与 不缩小得到的结果大小是差不多的
    
    return loss


@MODELS.register_module()
class FALoss_v2(nn.Module):
    """
    Feature Affinity loss 的实现

    改进具体的实现方案, 将DSRL 中计算相似性矩阵的方法转换成 计算 gram 矩阵

    增加 使用 l2 loss 的切换

    """
    def __init__(self, 
                # fun='none',
                reduction='mean',
                loss_weight=1.0,
                use_subsample=True, # 默认为 True
                use_l2=False, # 默认为使用l1 loss 来学习对齐 两个输入feat间的 gram矩阵
                **kwargs) -> None:
        super(FALoss_v2, self).__init__()
        assert reduction in ['none', 'sum', 'mean']
        self.reduction = reduction
        self.loss_weight = loss_weight
        self.use_subsample = use_subsample
        self.use_l2 = use_l2
       

    def forward(self, det, sr,
                weight=None,
                avg_factor=None,
                reduction_override=None,
                **kwargs):
        """Forward function.

        Args:
            det (torch.Tensor): det net Predicted convexes.
            sr (torch.Tensor): sr net Predicted convexes.
            weight (torch.Tensor, optional): The weight of loss for each
                prediction. Defaults to None.
            avg_factor (int, optional): Average factor that is used to average
                the loss. Defaults to None.
            pred_decode (torch.Tensor): Predicted decode bboxes.
            targets_decode (torch.Tensor): Corresponding gt decode bboxes.
            reduction_override (str, optional): The reduction method used to
               override the original reduction method of the loss.
               Defaults to None.

        Returns:
            loss (torch.Tensor)
        """
        assert reduction_override in (None, 'none', 'mean', 'sum')
        reduction = (
            reduction_override if reduction_override else self.reduction)
        if (weight is not None) and (not torch.any(weight > 0)) and (
                reduction != 'none'):
            return (det * weight).sum()
        
        if weight is not None and weight.dim() > 1:
            # pdb.set_trace()
            assert weight.shape == det.shape
            weight = weight.mean(-1)

        return fa_loss_v2(
            det,
            sr,
            weight=weight,
            avg_factor=avg_factor,
            reduction=reduction,
            use_subsample=self.use_subsample, # 控制是否做下采样...不能这样操作？？？...成功了 把 **kwargs给注释掉就可以了
            use_l2=self.use_l2
            ) * self.loss_weight






