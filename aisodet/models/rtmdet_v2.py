# Copyright (c) OpenMMLab. All rights reserved.
import torch
from mmengine.dist import get_world_size
from mmengine.logging import print_log

# from mmdet.registry import MODELS
from aisodet.registry import MODELS
from mmdet.utils import ConfigType, OptConfigType, OptMultiConfig
from mmdet.structures import OptSampleList, SampleList
from mmdet.models.detectors.single_stage import SingleStageDetector

from mmengine import MessageHub
from mmengine.visualization import Visualizer
import pdb
from torch import Tensor
import numpy as np

@MODELS.register_module()
class RTMDetv2(SingleStageDetector):
    """
    用于可视化输出结果
    
    
    Implementation of RTMDet.

    Args:
        backbone (:obj:`ConfigDict` or dict): The backbone module.
        neck (:obj:`ConfigDict` or dict): The neck module.
        bbox_head (:obj:`ConfigDict` or dict): The bbox head module.
        train_cfg (:obj:`ConfigDict` or dict, optional): The training config
            of ATSS. Defaults to None.
        test_cfg (:obj:`ConfigDict` or dict, optional): The testing config
            of ATSS. Defaults to None.
        data_preprocessor (:obj:`ConfigDict` or dict, optional): Config of
            :class:`DetDataPreprocessor` to process the input data.
            Defaults to None.
        init_cfg (:obj:`ConfigDict` or dict, optional): the config to control
            the initialization. Defaults to None.
        use_syncbn (bool): Whether to use SyncBatchNorm. Defaults to True.
    """

    def __init__(self,
                 backbone: ConfigType,
                 neck: ConfigType,
                 bbox_head: ConfigType,
                 train_cfg: OptConfigType = None,
                 test_cfg: OptConfigType = None,
                 data_preprocessor: OptConfigType = None,
                 init_cfg: OptMultiConfig = None,
                 use_syncbn: bool = True) -> None:
        super().__init__(
            backbone=backbone,
            neck=neck,
            bbox_head=bbox_head,
            train_cfg=train_cfg,
            test_cfg=test_cfg,
            data_preprocessor=data_preprocessor,
            init_cfg=init_cfg)

        self.data_norm_cfg = data_preprocessor
        # gt hr img 需要的正则化参数
        self.mean = torch.tensor(self.data_norm_cfg.mean).view(-1, 1, 1)
        self.std = torch.tensor(self.data_norm_cfg.std).view(-1, 1, 1)
        
        # TODO： Waiting for mmengine support
        if use_syncbn and get_world_size() > 1:
            torch.nn.SyncBatchNorm.convert_sync_batchnorm(self)
            print_log('Using SyncBatchNorm()', 'current')

    def predict(self,
                batch_inputs: Tensor,
                batch_data_samples: SampleList,
                rescale: bool = True) -> SampleList:
        """Predict results from a batch of inputs and data samples with post-
        processing.

        Args:
            batch_inputs (Tensor): Inputs with shape (N, C, H, W).
            batch_data_samples (List[:obj:`DetDataSample`]): The Data
                Samples. It usually includes information such as
                `gt_instance`, `gt_panoptic_seg` and `gt_sem_seg`.
            rescale (bool): Whether to rescale the results.
                Defaults to True.

        Returns:
            list[:obj:`DetDataSample`]: Detection results of the
            input images. Each DetDataSample usually contain
            'pred_instances'. And the ``pred_instances`` usually
            contains following keys.

                - scores (Tensor): Classification scores, has a shape
                    (num_instance, )
                - labels (Tensor): Labels of bboxes, has a shape
                    (num_instances, ).
                - bboxes (Tensor): Has a shape (num_instances, 4),
                    the last dimension 4 arrange as (x1, y1, x2, y2).
        """
        x = self.extract_feat(batch_inputs)
        # pdb.set_trace()
        results_list = self.bbox_head.predict(
            x, batch_data_samples, rescale=rescale)
        batch_data_samples = self.add_pred_to_datasample(
            batch_data_samples, results_list)
        
        """
        各个特征图的 heatmap 可是化增加在这里
        """
        message_hub = MessageHub.get_current_instance()

        # pdb.set_trace()
        assert batch_inputs.size(0) == 1, "err must just 1 img"
        
        # ————————————————得到原图，并还原为归一化前的原图————————————————
        orig_img = batch_inputs[0].permute(1, 2, 0).cpu().numpy()
        # 这里是 hard code
        # mean = np.array([103.53, 116.28, 123.675]).reshape((1, 1, 3))
        # std = np.array([57.375, 57.12, 58.395]).reshape((1, 1, 3))
        mean = self.mean.reshape(1, 1, 3).cpu().numpy()
        std = self.std.reshape(1, 1, 3).cpu().numpy()
        orig_img = orig_img * std + mean
        # bgr->rgb
        orig_img = orig_img[..., ::-1].astype(np.uint8)  
        # ———————————————— end ————————————————

        # pdb.set_trace()
        det_visualizer = Visualizer.get_current_instance()
        heatmaps = [] # 往外传递的 heatmap 数据
        
        # 绘制backbone 后的 特征图并保存到 heatmaps
        # pdb.set_trace()
        drawn_p2 = det_visualizer.draw_featmap(x[0][0], orig_img, channel_reduction='select_max')
        heatmaps.append(drawn_p2)

        # 将数据存储起来，在 hook 中获取并存储
        message_hub.update_info('heatmaps', heatmaps)

        return batch_data_samples