"""
自编的一个支持引入sr分支的检测架构输入变换
"""

import numpy as np
from mmcv.transforms import to_tensor
from mmcv.transforms.base import BaseTransform
from mmengine.structures import InstanceData, PixelData, BaseDataElement


from mmdet.structures import DetDataSample
from mmdet.structures.bbox import BaseBoxes

from aisodet.registry import TRANSFORMS
import pdb



@TRANSFORMS.register_module()
class PackSRDetInputs(BaseTransform):
    """
    在 mmdet 的PackDetInputs 基础上，拓展增加一个hr image 作为个 pixel mask gt 用于 SR 中
    
    
    Pack the inputs data for the detection / semantic segmentation /
    panoptic segmentation.

    The ``img_meta`` item is always populated.  The contents of the
    ``img_meta`` dictionary depends on ``meta_keys``. By default this includes:

        - ``img_id``: id of the image

        - ``img_path``: path to the image file

        - ``ori_shape``: original shape of the image as a tuple (h, w)

        - ``img_shape``: shape of the image input to the network as a tuple \
            (h, w).  Note that images may be zero padded on the \
            bottom/right if the batch tensor is larger than this shape.

        - ``scale_factor``: a float indicating the preprocessing scale

        - ``flip``: a boolean indicating if image flip transform was used

        - ``flip_direction``: the flipping direction

    Args:
        meta_keys (Sequence[str], optional): Meta keys to be converted to
            ``mmcv.DataContainer`` and collected in ``data[img_metas]``.
            Default: ``('img_id', 'img_path', 'ori_shape', 'img_shape',
            'scale_factor', 'flip', 'flip_direction')``
    """
    mapping_table = {
        'gt_bboxes': 'bboxes',
        'gt_bboxes_labels': 'labels',
        'gt_masks': 'masks'
    }
    def __init__(self,
                meta_keys=('img_id', 'img_path', 'ori_shape', 'img_shape',
                            'scale_factor', 'flip', 'flip_direction')) -> None:
        self.meta_keys = meta_keys
    
    def transform(self, results: dict) -> dict:
        """Method to pack the input data.

        Args:
            results (dict): Result dict from the data pipeline.

        Returns:
            dict:

            - 'inputs' (obj:`torch.Tensor`): The forward data of models.
            - 'data_sample' (obj:`DetDataSample`): The annotation info of the
                sample.
        """
        packed_results = dict()
        if 'img' in results:
            img = results['img']
            if len(img.shape) < 3:
                img = np.expand_dims(img, -1)
            # To improve the computational speed by by 3-5 times, apply:
            # If image is not contiguous, use
            # `numpy.transpose()` followed by `numpy.ascontiguousarray()`
            # If image is already contiguous, use
            # `torch.permute()` followed by `torch.contiguous()`
            # Refer to https://github.com/open-mmlab/mmdetection/pull/9533
            # for more details
            if not img.flags.c_contiguous:
                img = np.ascontiguousarray(img.transpose(2, 0, 1))
                img = to_tensor(img)
            else:
                img = to_tensor(img).permute(2, 0, 1).contiguous()

            packed_results['inputs'] = to_tensor(img)

        if 'gt_ignore_flags' in results:
            valid_idx = np.where(results['gt_ignore_flags'] == 0)[0]
            ignore_idx = np.where(results['gt_ignore_flags'] == 1)[0]

        # 增加的用于sr 的 hr img
        if 'gt_hr_img' in results:
            hr_img = results['gt_hr_img']
            if len(hr_img.shape) < 3:
                hr_img = np.expand_dims(hr_img, -1)
            
            # hr_img = np.ascontiguousarray(hr_img.transpose(2, 0, 1))
            if not hr_img.flags.c_contiguous:
                hr_img = np.ascontiguousarray(hr_img.transpose(2, 0, 1))
                hr_img = to_tensor(hr_img)
            else:
                hr_img = to_tensor(hr_img).permute(2, 0, 1).contiguous()


        # pdb.set_trace()
        data_sample = SRDetDataSample() # 这里也是新加的
        instance_data = InstanceData()
        ignore_instance_data = InstanceData()

        
        # pdb.set_trace()
        for key in self.mapping_table.keys():
            if key not in results:
                continue
            if key == 'gt_masks' or isinstance(results[key], BaseBoxes):
                if 'gt_ignore_flags' in results:
                    instance_data[
                        self.mapping_table[key]] = results[key][valid_idx]
                    ignore_instance_data[
                        self.mapping_table[key]] = results[key][ignore_idx]
                else:
                    instance_data[self.mapping_table[key]] = results[key]
            else:
                if 'gt_ignore_flags' in results:
                    instance_data[self.mapping_table[key]] = to_tensor(
                        results[key][valid_idx])
                    ignore_instance_data[self.mapping_table[key]] = to_tensor(
                        results[key][ignore_idx])
                else:
                    instance_data[self.mapping_table[key]] = to_tensor(
                        results[key])

        data_sample.gt_instances = instance_data
        data_sample.ignored_instances = ignore_instance_data

        if 'proposals' in results:
            proposals = InstanceData(
                bboxes=to_tensor(results['proposals']),
                scores=to_tensor(results['proposals_scores']))
            data_sample.proposals = proposals

        if 'gt_seg_map' in results:
            gt_sem_seg_data = dict(
                sem_seg=to_tensor(results['gt_seg_map'][None, ...].copy()))
            data_sample.gt_sem_seg = PixelData(**gt_sem_seg_data)

        # 新增的hr img处理放在这边
        if 'gt_hr_img' in results:
            gt_hr_img_data = dict(
                hr_img = hr_img)
            data_sample.gt_hr_img = PixelData(**gt_hr_img_data)
        
        # pdb.set_trace()
        img_meta = {}
        for key in self.meta_keys:
            assert key in results, f'`{key}` is not found in `results`, ' \
                f'the valid keys are {list(results)}.'
            img_meta[key] = results[key]

        data_sample.set_metainfo(img_meta)
        packed_results['data_samples'] = data_sample

        # pdb.set_trace()
        return packed_results



# 先定义一个增加了sr分支的数据sample
class SRDetDataSample(DetDataSample):
    @property
    def gt_hr_img(self) -> PixelData:
        return self._gt_hr_img

    @gt_hr_img.setter
    def gt_hr_img(self, value: PixelData):
        self.set_field(value, '_gt_hr_img', dtype=PixelData)

    @gt_hr_img.deleter
    def gt_hr_img(self):
        del self._gt_hr_img

    # 再新增一个sr分支预测的模型输出
    @property
    def pred_sr_img(self) -> PixelData:
        return self._pred_sr_img
    
    @pred_sr_img.setter
    def pred_sr_img(self, value: PixelData):
        self.set_field(value, '_pred_sr_img', dtype=PixelData)

    @pred_sr_img.deleter
    def pred_sr_img(self):
        del self._pred_sr_img