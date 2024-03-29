# Copyright (c) OpenMMLab. All rights reserved.
import glob
import os.path as osp
from typing import List, Tuple

from mmengine.dataset import BaseDataset

from aisodet.registry import DATASETS
import pdb
import copy

@DATASETS.register_module()
class SR_DOTADataset(BaseDataset):
    """
    我这里在dota 数据集结构的基础上，引入 sr 分支需要 hr img

    DOTA dataset for detection.

    Note: ``ann_file`` in DOTADataset is different from the BaseDataset.
    In BaseDataset, it is the path of an annotation file. In DOTADataset,
    it is the path of a folder containing XML files.

    Args:
        img_shape (tuple[int]): The shape of images. Due to the huge size
            of the remote sensing image, we will cut it into slices with
            the same shape. Defaults to (1024, 1024).
        diff_thr (int): The difficulty threshold of ground truth. Bboxes
            with difficulty higher than it will be ignored. The range of this
            value should be non-negative integer. Defaults to 100.
    """

    METAINFO = {
        # 'CLASSES': # 旧版的key
        'classes': # 新版的key
        ('plane', 'baseball-diamond', 'bridge', 'ground-track-field',
         'small-vehicle', 'large-vehicle', 'ship', 'tennis-court',
         'basketball-court', 'storage-tank', 'soccer-ball-field', 'roundabout',
         'harbor', 'swimming-pool', 'helicopter'),
        # PALETTE is a list of color tuples, which is used for visualization.
        'PALETTE': [(165, 42, 42), (189, 183, 107), (0, 255, 0), (255, 0, 0),
                    (138, 43, 226), (255, 128, 0), (255, 0, 255),
                    (0, 255, 255), (255, 193, 193), (0, 51, 153),
                    (255, 250, 205), (0, 139, 139), (255, 255, 0),
                    (147, 116, 116), (0, 0, 255)]
    }

    def __init__(self,
                 hr_data_path: str = '', # 新增的高分辨率图像的路径
                 img_shape: Tuple[int, int] = (512, 512), # 在sr_dota 中默认使用 512x512 的原始图像
                 diff_thr: int = 100,
                 **kwargs) -> None:
        self.hr_data_path = hr_data_path ## 新增的高分辨率图像的路径 
        self.img_shape = img_shape
        self.diff_thr = diff_thr
        super().__init__(**kwargs)
        # pdb.set_trace()
        

    def load_data_list(self) -> List[dict]:
        """Load annotations from an annotation file named as ``self.ann_file``
        Returns:
            List[dict]: A list of annotation.
        """  # noqa: E501
        cls_map = {c: i
                   for i, c in enumerate(self.metainfo['classes'])
                   }  # in mmdet v2.0 label is 0-based
        
        # pdb.set_trace()
        data_list = []
        if self.ann_file == '': # ann flie 为空的时候，就是仅加载img （test时候）
            img_files = glob.glob(
                osp.join(self.data_prefix['img_path'], '*.png'))
            for img_path in img_files:
                data_info = {}
                data_info['img_path'] = img_path
                img_name = osp.split(img_path)[1]
                data_info['file_name'] = img_name
                img_id = img_name[:-4]
                data_info['img_id'] = img_id
                data_info['height'] = self.img_shape[0]
                data_info['width'] = self.img_shape[1]

                instance = dict(bbox=[], bbox_label=[], ignore_flag=0)
                data_info['instances'] = [instance]
                data_list.append(data_info)

            return data_list
        else:
            txt_files = glob.glob(osp.join(self.ann_file, '*.txt'))
            if len(txt_files) == 0:
                raise ValueError('There is no txt file in '
                                 f'{self.ann_file}')
            for txt_file in txt_files:
                data_info = {}
                img_id = osp.split(txt_file)[1][:-4]
                data_info['img_id'] = img_id
                img_name = img_id + '.png'
                data_info['file_name'] = img_name
                data_info['img_path'] = osp.join(self.data_prefix['img_path'],
                                                 img_name)
                data_info['height'] = self.img_shape[0]
                data_info['width'] = self.img_shape[1]

                # 自己新增的加入hr img 路径信息
                data_info['hr_path'] = osp.join(self.hr_data_path, img_name)

                instances = []
                with open(txt_file) as f:
                    s = f.readlines()
                    for si in s:
                        instance = {}
                        bbox_info = si.split()
                        instance['bbox'] = [float(i) for i in bbox_info[:8]]
                        cls_name = bbox_info[8]
                        instance['bbox_label'] = cls_map[cls_name]
                        difficulty = int(bbox_info[9])
                        if difficulty > self.diff_thr:
                            instance['ignore_flag'] = 1
                        else:
                            instance['ignore_flag'] = 0
                        instances.append(instance)
                data_info['instances'] = instances
                data_list.append(data_info)

            return data_list

    def filter_data(self) -> List[dict]:
        """Filter annotations according to filter_cfg.

        Returns:
            List[dict]: Filtered results.
        """
        if self.test_mode:
            return self.data_list

        filter_empty_gt = self.filter_cfg.get('filter_empty_gt', False) \
            if self.filter_cfg is not None else False

        valid_data_infos = []
        for i, data_info in enumerate(self.data_list):
            if filter_empty_gt and len(data_info['instances']) == 0:
                continue
            valid_data_infos.append(data_info)

        return valid_data_infos

    def get_cat_ids(self, idx: int) -> List[int]:
        """Get DOTA category ids by index.

        Args:
            idx (int): Index of data.
        Returns:
            List[int]: All categories in the image of specified index.
        """

        instances = self.get_data_info(idx)['instances']
        return [instance['bbox_label'] for instance in instances]
