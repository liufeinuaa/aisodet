import glob
import os.path as osp
from typing import List, Tuple, Optional, Union
from mmengine.dataset import BaseDataset
from aisodet.registry import DATASETS
import mmcv
import numpy as np
import torch
from mmengine.dataset import BaseDataset
from mmengine.fileio import FileClient, list_from_file
from mmrotate.structures.bbox import rbox2qbox

import pdb
import copy

@DATASETS.register_module()
class SR_AODDataset(BaseDataset):
    """
    结合 dota 和 hrsc 完成对 aod 数据集的读取，并且引入 sr 分支需要 hr img
    
    """
    METAINFO = {
        'classes': # 新版的key
        ('car', 'airplane'),
        # PALETTE is a list of color tuples, which is used for visualization.
        'PALETTE': [(165, 42, 42), (189, 183, 107)]
    }

    def __init__(self, 
                 data_roots: str = '',
                 split_sets: str = '',
                 img_subdir: str = 'AllImages',
                 ann_subdir: str = 'Annotations',
                 file_client_args: dict = dict(backend='disk'),
                 **kwargs):
        
        self.data_roots = data_roots
        self.split_sets = split_sets
        self.img_subdir = img_subdir
        self.ann_subdir = ann_subdir
        self.file_client_args = file_client_args
        self.file_client = FileClient(**self.file_client_args)
        super().__init__(**kwargs)


    def load_data_list(self) -> List[dict]:
        """Load annotations from an annotation file name
        Returns:
            List[dict]: A list of annotation.
        """
        cls_map = {c: i
                   for i, c in enumerate(self.metainfo['classes'])
                   }  # in mmdet v2.0 label is 0-based

        self.ann_file = osp.join(self.data_roots, self.split_sets)

        # pdb.set_trace()
        data_list = []
        img_ids = list_from_file(
            self.ann_file, file_client_args=self.file_client_args
        )

        for img_id in img_ids:
            data_info = {}

            file_name = osp.join(self.data_roots, self.img_subdir, f'{img_id}.png')
            txt_path = osp.join(self.data_roots, self.ann_subdir, f'{img_id}.txt')

            data_info['img_path'] = file_name
            data_info['img_id'] = img_id
            data_info['file_name'] = f'{img_id}.png'

# 新增 hr img 的路径。。。。我这边就让其和 lr img 的一致，lr 通过双三次下采样直接得出
            data_info['hr_path'] = file_name

            if osp.getsize(txt_path) == 0: # 保证不是空的txt
                continue
            
            instances = []
            with open(txt_path) as f: # 打开标注ann 的txt文件
                tmpf = f.readlines()
                for ann in tmpf:
                    instance = {}
                    # pdb.set_trace()
                    bbox_info = ann.split()
                    instance['bbox'] = [float(i) for i in bbox_info[1:9]]
                    cls_name = bbox_info[0]
                    instance['bbox_label'] = cls_map[cls_name]
                    instance['ignore_flag'] = 0

                    instances.append(instance)

            data_info['instances'] = instances

            data_list.append(data_info)

        # pdb.set_trace()
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
