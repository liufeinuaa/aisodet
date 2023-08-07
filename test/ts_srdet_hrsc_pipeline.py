"""
将hr图像作为gt的一部分插入data sample中
"""


# from mmrotate.datasets import DOTADataset
# from aisodet.datasets import SR_DOTADataset
from aisodet.datasets import SR_HRSCDataset
from mmcv.image import imwrite
import os.path as osp
import codecs
import tqdm
import os
from aisodet.utils import register_all_modules
import pdb

register_all_modules()
# CLSES = ('plane', 'baseball-diamond', 'bridge', 'ground-track-field',
#          'small-vehicle', 'large-vehicle', 'ship', 'tennis-court',
#          'basketball-court', 'storage-tank', 'soccer-ball-field', 'roundabout',
#          'harbor', 'swimming-pool', 'helicopter')
# label_map = {i:c for i, c in enumerate(CLSES)}

def draw_img(img, path):
    img = img.permute(1,2,0)
    img_np = img.numpy()
    imwrite(img_np, path)


ts_pipeline = [
    dict(type='mmdet.LoadImageFromFile'),
    # dict(type='mmdet.LoadAnnotations', 
    #     with_bbox=True, box_type='qbox'),
    # dict(
    #     type='mmdet.PackDetInputs',
    #     meta_keys=('img_id', 'img_path', 'ori_shape', 'img_shape',
    #                ))
    dict(type='LoadAnnotations_sr', 
        with_bbox=True, box_type='qbox', with_hr_img=True),
    dict(type='mmrotate.ConvertBoxType', box_type_mapping=dict(gt_bboxes='rbox')),
    # dict(type='Resize_sr', scale=(400, 400), hr_scale_factor=2., keep_ratio=True, interpolation="bicubic"),
    dict(type='Resize_sr', scale=(400, 400), hr_scale_factor=2, keep_ratio=True, 
         interpolation="bicubic"
         ),
    dict( # 看随机翻转能不能同时生效...ok
        type='RandomFlip_sr',
        # prob=0.75,
        # direction=['horizontal', 'vertical', 'diagonal']
        prob=1.,
        direction=['horizontal', 'vertical',]
        ),
    dict(
        # type='RandomRotate',
        type='RandomRotate2',
        # type='RandomRotate_sr',
        # prob=0.5,
        prob=1,
        angle_range=180,
        rect_obj_labels=[9, 11],
        rotate_type='Rotate_sr',
        ),
    dict(
        type='Pad_sr', 
        size_divisor=32,
        ),
    dict(
        type='PackSRDetInputs',
        # meta_keys=('img_id', 'img_path', 'ori_shape', 'img_shape',
        #             'gt_hr_img_shape', 'classes'
        #            )
                   )
]

# PackSRDetInputs


# train
datasets = SR_HRSCDataset(
    
    data_root='data3/HRSC2016/',

    ann_file='ImageSets/trainval.txt',
    data_prefix=dict(sub_data_root='FullDataSet/'),
    filter_cfg=dict(filter_empty_gt=True),
    pipeline=ts_pipeline,
)


# pdb.set_trace()

for i, data in enumerate(datasets):
    # pdb.set_trace()
    img = data['inputs']
    hr_img = data['data_samples'].gt_hr_img.hr_img 
    
    # pdb.set_trace()
    draw_img(img, 'ts_srdet_hrsc_img.png')
    draw_img(hr_img, 'ts_srdet_hrsc_hrimg.png')
    

    pdb.set_trace()




