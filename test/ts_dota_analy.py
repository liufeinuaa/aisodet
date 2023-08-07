"""
对 dota hr 和 lr 情况下的, 大小目标数量分布进行分析
"""


# from mmrotate.datasets import DOTADataset
from aisodet.datasets import SR_DOTADataset
from mmcv.image import imwrite
import os.path as osp
import codecs
import tqdm
import os
from aisodet.utils import register_all_modules
import pdb
import tqdm
import pickle as pkl

register_all_modules()
CLSES = ('plane', 'baseball-diamond', 'bridge', 'ground-track-field',
         'small-vehicle', 'large-vehicle', 'ship', 'tennis-court',
         'basketball-court', 'storage-tank', 'soccer-ball-field', 'roundabout',
         'harbor', 'swimming-pool', 'helicopter')
label_map = {i:c for i, c in enumerate(CLSES)}

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
    # dict( # 看随机翻转能不能同时生效...ok
    #     type='RandomFlip_sr',
    #     # prob=0.75,
    #     # direction=['horizontal', 'vertical', 'diagonal']
    #     prob=1.,
    #     direction=['horizontal', 'vertical',]
    #     ),
    # dict(
    #     # type='RandomRotate',
    #     type='RandomRotate2',
    #     # type='RandomRotate_sr',
    #     # prob=0.5,
    #     prob=1,
    #     angle_range=180,
    #     rect_obj_labels=[9, 11],
    #     rotate_type='Rotate_sr',
    #     ),
    # dict(type='mmdet.Resize', scale=(512, 512), keep_ratio=True),
    dict(type='mmdet.Resize', scale=(1024, 1024), keep_ratio=True),
    
    dict(
        # type='mmdet.RandomFlip',
        type='RandomFlip_sr',
        prob=0., # 概率归零,就是不执行这个操作其实
        direction=['horizontal', 'vertical', 'diagonal']),
    dict(
        # type='RandomRotate',
        type='RandomRotate2',
        # type='RandomRotate_sr',
        prob=0., # 概率归零,就是不执行这个操作其实
        angle_range=180,
        rect_obj_labels=[9, 11],
        rotate_type='Rotate_sr',
        ),

    dict(
        type='mmdet.Pad', 
        size=(1024, 1024),
        # size=(512, 512),
        pad_val=dict(img=(114, 114, 114))),
    dict(
        type='PackSRDetInputs',)
]

# PackSRDetInputs


# train
datasets = SR_DOTADataset(
    # 收集 lr512 下的 obj 大小分布
    # data_root = 'data3/split_lr512_dota/', # ub18 m2 ssd 下的路径
    # hr_data_path='data3/split_1024_dota1_0/trainval/images/',

    # 收集 hr 下的 obj 大小分布
    data_root = 'data3/split_1024_dota1_0/', # ub18 m2 ssd 下的路径
    hr_data_path='data3/split_lr512_dota/trainval/images/',

    ann_file='trainval/annfiles/',
    data_prefix=dict(img_path='trainval/images/'),
    # img_shape=(512, 512),
    img_shape=(1024, 1024),
    filter_cfg=dict(filter_empty_gt=True),
    pipeline=ts_pipeline,
)


# pdb.set_trace()
tiny_obj = []
small_obj = []
medium_obj = []
large_obj = []
 
tiny_areas = 16*16
small_areas = 32*32
medium_areas = 96*96

# 完整的运行一次太耗时了....因此我这边还是采用运行一次收集后保存到 pkl中
# for i, data in enumerate(datasets):
    
#     """
#     查看 img
#     """
#     # pdb.set_trace()
#     # img = data['inputs']
#     # hr_img = data['data_samples'].gt_hr_img.hr_img 
#     # # pdb.set_trace()
#     # draw_img(img, 'ts_dota_img.png')
#     # draw_img(hr_img, 'ts_dota_hrimg.png')
    
#     """
#     处理 每个具体的 obj
#     """
#     gt_instances = data['data_samples'].gt_instances
#     rbboxes = gt_instances.bboxes # 取得rbbox 的具体信息 (x, y, w, h. angle)
#     rbboxes = rbboxes.tensor

#     # 计算各个obj 的面积
#     obj_areas = rbboxes[:, 2] * rbboxes[:, 3]
#     for area in obj_areas:
#         # pdb.set_trace()
#         if area <= tiny_areas:
#             tiny_obj.append(area)
#         elif (area > tiny_areas) and (area <= small_areas):
#             small_obj.append(area)
#         elif (area > small_areas) and (area <= medium_areas):
#             medium_obj.append(area)
#         elif area > medium_areas:
#             large_obj.append(area)
#         else:
#             pass
        
#     print('process 1 image done')
#     # pdb.set_trace()
# print('get detail obj nums done')

# lr512 下的结果
# pkl.dump(tiny_obj, open('./dota_tiny_objs.pkl', 'wb'), -1) 
# pkl.dump(small_obj, open('./dota_small_objs.pkl', 'wb'), -1) 
# pkl.dump(medium_obj, open('./dota_medium_objs.pkl', 'wb'), -1) 
# pkl.dump(large_obj, open('./dota_large_objs.pkl', 'wb'), -1) 

# hr 下的结果
# pkl.dump(tiny_obj, open('./dota_hr_tiny_objs.pkl', 'wb'), -1) 
# pkl.dump(small_obj, open('./dota_hr_small_objs.pkl', 'wb'), -1) 
# pkl.dump(medium_obj, open('./dota_hr_medium_objs.pkl', 'wb'), -1) 
# pkl.dump(large_obj, open('./dota_hr_large_objs.pkl', 'wb'), -1) 

# pdb.set_trace()




# ————————————————————数据处理及绘图部分——————————————————


# 得到 lr 下的结果
tiny_obj = pkl.load(open('./dota_tiny_objs.pkl', 'rb')) 
small_obj = pkl.load(open('./dota_small_objs.pkl', 'rb')) 
medium_obj = pkl.load(open('./dota_medium_objs.pkl', 'rb')) 
large_obj = pkl.load(open('./dota_large_objs.pkl', 'rb')) 

nums_lr_tiny = len(tiny_obj)
nums_lr_small = len(small_obj)
nums_lr_medium = len(medium_obj)
nums_lr_large = len(large_obj)


# 得到 hr 下的结果
tiny_obj = pkl.load(open('./dota_hr_tiny_objs.pkl', 'rb')) 
small_obj = pkl.load(open('./dota_hr_small_objs.pkl', 'rb')) 
medium_obj = pkl.load(open('./dota_hr_medium_objs.pkl', 'rb')) 
large_obj = pkl.load(open('./dota_hr_large_objs.pkl', 'rb')) 

nums_hr_tiny = len(tiny_obj)
nums_hr_small = len(small_obj)
nums_hr_medium = len(medium_obj)
nums_hr_large = len(large_obj)



lr_sums = nums_lr_tiny + nums_lr_small + nums_lr_medium + nums_lr_large
hr_sums = nums_hr_tiny + nums_hr_small + nums_hr_medium + nums_hr_large

lr_nums = [148578, 67094, 27093, 3188] # 对应于 tiny, small, medium, large objects
hr_nums = [22180, 126398, 80699, 16676]


# 绘制直方图图
import matplotlib.pyplot as plt
import numpy as np

plt.bar()






pdb.set_trace()
