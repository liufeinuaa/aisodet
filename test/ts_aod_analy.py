"""
对 dota hr 和 lr 情况下的, 大小目标数量分布进行分析
"""


# from mmrotate.datasets import DOTADataset
from aisodet.datasets import SR_DOTADataset
from aisodet.datasets import SR_AODDataset
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
CLSES = ('car', 'airplane')
label_map = {i:c for i, c in enumerate(CLSES)}

def draw_img(img, path):
    img = img.permute(1,2,0)
    img_np = img.numpy()
    imwrite(img_np, path)

file_client_args = dict(backend='disk')

hr_pipeline = [
    dict(type='mmdet.LoadImageFromFile', file_client_args=file_client_args),
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
    dict(type='mmdet.Resize', scale=(832, 832), keep_ratio=True),
    
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
        # rect_obj_labels=[9, 11],
        rotate_type='Rotate_sr',
        ),

    dict(
        type='mmdet.Pad', 
        size=(832, 832),
        # size=(512, 512),
        pad_val=dict(img=(114, 114, 114))),
    dict(
        type='PackSRDetInputs',)
]



lr_pipeline = [
    dict(type='mmdet.LoadImageFromFile', file_client_args=file_client_args),
    # dict(type='mmdet.LoadAnnotations', 
    #     with_bbox=True, box_type='qbox'),
    # dict(
    #     type='mmdet.PackDetInputs',
    #     meta_keys=('img_id', 'img_path', 'ori_shape', 'img_shape',
    #                ))
    dict(type='LoadAnnotations_sr', 
        with_bbox=True, box_type='qbox', with_hr_img=True),
    dict(type='mmrotate.ConvertBoxType', box_type_mapping=dict(gt_bboxes='rbox')),
    
    # dict(type='mmdet.Resize', scale=(832, 832), keep_ratio=True),
    dict(type='mmdet.Resize', scale=(416, 416), keep_ratio=True),
    
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
        # rect_obj_labels=[9, 11],
        rotate_type='Rotate_sr',
        ),

    dict(
        type='mmdet.Pad', 
        # size=(832, 832),
        size=(416, 416),
        pad_val=dict(img=(114, 114, 114))),
    dict(
        type='PackSRDetInputs',)
]



# PackSRDetInputs


# train
datasets = SR_AODDataset(
    data_roots = 'data3/UCAS_AOD/',
    split_sets='ImageSets/trainval.txt',
    filter_cfg=dict(filter_empty_gt=True),
    # pipeline=hr_pipeline, # 收集 hr 上的信息
    pipeline=lr_pipeline, # 收集 lr 上的信息
)



def get_object_size_dist(datasets, type='hr'):

    # pdb.set_trace()
    tiny_obj = []
    small_obj = []
    medium_obj = []
    large_obj = []
    
    tiny_areas = 16*16
    small_areas = 32*32
    medium_areas = 96*96

    # 完整的运行一次太耗时了....因此我这边还是采用运行一次收集后保存到 pkl中
    for i, data in enumerate(datasets):
        
        """
        查看 img
        """
        # pdb.set_trace()
        # img = data['inputs']
        # hr_img = data['data_samples'].gt_hr_img.hr_img 
        # # pdb.set_trace()
        # draw_img(img, 'ts_dota_img.png')
        # draw_img(hr_img, 'ts_dota_hrimg.png')
        
        """
        处理 每个具体的 obj
        """
        gt_instances = data['data_samples'].gt_instances
        rbboxes = gt_instances.bboxes # 取得rbbox 的具体信息 (x, y, w, h. angle)
        rbboxes = rbboxes.tensor

        # 计算各个obj 的面积
        obj_areas = rbboxes[:, 2] * rbboxes[:, 3]
        for area in obj_areas:
            # pdb.set_trace()
            if area <= tiny_areas:
                tiny_obj.append(area)
            elif (area > tiny_areas) and (area <= small_areas):
                small_obj.append(area)
            elif (area > small_areas) and (area <= medium_areas):
                medium_obj.append(area)
            elif area > medium_areas:
                large_obj.append(area)
            else:
                pass
            
        print('process 1 image done')
        # pdb.set_trace()
    print('get detail obj nums done')

    # lr512 下的结果
    pkl.dump(tiny_obj, open(f'./aod_{type}_tiny_objs.pkl', 'wb'), -1) 
    pkl.dump(small_obj, open(f'./aod_{type}_small_objs.pkl', 'wb'), -1) 
    pkl.dump(medium_obj, open(f'./aod_{type}_medium_objs.pkl', 'wb'), -1) 
    pkl.dump(large_obj, open(f'./aod_{type}_large_objs.pkl', 'wb'), -1) 







# hr 下的结果
# pkl.dump(tiny_obj, open('./dota_hr_tiny_objs.pkl', 'wb'), -1) 
# pkl.dump(small_obj, open('./dota_hr_small_objs.pkl', 'wb'), -1) 
# pkl.dump(medium_obj, open('./dota_hr_medium_objs.pkl', 'wb'), -1) 
# pkl.dump(large_obj, open('./dota_hr_large_objs.pkl', 'wb'), -1) 

# pdb.set_trace()


# ————————————————————数据处理及绘图部分——————————————————

# get_object_size_dist(datasets, type='hr')
# get_object_size_dist(datasets, type='lr')

# 得到 lr 下的结果
tiny_obj = pkl.load(open('./aod_lr_tiny_objs.pkl', 'rb')) 
small_obj = pkl.load(open('./aod_lr_small_objs.pkl', 'rb')) 
medium_obj = pkl.load(open('./aod_lr_medium_objs.pkl', 'rb')) 
large_obj = pkl.load(open('./aod_lr_large_objs.pkl', 'rb')) 

nums_lr_tiny = len(tiny_obj)
nums_lr_small = len(small_obj)
nums_lr_medium = len(medium_obj)
nums_lr_large = len(large_obj)

# pdb.set_trace()

# 得到 hr 下的结果
tiny_obj = pkl.load(open('./aod_hr_tiny_objs.pkl', 'rb')) 
small_obj = pkl.load(open('./aod_hr_small_objs.pkl', 'rb')) 
medium_obj = pkl.load(open('./aod_hr_medium_objs.pkl', 'rb')) 
large_obj = pkl.load(open('./aod_hr_large_objs.pkl', 'rb')) 

nums_hr_tiny = len(tiny_obj)
nums_hr_small = len(small_obj)
nums_hr_medium = len(medium_obj)
nums_hr_large = len(large_obj)

# pdb.set_trace()

lr_sums = nums_lr_tiny + nums_lr_small + nums_lr_medium + nums_lr_large
hr_sums = nums_hr_tiny + nums_hr_small + nums_hr_medium + nums_hr_large

pdb.set_trace()

# dota 上的结果
# lr_nums = [148578, 67094, 27093, 3188] # 对应于 tiny, small, medium, large objects
# hr_nums = [22180, 126398, 80699, 16676]

# aod 上的结果
lr_nums = [5587, 3492, 908, 0] # 对应于 tiny, small, medium, large objects
hr_nums = [109, 5476, 4301, 101]


