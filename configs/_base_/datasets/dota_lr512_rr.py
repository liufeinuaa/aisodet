# dataset settings
dataset_type = 'mmrotate.DOTADataset'
# data_root = 'data/split_ss_dota/'
# data_root = 'data/split_1024_dota1_0/' # 机械硬盘下的路径
data_root = 'data2/split_lr512_dota/' # m2 ssd 下的路径

"""
可以看到将数据集放到ssd中后，训来需要的 data_time 降低了一倍，即能加速近一倍的数据读取销量
"""

file_client_args = dict(backend='disk')

train_pipeline = [
    dict(type='mmdet.LoadImageFromFile', file_client_args=file_client_args),
    dict(type='mmdet.LoadAnnotations', with_bbox=True, box_type='qbox'),
    dict(type='mmrotate.ConvertBoxType', box_type_mapping=dict(gt_bboxes='rbox')),
    # dict(type='mmdet.Resize', scale=(1024, 1024), keep_ratio=True),
    dict(type='mmdet.Resize', scale=(512, 512), keep_ratio=True),
    dict(
        type='mmdet.RandomFlip',
        prob=0.75,
        direction=['horizontal', 'vertical', 'diagonal']),
    dict(
        # type='RandomRotate',
        type='RandomRotate2',
        # type='RandomRotate_sr',
        prob=0.5,
        angle_range=180,
        rect_obj_labels=[9, 11],
        # rotate_type='Rotate_sr',
        ),
    dict(
        type='mmdet.Pad', 
        # size=(1024, 1024),
        size=(512, 512),
        pad_val=dict(img=(114, 114, 114))),
    dict(type='mmdet.PackDetInputs')
]
val_pipeline = [
    dict(type='mmdet.LoadImageFromFile', file_client_args=file_client_args),
    dict(type='mmdet.Resize', scale=(512, 512), keep_ratio=True),
    # avoid bboxes being resized
    dict(type='mmdet.LoadAnnotations', with_bbox=True, box_type='qbox'),
    dict(type='mmrotate.ConvertBoxType', box_type_mapping=dict(gt_bboxes='rbox')),
    dict(
        type='mmdet.Pad', size=(512, 512),
        pad_val=dict(img=(114, 114, 114))),
    dict(
        type='mmdet.PackDetInputs',
        meta_keys=('img_id', 'img_path', 'ori_shape', 'img_shape',
                   'scale_factor'))
]
test_pipeline = [
    dict(type='mmdet.LoadImageFromFile', file_client_args=file_client_args),
    dict(type='mmdet.Resize', scale=(512, 512), keep_ratio=True),
    dict(
        type='mmdet.Pad', size=(512, 512),
        pad_val=dict(img=(114, 114, 114))),
    dict(
        type='mmdet.PackDetInputs',
        meta_keys=('img_id', 'img_path', 'ori_shape', 'img_shape',
                   'scale_factor'))
]
train_dataloader = dict(
    # 默认的bs配置
    batch_size=8,
    num_workers=8,
    persistent_workers=True,
    sampler=dict(type='DefaultSampler', shuffle=True),
    batch_sampler=None,
    dataset=dict(
        type=dataset_type,
        data_root=data_root,
        ann_file='trainval/annfiles/',
        data_prefix=dict(img_path='trainval/images/'),
        # img_shape=(1024, 1024),
        img_shape=(512, 512),
        filter_cfg=dict(filter_empty_gt=True),
        pipeline=train_pipeline))
val_dataloader = dict(
    batch_size=1,
    num_workers=2,
    persistent_workers=True,
    drop_last=False,
    sampler=dict(type='DefaultSampler', shuffle=False),
    dataset=dict(
        type=dataset_type,
        data_root=data_root,
        # ann_file='trainval/annfiles/',
        # data_prefix=dict(img_path='trainval/images/'),
        ann_file='val/annfiles/',
        data_prefix=dict(img_path='val/images/'),
        # img_shape=(1024, 1024),
        img_shape=(512, 512),
        test_mode=True,
        pipeline=val_pipeline))
val_evaluator = dict(type='mmrotate.DOTAMetric', metric='mAP')

# 不提交服务器进行测试，只是想离线进行val
# test_dataloader = val_dataloader
# test_evaluator = val_evaluator

# 提交服务器进行测试的时候的代码
# inference on test dataset and format the output results
# for submission. Note: the test set has no annotation.
test_dataloader = dict(
    batch_size=1,
    num_workers=2,
    persistent_workers=True,
    drop_last=False,
    sampler=dict(type='DefaultSampler', shuffle=False),
    dataset=dict(
        type=dataset_type,
        data_root=data_root,
        data_prefix=dict(img_path='test/images/'),
        # img_shape=(1024, 1024),
        # img_shape=(512, 512), # 这里给不给参数都不会影响后面的模型参数
        test_mode=True,
        pipeline=test_pipeline))
test_evaluator = dict(
    type='mmrotate.DOTAMetric',
    format_only=True,
    merge_patches=True,
    outfile_prefix='./work_dirs/dota/Task1')
