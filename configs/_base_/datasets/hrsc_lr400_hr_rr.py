# dataset settings
dataset_type = 'SR_HRSCDataset'
data_root = 'data3/HRSC2016/'

file_client_args = dict(backend='disk')

train_pipeline = [
    dict(type='mmdet.LoadImageFromFile', file_client_args=file_client_args),
    # dict(type='mmdet.LoadAnnotations', with_bbox=True, box_type='qbox'),
    dict(type='LoadAnnotations_sr', with_bbox=True, box_type='qbox'),
    dict(type='mmrotate.ConvertBoxType', box_type_mapping=dict(gt_bboxes='rbox')),
    dict(type='Resize_sr', scale=(256, 256), hr_scale_factor=2, keep_ratio=True, interpolation="bicubic"),
    dict(
        # type='mmdet.RandomFlip',
        type='RandomFlip_sr',
        prob=0.75,
        direction=['horizontal', 'vertical', 'diagonal']),
    # dict(type='mmdet.PackDetInputs')
    # dict(
    #     # type='RandomRotate',
    #     type='RandomRotate2',
    #     # type='RandomRotate_sr',
    #     # prob=0.5,
    #     prob=0.5,
    #     angle_range=180,
    #     # rect_obj_labels=[9, 11],
    #     rotate_type='Rotate_sr',
    #     ),
    dict( # 必须要将图片填充，到 32的倍数才行，否则没法训练和测试。。。tmd
        type='Pad_sr', 
        size_divisor=32,
        ),
    dict(type='PackSRDetInputs')
]
val_pipeline = [
    dict(type='mmdet.LoadImageFromFile', file_client_args=file_client_args),
    # dict(type='mmdet.Resize', scale=(800, 512), keep_ratio=True),
    dict(type='Resize_sr', scale=(256, 256), hr_scale_factor=2, keep_ratio=True, interpolation="bicubic"),
    # avoid bboxes being resized
    dict(type='LoadAnnotations_sr', with_bbox=True, box_type='qbox'),
    dict(type='mmrotate.ConvertBoxType', box_type_mapping=dict(gt_bboxes='rbox')),
    # dict(type='Resize_sr', scale=(256, 256), hr_scale_factor=2, keep_ratio=True, interpolation="bicubic"),
    dict( # 必须要将图片填充，到 32的倍数才行，否则没法训练和测试。。。tmd
        type='Pad_sr', 
        size_divisor=32,
        ),
    dict(
        type='PackSRDetInputs',
        meta_keys=('img_id', 'img_path', 'ori_shape', 'img_shape',
                   'scale_factor'))
    # dict(type='PackSRDetInputs')
]
# test_pipeline = [
#     dict(type='mmdet.LoadImageFromFile', file_client_args=file_client_args),
#     dict(type='mmdet.Resize', scale=(800, 512), keep_ratio=True),
#     dict(
#         type='PackSRDetInputs',
#         meta_keys=('img_id', 'img_path', 'ori_shape', 'img_shape',
#                    'scale_factor'))
# ]
train_dataloader = dict(
    # batch_size=2,
    # num_workers=2,
    batch_size=8,
    num_workers=8,
    persistent_workers=True,
    sampler=dict(type='DefaultSampler', shuffle=True),
    batch_sampler=None,
    dataset=dict(
        type=dataset_type,
        data_root=data_root,
        ann_file='ImageSets/trainval.txt',
        data_prefix=dict(sub_data_root='FullDataSet/'),
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
        ann_file='ImageSets/test.txt',
        data_prefix=dict(sub_data_root='FullDataSet/'),
        test_mode=True,
        pipeline=val_pipeline))
test_dataloader = val_dataloader

val_evaluator = dict(type='mmrotate.DOTAMetric', metric='mAP')
test_evaluator = val_evaluator










# ————————————————————————————————————

# dataset settings
# dataset_type = 'HRSCDataset'
# # data_root = 'data/hrsc/'
# data_root = 'data/HRSC2016/'
# img_norm_cfg = dict(
#     mean=[123.675, 116.28, 103.53], std=[58.395, 57.12, 57.375], to_rgb=True)
# train_pipeline = [
#     dict(type='LoadImageFromFile'),
#     dict(type='LoadAnnotations', with_bbox=True),
#     dict(type='RResize', img_scale=(800, 800)),
#     dict(type='RRandomFlip', flip_ratio=0.5),
#     dict(type='Normalize', **img_norm_cfg),
#     dict(type='Pad', size_divisor=32),
#     dict(type='DefaultFormatBundle'),
#     dict(type='Collect', keys=['img', 'gt_bboxes', 'gt_labels'])
# ]
# test_pipeline = [
#     dict(type='LoadImageFromFile'),
#     dict(
#         type='MultiScaleFlipAug',
#         img_scale=(800, 800),
#         flip=False,
#         transforms=[
#             dict(type='RResize'),
#             dict(type='Normalize', **img_norm_cfg),
#             dict(type='Pad', size_divisor=32),
#             dict(type='DefaultFormatBundle'),
#             dict(type='Collect', keys=['img'])
#         ])
# ]
# data = dict(
#     samples_per_gpu=2,
#     workers_per_gpu=2,
#     train=dict(
#         type=dataset_type,
#         classwise=False, # 这边这个只是控制类别的，是否都为ship类，而不是bbox的旋转方向的控制
#         ann_file=data_root + 'ImageSets/trainval.txt',
#         ann_subdir=data_root + 'FullDataSet/Annotations/',
#         img_subdir=data_root + 'FullDataSet/AllImages/',
#         pipeline=train_pipeline),
#     val=dict(
#         type=dataset_type,
#         classwise=False,
#         ann_file=data_root + 'ImageSets/trainval.txt',
#         ann_subdir=data_root + 'FullDataSet/Annotations/',
#         img_subdir=data_root + 'FullDataSet/AllImages/',
#         pipeline=test_pipeline),
#     test=dict(
#         type=dataset_type,
#         classwise=False,
#         ann_file=data_root + 'ImageSets/test.txt',
#         ann_subdir=data_root + 'FullDataSet/Annotations/',
#         img_subdir=data_root + 'FullDataSet/AllImages/',
#         pipeline=test_pipeline))