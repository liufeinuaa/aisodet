dataset_type = 'SR_AODDataset'
data_root = 'data2/UCAS_AOD/'

file_client_args = dict(backend='disk')

train_pipeline = [
    dict(type='mmdet.LoadImageFromFile', file_client_args=file_client_args),
    dict(type='LoadAnnotations_sr', with_bbox=True, box_type='qbox'),
    dict(type='mmrotate.ConvertBoxType', box_type_mapping=dict(gt_bboxes='rbox')),
    dict(type='Resize_sr_v2', scale=(416, 416), hr_scale_factor=2, keep_ratio=True, interpolation="bicubic"),
    dict(
        type='RandomFlip_sr',
        prob=0.75,
        direction=['horizontal', 'vertical', 'diagonal']),
    dict(
        type='RandomRotate2',
        prob=0.5,
        angle_range=180,
        rotate_type='Rotate_sr',
        ),
    dict( # 必须要将图片填充，到 32的倍数才行，否则没法训练和测试。。。tmd
        type='Pad_sr', 
        # size_divisor=32,
        size=(416, 416),
        pad_val=dict(img=(114, 114, 114))
        ),
    dict(type='PackSRDetInputs')
]

val_pipeline = [
    dict(type='mmdet.LoadImageFromFile', file_client_args=file_client_args),
    dict(type='Resize_sr_v2', scale=(416, 416), hr_scale_factor=2, keep_ratio=True, interpolation="bicubic"),
    # avoid bboxes being resized
    dict(type='LoadAnnotations_sr', with_bbox=True, box_type='qbox'),
    dict(type='mmrotate.ConvertBoxType', box_type_mapping=dict(gt_bboxes='rbox')),
    dict( # 必须要将图片填充，到 32的倍数才行，否则没法训练和测试。。。tmd
        type='Pad_sr', 
        # size_divisor=32,
        size=(416, 416),
        pad_val=dict(img=(114, 114, 114))
        ),
    dict(
        type='PackSRDetInputs',
        meta_keys=('img_id', 'img_path', 'ori_shape', 'img_shape',
                   'scale_factor'))
]

train_dataloader = dict(
    batch_size=8,
    num_workers=8,
    persistent_workers=True,
    sampler=dict(type='DefaultSampler', shuffle=True),
    batch_sampler=None,
    dataset=dict(
        type=dataset_type,
        data_roots=data_root,
        split_sets='ImageSets/trainval.txt',
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
        data_roots=data_root,
        split_sets='ImageSets/test.txt',
        test_mode=True,
        pipeline=val_pipeline))
test_dataloader = val_dataloader

val_evaluator = [
    dict(
        type='mmrotate.DOTAMetric',
        eval_mode='11points',
        prefix='dota_ap07',
        metric='mAP'),
    dict(
        type='mmrotate.DOTAMetric', eval_mode='area', prefix='dota_ap12', metric='mAP'),
]
test_evaluator = val_evaluator