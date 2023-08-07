_base_ = [
    '../../../../configs/_base_/datasets/dota_lr512_hr_rr.py',
    "../../../../configs/_base_/default_runtime_ema.py",
    "../../../../configs/_base_/schedules/schedule_3x_adamw.py",
]

# imagenet 预训练ckpt的下载地址
checkpoint = 'https://download.openmmlab.com/mmdetection/v3.0/rtmdet/cspnext_rsb_pretrain/cspnext-tiny_imagenet_600e.pth' 

# 自编模块的加载
custom_imports = dict(
    imports=['projects.esrtmdet',], allow_failed_imports=False)

angle_version = 'le90'
model = dict(
    type='ESRTMDet',
    data_preprocessor=dict(
        type='mmdet.DetDataPreprocessor',
        mean=[103.53, 116.28, 123.675],
        std=[57.375, 57.12, 58.395],
        bgr_to_rgb=False,
        boxtype2tensor=False,
        batch_augments=None),

    backbone=dict(
        type='mmdet.CSPNeXt',
        arch='P5',
        expand_ratio=0.5,
        deepen_factor=0.167,
        widen_factor=0.375,
        channel_attention=True,
        out_indices=(0, 1, 2, 3, 4), 
        norm_cfg=dict(type='SyncBN'),
        act_cfg=dict(type='SiLU'),
        init_cfg=dict(
            type='Pretrained', prefix='backbone.', 
            checkpoint=checkpoint,
            )),
    neck=dict(
        type='mmdet.CSPNeXtPAFPN',
        in_channels=[96, 192, 384],
        out_channels=96,
        num_csp_blocks=1,
        expand_ratio=0.5,
        norm_cfg=dict(type='SyncBN'),
        act_cfg=dict(type='SiLU'),
        # init_cfg=dict(
        #     type='Pretrained', prefix='neck.', checkpoint=checkpoint),
        ),
    bbox_head=dict(
        type='RotatedRTMDetSepBNHead',
        num_classes=15,
        in_channels=96,
        stacked_convs=2,
        feat_channels=96,
        angle_version=angle_version,
        anchor_generator=dict(
            type='mmdet.MlvlPointGenerator', offset=0, strides=[4, 8, 16],
            ),
        bbox_coder=dict(
            type='mmrotate.DistanceAnglePointCoder', angle_version=angle_version),
        loss_cls=dict(
            type='mmdet.QualityFocalLoss',
            use_sigmoid=True,
            beta=2.0,
            loss_weight=1.0),
        loss_bbox=dict(type='mmrotate.RotatedIoULoss', mode='linear', loss_weight=2.0),
        with_objectness=False,
        exp_on_reg=False,
        share_conv=True,
        pred_kernel_size=1,
        use_hbbox_loss=False,
        scale_angle=False,
        loss_angle=None,
        norm_cfg=dict(type='SyncBN'),
        act_cfg=dict(type='SiLU'),
        # init_cfg=dict(
        #     type='Pretrained', prefix='bbox_head.', checkpoint=checkpoint)
        ),
    # 插入det网络中的轻量级的feat上采样超分模块
    det_up_sample_blocks=dict(
        # type='upSBs_v3', 
        type='upSBs', 
        # use_cspxt_blks=True,
        use_cspxt_blks=False,
        use_rfa=False, # 启用 rfa 密集连接架构
        insert_position="after_backbone", # 上采样模块插入的点位，是在backbone后还是neck后（原本sr3架构是插入在neck后的结构）
        feat_chn=[96, 192, 384], # list 形式用于在backbone后插入
        use_pixelshuffle=True,
        use_res_represent=True, # 新增使用残差表示学习的切换
    ),
    sr_branch=dict(
        use_encoder=True, # 判断 是否使用 单独训练的 sr encoder 模块
        align_encoder_feat=True,
        use_fg_mask=False,
        use_cls_mask=False,
        mask_as_weight=True, # 将 cls mask 做为 loss weight
        pre_encoder=dict(
            type='CSPNetXt_stem',
            begin_chn=64,
            widen_factor=0.375,
        ),
        sr_arch=dict(
            type='EDSRnet',
            in_channels=24,
            upscale_factor=4,
            num_blocks=4,
        ),
        sr_loss=dict(
            type='mmdet.MSELoss',
            loss_weight=1.0,
        ),  
        align_feat=True,
        use_affinity=True, # 是否对 neck 出来的 feat 做下处理，再对齐特征
        align_with_resrepfeat=True,
        align_one_feat=True,
        align_loss=dict(
            type='FALoss_v2',
            loss_weight=10,
            use_subsample=False,
            use_l2=True,)
    ),

    train_cfg=dict(
        det=dict(
            assigner=dict(
                    type='mmdet.DynamicSoftLabelAssigner',
                    iou_calculator=dict(type='mmrotate.RBboxOverlaps2D'),
                    topk=13),
            allowed_border=-1,
            pos_weight=-1,
            debug=False),
        # sr 分支的配置参数
        sr=dict(use_align_up_blocks=True,),
        ),
    test_cfg=dict(
        nms_pre=2000,
        min_bbox_size=0,
        score_thr=0.05,
        nms=dict(type='nms_rotated', iou_threshold=0.1),
        max_per_img=2000,
# train 时需要注释掉
        test_bbox_up2x=True, # 要是在训练时没注释掉，在训练时候val上的测试得到的效果就会很差。。。一直为0
        ),
)

# tiny 尺寸模型能承载的 bs
train_dataloader = dict(batch_size=16, num_workers=16)

# 自己增加的log保存路径
work_dir="./work_dirs/aisodet-ts/{{fileBasenameNoExtension}}"

# 开启amp得显式的设置下面的参数
base_lr = 0.004 / 8 # 对应于bs=16
optim_wrapper = dict(
    type='AmpOptimWrapper',
    optimizer=dict(type='AdamW', lr=base_lr, weight_decay=0.05),
    paramwise_cfg=dict(
        norm_decay_mult=0, bias_decay_mult=0, bypass_duplicate=True))

# 指定提交dota test服务器zip文件的保存路径
test_evaluator = dict(
    outfile_prefix='./work_dirs/dotav1_Task1/{{fileBasenameNoExtension}}')

# train
# python tools/train.py projects/esrtmdet/configs/dota/esrtmdet_tiny_3x_dota_lr_amp.py

# test
# python tools/test.py projects/esrtmdet/configs/dota/esrtmdet_tiny_3x_dota_lr_amp.py work_dirs/aisodet-ts/DOTA/esrtmdet_tiny_3x_dota.pth



