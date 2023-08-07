default_scope = 'aisodet'

default_hooks = dict(
    timer=dict(type='IterTimerHook'),
    logger=dict(type='LoggerHook', interval=50),
    param_scheduler=dict(type='ParamSchedulerHook'),
    checkpoint=dict(type='CheckpointHook', interval=12, max_keep_ckpts=3), # 新增了个max_keep_ckpt参数？？
    sampler_seed=dict(type='DistSamplerSeedHook'),
    visualization=dict(type='mmdet.DetVisualizationHook'))

env_cfg = dict(
    cudnn_benchmark=False,
    mp_cfg=dict(mp_start_method='fork', opencv_num_threads=0),
    dist_cfg=dict(backend='nccl'),
)

vis_backends = [dict(type='LocalVisBackend')]
visualizer = dict(
    type='mmrotate.RotLocalVisualizer', vis_backends=vis_backends, name='visualizer')
log_processor = dict(type='LogProcessor', window_size=50, by_epoch=True)

log_level = 'INFO'
load_from = None
resume = False

# 启用 ema hook
custom_hooks = [
    dict(type='mmdet.NumClassCheckHook'), # 分类数目检查
    dict(
        type='EMAHook',
        ema_type='mmdet.ExpMomentumEMA', # 使用指数动量emd
        momentum=0.0002,
        update_buffers=True,
        priority=49)
]