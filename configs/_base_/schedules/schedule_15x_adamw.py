"""
主要用于dotamini4 上的饱和训练（充分训练完毕），使得得到的精度稳定下来
"""

max_epochs = 15 * 12
base_lr = 0.004 / 16 # (0.00025) 对应于bs=8
interval = 12
# interval = 1

train_cfg = dict(
    type='EpochBasedTrainLoop', max_epochs=max_epochs, val_interval=interval)
val_cfg = dict(type='ValLoop')
test_cfg = dict(type='TestLoop')

# learning rate
param_scheduler = [
    dict(
        type='LinearLR',
        start_factor=1.0e-5,
        by_epoch=False,
        begin=0,
        end=1000), # warmup 结束的iters
    dict(
        type='CosineAnnealingLR', # 这段学习率的配置有点意思
        eta_min=base_lr * 0.05,
        begin=max_epochs // 2,
        end=max_epochs,
        T_max=max_epochs // 2,
        by_epoch=True,
        convert_to_iter_based=True),
]

# optimizer
optim_wrapper = dict(
    type='OptimWrapper',
    optimizer=dict(type='AdamW', lr=base_lr, weight_decay=0.05),
    paramwise_cfg=dict(
        norm_decay_mult=0, bias_decay_mult=0, bypass_duplicate=True))