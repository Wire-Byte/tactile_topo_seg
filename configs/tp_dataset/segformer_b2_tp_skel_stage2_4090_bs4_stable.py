_base_ = ['./segformer_b2_tp_skel.py']

# Stage2 skeleton-only stable training config.
# Keeps structure unchanged while stabilizing optimization for 10k iters.

work_dir = 'work_dirs/segformer_b2_tp_skel_stage2_4090_bs4_stable'

model = dict(
    auxiliary_head=dict(
        in_channels=128,
        in_index=1,
        channels=64,
    ),
)

train_dataloader = dict(
    batch_size=4,
    num_workers=8,
    persistent_workers=True,
)

val_dataloader = dict(
    batch_size=1,
    num_workers=4,
    persistent_workers=True,
)
test_dataloader = val_dataloader

# Use conservative LR and keep AMP for 4090 bs4 throughput.
optim_wrapper = dict(
    type='AmpOptimWrapper',
    loss_scale='dynamic',
    optimizer=dict(lr=8e-5),
    clip_grad=dict(max_norm=1.0, norm_type=2),
)

# Align scheduler to actual 10k training length.
param_scheduler = [
    dict(type='LinearLR', start_factor=1e-6, by_epoch=False, begin=0, end=500),
    dict(type='PolyLR', eta_min=0.0, power=1.0, by_epoch=False, begin=500, end=10000),
]

default_hooks = dict(
    checkpoint=dict(
        type='CheckpointHook',
        by_epoch=False,
        interval=500,
        save_last=True,
        max_keep_ckpts=20,
        save_best='mIoU',
        rule='greater',
        out_dir='work_dirs/segformer_b2_tp_skel_stage2_4090_bs4_stable/ckpt',
    ),
    logger=dict(type='LoggerHook', interval=20),
)

train_cfg = dict(type='IterBasedTrainLoop', max_iters=10000, val_interval=500)
val_cfg = dict(type='ValLoop')
test_cfg = dict(type='TestLoop')
