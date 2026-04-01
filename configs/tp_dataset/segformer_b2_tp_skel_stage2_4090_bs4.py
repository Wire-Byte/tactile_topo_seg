_base_ = ['./segformer_b2_tp_skel.py']

# Skeleton-only rerun:
# - keep the plain TPSkeletonHead design (no clDice)
# - move the auxiliary head to stage-2 features, following the v2 placement
# - use the 4090 bs4 training setup for better throughput

work_dir = 'work_dirs/segformer_b2_tp_skel_stage2_4090_bs4'

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

optim_wrapper = dict(
    type='AmpOptimWrapper',
    loss_scale='dynamic',
    optimizer=dict(lr=1.2e-4),
)

default_hooks = dict(
    checkpoint=dict(
        type='CheckpointHook',
        by_epoch=False,
        interval=500,
        save_last=True,
        max_keep_ckpts=20,
        save_best='mIoU',
        rule='greater',
        out_dir='work_dirs/segformer_b2_tp_skel_stage2_4090_bs4/ckpt',
    ),
    logger=dict(type='LoggerHook', interval=20),
)
