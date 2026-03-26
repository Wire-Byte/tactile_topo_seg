_base_ = ['./segformer_b2_tp_skel_clDice_v2.py']

# 4090 24G throughput-oriented config.
# Target: larger train batch while keeping training stable.

work_dir = 'work_dirs/segformer_b2_tp_skel_cldice_v2_4090_bs4'

# Increase batch to better utilize 24G VRAM.
train_dataloader = dict(
    batch_size=4,
    num_workers=8,
    persistent_workers=True,
)

# Keep val/test batch_size=1 because original TP val images have variable shapes.
val_dataloader = dict(
    batch_size=1,
    num_workers=4,
    persistent_workers=True,
)
test_dataloader = val_dataloader

# Conservative LR scaling for larger batch (not full linear x4, to reduce instability risk).
# Enable AMP to keep bs4 within 24G VRAM.
optim_wrapper = dict(
    type='AmpOptimWrapper',
    loss_scale='dynamic',
    optimizer=dict(lr=1.2e-4),
)

# Keep frequent logs for debugging when switching to larger batch.
default_hooks = dict(
    checkpoint=dict(
        type='CheckpointHook',
        by_epoch=False,
        interval=500,
        save_last=True,
        max_keep_ckpts=20,
        save_best='mIoU',
        rule='greater',
        out_dir='work_dirs/segformer_b2_tp_skel_cldice_v2_4090_bs4/ckpt',
    ),
    logger=dict(type='LoggerHook', interval=20),
)
