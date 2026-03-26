_base_ = ['./segformer_b2_tp_skel_clDice.py']

# V2 goal:
# 1) Reduce clDice strength to avoid over-constraining area segmentation.
# 2) Reduce auxiliary skeleton branch dominance.
# 3) Use a deeper feature level for skeleton aux head to suppress texture noise.

work_dir = 'work_dirs/segformer_b2_tp_skel_cldice_v2'

model = dict(
    decode_head=dict(
        # CE(1.0) + Dice(0.5) + clDice(0.1)
        loss_decode=[
            dict(type='CrossEntropyLoss', use_sigmoid=False, loss_weight=1.0),
            dict(type='DiceLoss', use_sigmoid=False, loss_weight=0.5),
            dict(type='SoftCLDiceLoss', foreground_index=1, iterations=10, loss_weight=0.1),
        ],
    ),
    auxiliary_head=dict(
        # Move skeleton aux supervision to a deeper stage feature.
        in_channels=128,
        in_index=1,
        channels=64,
        # Reduce aux branch total weight.
        loss_bce=dict(type='CrossEntropyLoss', use_sigmoid=True, loss_weight=0.5),
        loss_dice=dict(type='DiceLoss', use_sigmoid=True, loss_weight=0.5),
        loss_cldice=dict(type='SoftCLDiceLoss', foreground_index=0, iterations=10, loss_weight=0.1),
    ),
)

# Keep same 10k training schedule and hooks style, but isolate ckpt path for V2.
default_hooks = dict(
    checkpoint=dict(
        type='CheckpointHook',
        by_epoch=False,
        interval=500,
        save_last=True,
        max_keep_ckpts=20,
        save_best='mIoU',
        rule='greater',
        out_dir='work_dirs/segformer_b2_tp_skel_cldice_v2/ckpt',
    ),
    logger=dict(type='LoggerHook', interval=20),
)
