_base_ = ['./segformer_b2_tp_skel_clDice_v2_4090_bs4.py']

# Skeleton head attach ablation: F1 (stage-1 feature)
work_dir = 'work_dirs/segformer_b2_tp_skel_cldice_v2_4090_bs4_F1'

model = dict(
    auxiliary_head=dict(
        in_channels=64,
        in_index=0,
        channels=64,
    ),
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
        out_dir='work_dirs/segformer_b2_tp_skel_cldice_v2_4090_bs4_F1/ckpt',
    ),
    logger=dict(type='LoggerHook', interval=20),
)
