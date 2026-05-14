_base_ = ['../tp_dataset/segformer_b2_tp_skel_stage2_4090_bs4_stable.py']

work_dir = 'work_dirs/yellowblock/mmseg/segformer_b2_skeleton_yellowblock_stage2_4090_bs4_stable'

train_dataloader = dict(
    dataset=dict(
        data_root='data/YellowBlock-TP',
        ann_file='Index/train.txt',
    )
)

val_pipeline = [
    dict(type='LoadImageFromFile'),
    dict(type='LoadAnnotations'),
    dict(type='MapTPLabels', mapping={255: 1}),
    dict(type='GenerateTPSkeleton', target_label=1),
    dict(type='PackSegInputsWithSkeleton'),
]

val_dataloader = dict(
    dataset=dict(
        data_root='data/YellowBlock-TP',
        ann_file='Index/val.txt',
        pipeline=val_pipeline,
    )
)

test_pipeline = val_pipeline
test_dataloader = dict(
    dataset=dict(
        data_root='data/YellowBlock-TP',
        ann_file='Index/test.txt',
        pipeline=test_pipeline,
    )
)

default_hooks = dict(
    checkpoint=dict(
        out_dir='work_dirs/yellowblock/mmseg/segformer_b2_skeleton_yellowblock_stage2_4090_bs4_stable/ckpt',
    )
)

train_cfg = dict(type='IterBasedTrainLoop', max_iters=10000, val_interval=500)
randomness = dict(seed=3407)
