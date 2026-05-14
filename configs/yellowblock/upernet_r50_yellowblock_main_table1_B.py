_base_ = ['../tp_dataset/upernet_r50_tp_main_table1_B.py']

work_dir = 'work_dirs/yellowblock/mmseg/upernet_r50_yellowblock_main_table1_B'

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
    dict(type='PackSegInputs'),
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
        out_dir='work_dirs/yellowblock/mmseg/upernet_r50_yellowblock_main_table1_B/ckpt',
    )
)

train_cfg = dict(type='IterBasedTrainLoop', max_iters=10000, val_interval=500)
randomness = dict(seed=3407)
