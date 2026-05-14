_base_ = ['../tp_dataset/segformer_b2_tp.py']

data_root = 'data/YellowBlock-TP'
work_dir = 'work_dirs/TP2Yellow/segformer_b2_tp8500_yellowblock'

val_pipeline = [
    dict(type='LoadImageFromFile'),
    dict(type='LoadAnnotations'),
    dict(type='MapTPLabels', mapping={255: 1}),
    dict(type='PackSegInputs'),
]
test_pipeline = val_pipeline

train_dataloader = dict(
    dataset=dict(
        data_root=data_root,
        ann_file='Index/train.txt',
        data_prefix=dict(img_path='JPEGImages', seg_map_path='GroundTruth'),
    ),
)

val_dataloader = dict(
    dataset=dict(
        data_root=data_root,
        ann_file='Index/val.txt',
        data_prefix=dict(img_path='JPEGImages', seg_map_path='GroundTruth'),
        pipeline=val_pipeline,
    ),
)

test_dataloader = dict(
    dataset=dict(
        data_root=data_root,
        ann_file='Index/test.txt',
        data_prefix=dict(img_path='JPEGImages', seg_map_path='GroundTruth'),
        pipeline=test_pipeline,
    ),
)

default_hooks = dict(
    checkpoint=dict(out_dir='work_dirs/TP2Yellow/segformer_b2_tp8500_yellowblock/ckpt'),
    logger=dict(type='LoggerHook', interval=20),
)
