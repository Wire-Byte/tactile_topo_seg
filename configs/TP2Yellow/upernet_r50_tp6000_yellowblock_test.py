_base_ = ['../tp_dataset/upernet_r50_tp_main_table1_B.py']

data_root = 'data/YellowBlock-TP'
work_dir = 'work_dirs/TP2Yellow/upernet_r50_tp6000_yellowblock'

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
    ),
)

test_dataloader = dict(
    dataset=dict(
        data_root=data_root,
        ann_file='Index/test.txt',
        data_prefix=dict(img_path='JPEGImages', seg_map_path='GroundTruth'),
    ),
)

default_hooks = dict(
    checkpoint=dict(out_dir='work_dirs/TP2Yellow/upernet_r50_tp6000_yellowblock/ckpt'),
    logger=dict(type='LoggerHook', interval=20),
)
