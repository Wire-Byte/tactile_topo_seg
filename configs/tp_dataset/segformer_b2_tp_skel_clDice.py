_base_ = [
    '../../third_party/mmsegmentation/configs/segformer/segformer_mit-b2_8xb1-160k_cityscapes-1024x1024.py',
]

custom_imports = dict(
    imports=[
        'mmseg_ext.datasets',
        'mmseg_ext.transforms',
        'mmseg_ext.models',
        'mmseg_ext.models.tp_cldice_loss',
        'mmseg_ext.models.tp_skeleton_head_clDice',
    ],
    allow_failed_imports=False,
)

# ==== dataset override (TP-Dataset) ====
dataset_type = 'TPDataset'
data_root = 'data/TP-Dataset'

train_pipeline = [
    dict(type='LoadImageFromFile'),
    dict(type='LoadAnnotations'),
    dict(type='MapTPLabels', mapping={255: 1}),
    dict(type='RandomResize', scale=(1024, 1024), ratio_range=(0.5, 2.0), keep_ratio=True),
    dict(type='RandomCrop', crop_size=(512, 512), cat_max_ratio=0.75),
    dict(type='RandomFlip', prob=0.5),
    dict(type='PhotoMetricDistortion'),
    dict(type='GenerateTPSkeleton', target_label=1),
    dict(type='PackSegInputsWithSkeleton'),
]

val_pipeline = [
    dict(type='LoadImageFromFile'),
    dict(type='LoadAnnotations'),
    dict(type='MapTPLabels', mapping={255: 1}),
    dict(type='GenerateTPSkeleton', target_label=1),
    dict(type='PackSegInputsWithSkeleton'),
]

test_pipeline = val_pipeline

train_dataloader = dict(
    batch_size=1,
    num_workers=6,
    persistent_workers=True,
    sampler=dict(type='InfiniteSampler', shuffle=True),
    dataset=dict(
        type=dataset_type,
        data_root=data_root,
        data_prefix=dict(img_path='JPEGImages', seg_map_path='GroundTruth'),
        ann_file='Index/train.txt',
        pipeline=train_pipeline,
    ),
)

val_dataloader = dict(
    batch_size=1,
    num_workers=4,
    persistent_workers=True,
    sampler=dict(type='DefaultSampler', shuffle=False),
    dataset=dict(
        type=dataset_type,
        data_root=data_root,
        data_prefix=dict(img_path='JPEGImages', seg_map_path='GroundTruth'),
        ann_file='Index/val.txt',
        pipeline=val_pipeline,
    ),
)

test_dataloader = val_dataloader

val_evaluator = dict(type='IoUMetric', iou_metrics=['mIoU', 'mDice', 'mFscore'])
test_evaluator = val_evaluator

# ==== model override ====
# Main head: CE + Dice + clDice
# Aux skeleton head: BCE + Dice + clDice
model = dict(
    decode_head=dict(
        num_classes=2,
        loss_decode=[
            dict(type='CrossEntropyLoss', use_sigmoid=False, loss_weight=1.0),
            dict(type='DiceLoss', use_sigmoid=False, loss_weight=0.5),
            dict(type='SoftCLDiceLoss', foreground_index=1, iterations=10, loss_weight=0.3),
        ],
    ),
    auxiliary_head=dict(
        type='TPSkeletonHeadClDice',
        in_channels=64,
        in_index=0,
        input_transform=None,
        channels=64,
        loss_bce=dict(type='CrossEntropyLoss', use_sigmoid=True, loss_weight=1.0),
        loss_dice=dict(type='DiceLoss', use_sigmoid=True, loss_weight=1.0),
        loss_cldice=dict(type='SoftCLDiceLoss', foreground_index=0, iterations=10, loss_weight=0.3),
    ),
)

work_dir = 'work_dirs/segformer_b2_tp_skel_cldice'

# Keep 10k-iter schedule aligned with actual training length.
param_scheduler = [
    dict(type='LinearLR', start_factor=1e-6, by_epoch=False, begin=0, end=500),
    dict(type='PolyLR', eta_min=0.0, power=1.0, by_epoch=False, begin=500, end=10000),
]

# Add gentle gradient clipping for stability when clDice is enabled.
optim_wrapper = dict(
    clip_grad=dict(max_norm=1.0, norm_type=2),
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
        out_dir='work_dirs/segformer_b2_tp_skel_cldice/ckpt',
    ),
    logger=dict(type='LoggerHook', interval=20),
)

train_cfg = dict(type='IterBasedTrainLoop', max_iters=10000, val_interval=500)
val_cfg = dict(type='ValLoop')
test_cfg = dict(type='TestLoop')
