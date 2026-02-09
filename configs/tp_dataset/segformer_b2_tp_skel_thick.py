_base_ = [
    '../../third_party/mmsegmentation/configs/segformer/segformer_mit-b2_8xb1-160k_cityscapes-1024x1024.py',
]

custom_imports = dict(
    imports=[
        'mmseg_ext.datasets',
        'mmseg_ext.transforms',
        'mmseg_ext.models',
        'mmseg_ext.models.tp_skeleton_head',
        'mmseg_ext.models.tp_encoder_decoder_thick',
    ],
    allow_failed_imports=False
)

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
    dict(type='GenerateSkeletonDistMap', clip_max=64.0, use_cv2=True),

    dict(type='PackSegInputsWithSkeleton'),
]

val_pipeline = [
    dict(type='LoadImageFromFile'),
    dict(type='LoadAnnotations'),
    dict(type='MapTPLabels', mapping={255: 1}),
    dict(type='GenerateTPSkeleton', target_label=1),
    dict(type='GenerateSkeletonDistMap', clip_max=64.0, use_cv2=True),
    dict(type='PackSegInputsWithSkeleton'),
]
test_pipeline = val_pipeline

train_dataloader = dict(
    batch_size=1,
    num_workers=4,
    persistent_workers=True,
    sampler=dict(type='InfiniteSampler', shuffle=True),
    dataset=dict(
        type=dataset_type,
        data_root=data_root,
        data_prefix=dict(img_path='JPEGImages', seg_map_path='GroundTruth'),
        ann_file='Index/train.txt',
        pipeline=train_pipeline,
    )
)

val_dataloader = dict(
    batch_size=1,
    num_workers=2,
    persistent_workers=True,
    sampler=dict(type='DefaultSampler', shuffle=False),
    dataset=dict(
        type=dataset_type,
        data_root=data_root,
        data_prefix=dict(img_path='JPEGImages', seg_map_path='GroundTruth'),
        ann_file='Index/val.txt',
        pipeline=val_pipeline,
    )
)
test_dataloader = val_dataloader

val_evaluator = dict(type='IoUMetric', iou_metrics=['mIoU', 'mDice', 'mFscore'])
test_evaluator = val_evaluator

# ===== model =====
model = dict(
    type='TPEncoderDecoderThick',
    lambda_thick=0.05,  # 先从 0.05 起步（推荐网格：0.02/0.05/0.1）
    thick_tau=8.0,      # 允许厚度（像素），推荐网格：6/8/10

    decode_head=dict(num_classes=2),

    auxiliary_head=dict(
        type='TPSkeletonHead',
        in_channels=64,
        in_index=0,
        input_transform=None,
        channels=64,
        loss_bce=dict(type='CrossEntropyLoss', use_sigmoid=True, loss_weight=1.0),
        loss_dice=dict(type='DiceLoss', use_sigmoid=True, loss_weight=1.0),
    )
)

# ===== runtime =====
work_dir = 'work_dirs/segformer_b2_tp_skel_thick'

default_hooks = dict(
    checkpoint=dict(
        type='CheckpointHook',
        by_epoch=False,
        interval=500,
        save_last=True,
        max_keep_ckpts=30,
        save_best='mIoU',
        rule='greater',
        out_dir='work_dirs/segformer_b2_tp_skel_thick/ckpt',
    ),
    logger=dict(type='LoggerHook', interval=50),
)

train_cfg = dict(type='IterBasedTrainLoop', max_iters=10000, val_interval=500)
val_cfg = dict(type='ValLoop')
test_cfg = dict(type='TestLoop')
