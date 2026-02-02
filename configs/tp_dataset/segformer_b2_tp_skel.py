_base_ = [
    '../../third_party/mmsegmentation/configs/segformer/segformer_mit-b2_8xb1-160k_cityscapes-1024x1024.py',
]

# import your custom dataset + transforms + models
custom_imports = dict(
    imports=[
        'mmseg_ext.datasets',
        'mmseg_ext.transforms',
        'mmseg_ext.models',
        'mmseg_ext.models.tp_skeleton_head',
    ],
    allow_failed_imports=False
)

# ==== dataset override (TP-Dataset) ====
dataset_type = 'TPDataset'
data_root = 'data/TP-Dataset'

# ---- pipelines ----
# NOTE: 先确保你的 MapTPLabels 已经注册并能把 255->1（你之前就是这么做的）
train_pipeline = [
    dict(type='LoadImageFromFile'),
    dict(type='LoadAnnotations'),
    dict(type='MapTPLabels', mapping={255: 1}),

    # ✅ 先做几何变换，统一到 crop_size
    dict(type='RandomResize', scale=(1024, 1024), ratio_range=(0.5, 2.0), keep_ratio=True),
    dict(type='RandomCrop', crop_size=(512, 512), cat_max_ratio=0.75),
    dict(type='RandomFlip', prob=0.5),

    dict(type='PhotoMetricDistortion'),

    # ✅ 最后基于“最终 gt_sem_seg 尺寸”生成 skeleton
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

# ==== model override ====
# main seg head: 2 classes
# add auxiliary skeleton head: 1-channel binary
model = dict(
    decode_head=dict(num_classes=2),

    # 加一个 skeleton auxiliary head（单尺度输入）
    auxiliary_head=dict(
        type='TPSkeletonHead',
        in_channels=64,          # 这里要匹配你选的那一层特征的channel
        in_index=0,              # 你想用哪一层就填哪个index（取决于 backbone 输出）
        input_transform=None,    # 明确：单层，不做多层融合
        channels=64,
        loss_bce=dict(type='CrossEntropyLoss', use_sigmoid=True, loss_weight=1.0),
        loss_dice=dict(type='DiceLoss', use_sigmoid=True, loss_weight=1.0),
    )

)


# ==== runtime ====
work_dir = 'work_dirs/segformer_b2_tp'

# 你想每500 iter评估/保存的话，把默认 hook 显式写出来最稳
default_hooks = dict(
    checkpoint=dict(type='CheckpointHook', by_epoch=False, interval=500, max_keep_ckpts=10, save_best='mIoU'),
    logger=dict(type='LoggerHook', interval=50),
)

# 每500 iter 做一次 val（如果你想保持和之前一致）
train_cfg = dict(type='IterBasedTrainLoop', max_iters=10000, val_interval=500)
val_cfg = dict(type='ValLoop')
test_cfg = dict(type='TestLoop')
