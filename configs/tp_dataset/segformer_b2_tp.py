_base_ = [
    '../../third_party/mmsegmentation/configs/segformer/segformer_mit-b2_8xb1-160k_cityscapes-1024x1024.py',
]

# import your custom dataset + custom transforms
custom_imports = dict(
    imports=['mmseg_ext.datasets', 'mmseg_ext.transforms'],
    allow_failed_imports=False
)

# ==== dataset override (TP-Dataset) ====
dataset_type = 'TPDataset'
data_root = 'data/TP-Dataset'

# -----------------------
# pipelines
# -----------------------
# 训练时可以继续用 resize/crop（稳一点：沿用 cityscapes 风格）
train_pipeline = [
    dict(type='LoadImageFromFile'),
    dict(type='LoadAnnotations', reduce_zero_label=False),
    dict(type='MapTPLabels', mapping={255: 1}),  # ✅ TP: 255 -> 1
    # 下面这些是典型训练增强（你也可以按需删减）
    dict(type='RandomResize', scale=(2048, 1024), ratio_range=(0.5, 2.0), keep_ratio=True),
    dict(type='RandomCrop', crop_size=(1024, 1024), cat_max_ratio=0.75),
    dict(type='RandomFlip', prob=0.5),
    dict(type='PhotoMetricDistortion'),
    dict(type='PackSegInputs'),
]

# ✅ 关键：验证/测试不要 resize（避免 pred 回原图、gt 却是 resize 后的尺寸）
val_pipeline = [
    dict(type='LoadImageFromFile'),
    dict(type='LoadAnnotations', reduce_zero_label=False),
    dict(type='MapTPLabels', mapping={255: 1}),
    dict(type='PackSegInputs'),
]

# -----------------------
# dataloaders
# -----------------------
train_dataloader = dict(
    batch_size=2,
    num_workers=4,
    persistent_workers=True,
    sampler=dict(type='InfiniteSampler', shuffle=True),
    dataset=dict(
        type=dataset_type,
        data_root=data_root,
        data_prefix=dict(img_path='JPEGImages', seg_map_path='GroundTruth'),
        ann_file='Index/train.txt',
        pipeline=train_pipeline,   # ✅ 加上
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
        pipeline=val_pipeline,     # ✅ 加上
    )
)
test_dataloader = val_dataloader

# evaluators
val_evaluator = dict(type='IoUMetric', iou_metrics=['mIoU', 'mDice', 'mFscore'])
test_evaluator = val_evaluator

# ==== model override ====
model = dict(decode_head=dict(num_classes=2))

# ==== runtime ====
work_dir = 'work_dirs/segformer_b2_tp'
