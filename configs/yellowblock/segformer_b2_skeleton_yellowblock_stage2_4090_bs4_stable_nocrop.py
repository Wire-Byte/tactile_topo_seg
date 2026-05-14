_base_ = ['./segformer_b2_skeleton_yellowblock_stage2_4090_bs4_stable.py']

# Stabilized smoke pipeline for queue reliability:
# keep online skeleton generation but remove random crop/resize jitter.
train_pipeline = [
    dict(type='LoadImageFromFile'),
    dict(type='LoadAnnotations'),
    dict(type='MapTPLabels', mapping={255: 1}),
    dict(type='Resize', scale=(1024, 1024), keep_ratio=True),
    dict(type='RandomFlip', prob=0.5),
    dict(type='PhotoMetricDistortion'),
    dict(type='GenerateTPSkeleton', target_label=1),
    dict(type='PackSegInputsWithSkeleton'),
]

train_dataloader = dict(
    dataset=dict(
        pipeline=train_pipeline,
    )
)
