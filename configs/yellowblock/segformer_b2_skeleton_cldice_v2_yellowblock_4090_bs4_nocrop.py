_base_ = ['./segformer_b2_skeleton_cldice_v2_yellowblock_4090_bs4.py']

# Stabilized smoke pipeline for clDice branch:
# remove random resize/crop jitter while keeping online skeleton generation.
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
