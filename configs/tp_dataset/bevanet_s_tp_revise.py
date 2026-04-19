custom_imports = dict(
    imports=[
        'mmseg_ext.datasets',
        'mmseg_ext.transforms',
    ],
    allow_failed_imports=False,
)

experiment_name = 'bevanet_s_tp_revise'
model_name = 'bevanet_s'

repo_root = 'third_party/revise_models/bevanet'

# Lightweight priority: BEVANet-S style setting.
model_kwargs = dict(
    num_classes=2,
    branches=2,
    planes=32,
    num_blocks=[2, 2, 3, 3, 2],
    semantic_kernel_size=[0, 35, 35],
    detail_kernel_size=[0, 23],
    mlp_expand=[4, 4, 4],
    ppm_planes=96,
    head_planes=128,
    pretrained='',
)

data = dict(
    data_root='data/TP-Dataset',
    train_index='Index/train.txt',
    val_index='Index/val.txt',
    img_subdir='JPEGImages',
    gt_subdir='GroundTruth',
    image_size=(512, 512),
    normalize=True,
    val_keep_original_size=False,
)

train = dict(
    max_iters=10000,
    warmup_iters=500,
    val_interval=500,
    log_interval=50,
    batch_size=8,
    num_workers=8,
    lr=5e-4,
    weight_decay=1e-4,
    seed=3407,
    amp=True,
)

optimizer = dict(type='AdamW', betas=(0.9, 0.999))
