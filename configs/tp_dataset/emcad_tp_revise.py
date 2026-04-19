custom_imports = dict(
    imports=[
        'mmseg_ext.datasets',
        'mmseg_ext.transforms',
    ],
    allow_failed_imports=False,
)

experiment_name = 'emcad_tp_revise'
model_name = 'emcad'

repo_root = 'third_party/revise_models/emcad'

# Lightweight priority: PVTv2-B0 encoder variant.
model_kwargs = dict(
    num_classes=2,
    kernel_sizes=[1, 3, 5],
    expansion_factor=2,
    dw_parallel=True,
    add=True,
    lgag_ks=3,
    activation='relu6',
    encoder='pvt_v2_b0',
    pretrain=False,
    pretrained_dir='third_party/revise_models/emcad/pretrained_pth/pvt',
)

data = dict(
    data_root='data/TP-Dataset',
    train_index='Index/train.txt',
    val_index='Index/val.txt',
    img_subdir='JPEGImages',
    gt_subdir='GroundTruth',
    image_size=(512, 512),
    normalize=True,
    val_keep_original_size=True,
)

train = dict(
    max_iters=10000,
    warmup_iters=500,
    val_interval=500,
    log_interval=50,
    batch_size=6,
    num_workers=8,
    lr=3e-4,
    weight_decay=1e-4,
    seed=3407,
    amp=True,
)

optimizer = dict(type='AdamW', betas=(0.9, 0.999))
