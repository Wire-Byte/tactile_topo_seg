custom_imports = dict(
    imports=[
        'mmseg_ext.datasets',
        'mmseg_ext.transforms',
    ],
    allow_failed_imports=False,
)

experiment_name = 'vmunet_tp_revise'
model_name = 'vmunet'

repo_root = 'third_party/revise_models/VM-UNet'

# VM-UNet official default depths; no pretrain for from-scratch training.
model_kwargs = dict(
    input_channels=3,
    num_classes=2,
    depths=[2, 2, 9, 2],
    depths_decoder=[2, 9, 2, 2],
    drop_path_rate=0.2,
    load_ckpt_path=None,
)

data = dict(
    data_root='data/TP-Dataset',
    train_index='Index/train.txt',
    val_index='Index/val.txt',
    img_subdir='JPEGImages',
    gt_subdir='GroundTruth',
    image_size=(512, 512),
    val_keep_original_size=False,
)

train = dict(
    max_iters=10000,
    val_interval=500,
    log_interval=50,
    batch_size=4,
    num_workers=8,
    lr=2e-4,
    weight_decay=0.01,
    seed=3407,
    amp=True,
)

optimizer = dict(type='AdamW', betas=(0.9, 0.999))
