custom_imports = dict(
    imports=[
        'mmseg_ext.datasets',
        'mmseg_ext.transforms',
    ],
    allow_failed_imports=False,
)

experiment_name = 'vmunet_tp_revise_v2_pretrain'
model_name = 'vmunet'

repo_root = 'third_party/revise_models/VM-UNet'

# VM-UNet official default depths + official VMamba pretrained init.
model_kwargs = dict(
    input_channels=3,
    num_classes=2,
    depths=[2, 2, 9, 2],
    depths_decoder=[2, 9, 2, 2],
    drop_path_rate=0.2,
    load_ckpt_path='third_party/revise_models/VM-UNet/pre_trained_weights/vmamba_small_e238_ema.pth',
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
    warmup_iters=1000,
    val_interval=500,
    log_interval=50,
    batch_size=4,
    num_workers=8,
    lr=1e-4,
    weight_decay=0.01,
    seed=3407,
    amp=True,
    ema_momentum=0.999,
    eval_with_ema=True,
)

optimizer = dict(type='AdamW', betas=(0.9, 0.999))
