experiment_name = 'vmunet_yellowblock_revise_v2_pretrain'
model_name = 'vmunet'

repo_root = 'third_party/revise_models/VM-UNet'

model_kwargs = dict(
    input_channels=3,
    num_classes=2,
    depths=[2, 2, 9, 2],
    depths_decoder=[2, 9, 2, 2],
    drop_path_rate=0.2,
    load_ckpt_path='third_party/revise_models/VM-UNet/pre_trained_weights/vmamba_small_e238_ema.pth',
)

data = dict(
    data_root='data/YellowBlock-TP',
    train_index='Index/train.txt',
    val_index='Index/val.txt',
    topology_split='Index/val.txt',
    img_subdir='JPEGImages',
    gt_subdir='GroundTruth',
    topology_gt_subdir='GroundTruth',
    image_size=(512, 512),
    normalize=True,
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
    ema_momentum=0.0,
    eval_with_ema=False,
    log_subdir='yellowblock/revise',
    work_subdir='yellowblock/revise',
)

optimizer = dict(type='AdamW', betas=(0.9, 0.999))
