experiment_name = 'emcad_yellowblock_revise_v2_pretrain'
model_name = 'emcad'

repo_root = 'third_party/revise_models/emcad'

model_kwargs = dict(
    num_classes=2,
    kernel_sizes=[1, 3, 5],
    expansion_factor=2,
    dw_parallel=True,
    add=True,
    lgag_ks=3,
    activation='relu6',
    encoder='pvt_v2_b2',
    pretrain=True,
    pretrained_dir='third_party/revise_models/emcad/pretrained_pth/pvt',
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
    batch_size=6,
    num_workers=8,
    lr=2e-4,
    weight_decay=1e-4,
    seed=3407,
    amp=True,
    ema_momentum=0.0,
    eval_with_ema=False,
    log_subdir='yellowblock/revise',
    work_subdir='yellowblock/revise',
)

optimizer = dict(type='AdamW', betas=(0.9, 0.999))
