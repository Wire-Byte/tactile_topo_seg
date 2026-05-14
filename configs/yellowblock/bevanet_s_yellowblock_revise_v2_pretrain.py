experiment_name = 'bevanet_s_yellowblock_revise_v2_pretrain'
model_name = 'bevanet_s'

repo_root = 'third_party/revise_models/bevanet'

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
    pretrained='third_party/revise_models/bevanet/pretrained_models/imagenet/BEVANet_S_ImageNet.pth',
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
    batch_size=8,
    num_workers=8,
    lr=3e-4,
    weight_decay=1e-4,
    seed=3407,
    amp=True,
    ema_momentum=0.0,
    eval_with_ema=False,
    log_subdir='yellowblock/revise',
    work_subdir='yellowblock/revise',
)

optimizer = dict(type='AdamW', betas=(0.9, 0.999))
