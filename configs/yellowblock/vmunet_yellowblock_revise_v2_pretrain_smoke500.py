_base_ = ['./vmunet_yellowblock_revise_v2_pretrain.py']

experiment_name = 'vmunet_yellowblock_revise_v2_pretrain_smoke500'

train = dict(
    max_iters=500,
    warmup_iters=100,
    val_interval=500,
    log_interval=20,
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
