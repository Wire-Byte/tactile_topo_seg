_base_ = ['./bevanet_s_yellowblock_revise_v2_pretrain.py']

experiment_name = 'bevanet_s_yellowblock_revise_v2_pretrain_smoke500'

train = dict(
    max_iters=500,
    warmup_iters=100,
    val_interval=500,
    log_interval=20,
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
