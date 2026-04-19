# Skeleton Head Attach Ablation (F1/F2/F3/F4)

| Attach | status | best_iter | aAcc | mIoU | mDice | mFscore | mPrecision | mRecall |
|---|---|---:|---:|---:|---:|---:|---:|---:|
| F1 | ok | 9000 | 95.4800 | 83.8600 | 90.8300 | 90.8300 | 89.8700 | 91.8800 |
| F2 | missing_log | - | - | - | - | - | - | - |
| F3 | ok | 3500 | 94.2100 | 79.1700 | 87.6500 | 87.6500 | 88.6200 | 86.7500 |
| F4 | ok | 10000 | 96.0100 | 85.3200 | 91.7500 | 91.7500 | 91.4500 | 92.0600 |

## File Mapping

| Attach | config | work_dir | log |
|---|---|---|---|
| F1 | configs/tp_dataset/segformer_b2_tp_skel_clDice_v2_4090_bs4_F1.py | work_dirs/segformer_b2_tp_skel_cldice_v2_4090_bs4_F1 | logs/revise/train_segformer_b2_tp_skel_cldice_v2_4090_bs4_F1.log |
| F2 | configs/tp_dataset/segformer_b2_tp_skel_clDice_v2_4090_bs4.py | work_dirs/segformer_b2_tp_skel_cldice_v2_4090_bs4 | logs/revise/train_segformer_b2_tp_skel_cldice_v2_4090_bs4_F2.log |
| F3 | configs/tp_dataset/segformer_b2_tp_skel_clDice_v2_4090_bs4_F3.py | work_dirs/segformer_b2_tp_skel_cldice_v2_4090_bs4_F3 | logs/revise/train_segformer_b2_tp_skel_cldice_v2_4090_bs4_F3.log |
| F4 | configs/tp_dataset/segformer_b2_tp_skel_clDice_v2_4090_bs4_F4.py | work_dirs/segformer_b2_tp_skel_cldice_v2_4090_bs4_F4 | logs/revise/train_segformer_b2_tp_skel_cldice_v2_4090_bs4_F4.log |