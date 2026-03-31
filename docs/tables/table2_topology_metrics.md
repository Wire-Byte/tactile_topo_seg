# Table 2 Topology Comparison

## Setup
- Evaluation split: `data/TP-Dataset/Index/val.txt`
- Ground-truth root: `data/TP-Dataset/GroundTruth`
- Prediction export: `tools/export_preds_unique_by_index.py`
- Topology evaluation: `tools/eval_tp_topology.py`

## Results: Overlap And Connectivity Scores

| Model | Dice | IoU | clDice |
|---|---:|---:|---:|
| SegFormer-B2 (baseline) | 0.8478 | 0.7648 | 0.7103 |
| SegFormer-B2 + Skeleton | 0.8518 | 0.7668 | 0.7489 |
| SegFormer-B2 + Skeleton + clDice-v2 | 0.8821 | 0.8139 | 0.8545 |
| DeepLabV3+ (R50) | 0.8458 | 0.7718 | 0.7905 |
| PSPNet (R50) | 0.8421 | 0.7672 | 0.7986 |
| UPerNet (R50) | 0.8022 | 0.7080 | 0.7038 |

## Results: Structural Statistics

| Model | CC Pred | CC GT | LCR Pred | LCR GT | Holes Pred | Holes GT | FG Pred | FG GT |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| SegFormer-B2 (baseline) | 80.3772 | 1.4164 | 0.8792 | 0.9720 | 69.4982 | 0.0249 | 1122435.3274 | 923852.6085 |
| SegFormer-B2 + Skeleton | 39.9359 | 1.4164 | 0.8794 | 0.9720 | 28.2847 | 0.0249 | 1032539.7794 | 923852.6085 |
| SegFormer-B2 + Skeleton + clDice-v2 | 43.1388 | 1.4164 | 0.9135 | 0.9720 | 143.2278 | 0.0249 | 926624.5872 | 923852.6085 |
| DeepLabV3+ (R50) | 46.0320 | 1.4164 | 0.8965 | 0.9720 | 32.1922 | 0.0249 | 812956.5374 | 923852.6085 |
| PSPNet (R50) | 17.9680 | 1.4164 | 0.9046 | 0.9720 | 7.4733 | 0.0249 | 781145.5338 | 923852.6085 |
| UPerNet (R50) | 31.4875 | 1.4164 | 0.8532 | 0.9720 | 8.3843 | 0.0249 | 840545.5694 | 923852.6085 |

## Metric Meanings
- `Dice`: prediction mask and GT mask overlap score. Higher is better.
- `IoU`: intersection-over-union between prediction mask and GT mask. Higher is better.
- `clDice`: connectivity-aware overlap score computed from prediction and GT skeletons. Higher is better for thin connected structures like tactile paving.
- `CC Pred`: average number of connected components in predicted masks. Lower usually means less fragmentation.
- `CC GT`: average number of connected components in GT masks. This is the structural reference of the dataset and should be identical across models.
- `LCR Pred`: average largest-component ratio in predicted masks. Higher usually means the foreground is concentrated in one dominant connected structure.
- `LCR GT`: average largest-component ratio in GT masks. This is the structural reference of the dataset and should be identical across models.
- `Holes Pred`: average number of holes inside predicted foreground regions. Lower usually means cleaner masks.
- `Holes GT`: average number of holes inside GT masks. This is the structural reference of the dataset and should be identical across models.
- `FG Pred`: average foreground-pixel count in predicted masks. It reflects the predicted object scale or area.
- `FG GT`: average foreground-pixel count in GT masks. It is the area reference of the dataset and should be identical across models.

## Model Checkpoints
- `SegFormer-B2 (baseline)`
  config: `configs/tp_dataset/segformer_b2_tp.py`
  checkpoint: `work_dirs/segformer_b2_tp/iter_8500.pth`
- `SegFormer-B2 + Skeleton`
  config: `configs/tp_dataset/segformer_b2_tp_skel.py`
  checkpoint: `work_dirs/segformer_b2_tp_skel/ckpt/segformer_b2_tp_skel/best_mIoU_iter_8500.pth`
- `SegFormer-B2 + Skeleton + clDice-v2`
  config: `configs/tp_dataset/segformer_b2_tp_skel_clDice_v2_4090_bs4.py`
  checkpoint: `work_dirs/segformer_b2_tp_skel_cldice_v2_4090_bs4/ckpt/segformer_b2_tp_skel_cldice_v2_4090_bs4/best_mIoU_iter_8000.pth`
- `DeepLabV3+ (R50)`
  config: `configs/tp_dataset/deeplabv3plus_r50_tp_main_table1_B.py`
  checkpoint: `work_dirs/deeplabv3plus_r50_tp_main_table1_B/ckpt/deeplabv3plus_r50_tp_main_table1_B/best_mIoU_iter_9000.pth`
- `PSPNet (R50)`
  config: `configs/tp_dataset/pspnet_r50_tp_main_table1_B.py`
  checkpoint: `work_dirs/pspnet_r50_tp_main_table1_B/ckpt/pspnet_r50_tp_main_table1_B/best_mIoU_iter_9000.pth`
- `UPerNet (R50)`
  config: `configs/tp_dataset/upernet_r50_tp_main_table1_B.py`
  checkpoint: `work_dirs/upernet_r50_tp_main_table1_B/ckpt/upernet_r50_tp_main_table1_B/best_mIoU_iter_6000.pth`

## Artifact Paths
- Consolidated CSV: `docs/tables/table2_topology_metrics.csv`
- Detailed markdown: `docs/tables/table2_topology_metrics.md`
- Per-sample CSVs: `work_dirs/table2_topology/csv`
- Prediction masks: `work_dirs/table2_topology/preds`
- Logs: `logs/table2_topology`
