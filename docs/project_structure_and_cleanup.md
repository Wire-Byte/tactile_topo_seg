# tactile_topo_seg 工程梳理与清理记录（2026-03-25）

## 1. 项目目标与主线
- 任务：盲道识别（二分类语义分割：background / tactile_paving）。
- baseline：SegFormer-B2。
- 创新点：`mmseg_ext/models/tp_skeleton_head.py`（骨架辅助监督 head）。

## 2. 目录说明（清理后）

### 根目录
- `requirements.txt`
  - Python 依赖列表。
- `configs/`
  - 训练/评估配置。
- `data/`
  - 数据集目录。
- `docs/`
  - 文档与表格。
- `experiments/`
  - 实验记录目录（当前为空或仅保留框架）。
- `logs/`
  - 训练日志目录（可继续使用）。
- `mmseg_ext/`
  - 自定义数据集、transform、模型扩展（当前只保留主线需要）。
- `third_party/mmsegmentation/`
  - 上游 MMSegmentation 代码。
- `tools/`
  - 项目工具脚本（当前保留数据集 sanity 检查）。
- `work_dirs/`
  - 训练输出目录（当前保留 baseline/skel 两条主线）。

### configs
- `configs/tp_dataset/tp_dataset.py`
  - TP 数据集 dataloader 基础定义。
- `configs/tp_dataset/segformer_b2_tp.py`
  - 纯 baseline 配置（SegFormer-B2，不含 skeleton head）。
- `configs/tp_dataset/segformer_b2_tp_skel.py`
  - skeleton 创新版配置（含 `TPSkeletonHead` + skeleton target 生成与打包）。

### mmseg_ext
- `mmseg_ext/datasets/tp_dataset.py`
  - 自定义 `TPDataset`，定义类别与调色板，按 `Index/*.txt` 读取样本。
- `mmseg_ext/transforms/map_tp_labels.py`
  - 在线标签映射（如 255 -> 1）。
- `mmseg_ext/transforms/generate_skeleton.py`
  - 根据语义掩码生成骨架监督图 `gt_skeleton`。
- `mmseg_ext/transforms/pack_seg_inputs_ext.py`
  - 扩展打包，支持把 skeleton 等监督图写入 `data_samples`。
- `mmseg_ext/models/tp_skeleton_head.py`
  - 创新模块：骨架辅助 head（BCE + Dice）。

### tools
- `tools/check_dataset_sanity.py`
  - 用真实 pipeline 抽样检查 `gt_sem_seg` / `gt_skeleton` 是否正确。

### work_dirs
- `work_dirs/segformer_b2_tp/`
  - baseline 训练产物。
- `work_dirs/segformer_b2_tp_skel/`
  - skeleton 训练产物。

## 3. 本次已清理内容

### 代码层（已删除）
- `mmseg_ext/models/decode_heads/`（morph 分支）
- `mmseg_ext/models/losses/`（morph/thick 相关损失）
- `mmseg_ext/models/tp_encoder_decoder_thick.py`
- `mmseg_ext/transforms/generate_mask_dist_map.py`
- `mmseg_ext/transforms/generate_morph_maps.py`
- `mmseg_ext/transforms/generate_skel_dist_map.py`
- `tools_ext/`（拓扑对比/可视化测试脚本）

### 产物层（已删除）
- 所有 `__pycache__/`、`.ipynb_checkpoints/`
- `work_dirs/` 下测试/调试分支（`*morph*`、`*thick*`、`_debug*`、`ZZZ_*`、`vis_*`、`topk_*` 等）

## 4. 当前建议的日常使用入口
- baseline 训练：`configs/tp_dataset/segformer_b2_tp.py`
- skeleton 训练：`configs/tp_dataset/segformer_b2_tp_skel.py`
- 数据检查：`tools/check_dataset_sanity.py`

## 5. 后续可选优化
- 在根目录补一个 `README.md`，写清楚环境、训练、验证、推理命令。
- 给 `segformer_b2_tp_skel.py` 里关键超参（`in_index`、loss 权重）做注释模板，便于后续调参。
