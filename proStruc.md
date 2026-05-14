# tactile_topo_seg 项目结构说明

本文档按当前工程目录整理，面向“盲道/触觉铺装语义分割研究”使用。项目主线如下：

- 基线模型：`SegFormer-B2`，配置为 `configs/tp_dataset/segformer_b2_tp.py`。
- 创新点 1：`Skeleton Head` 辅助监督，核心配置为 `configs/tp_dataset/segformer_b2_tp_skel.py`，核心实现为 `mmseg_ext/models/tp_skeleton_head.py`。
- 创新点 2：`Skeleton Head + clDice` 拓扑损失，核心配置为 `configs/tp_dataset/segformer_b2_tp_skel_clDice_v2_4090_bs4.py`，核心实现为 `mmseg_ext/models/tp_skeleton_head_clDice.py` 和 `mmseg_ext/models/tp_cldice_loss.py`。
- 对比模型：`DeepLabV3+`、`UPerNet`、`PSPNet`、`VM-UNet`、`EMCAD`、`BEVANet`。
- 实验类型：主实验、拓扑指标实验、clDice 消融、skeleton head 挂载层消融、第三方对比模型训练、YellowBlock 扩展实验。

说明：

- `work_dirs/` 主要是训练权重、预测图、TensorBoard/vis_data、临时调试产物，且你说明大部分内容在服务器上，因此本文只说明目录用途，不展开逐文件梳理。
- `third_party/mmsegmentation/` 是上游 MMSegmentation 代码，文件量很大，本文按模块说明，不逐个列上游文件。
- `__pycache__/`、`.ipynb_checkpoints/`、`.venv/`、`.git/`、`.idea/` 属于缓存、环境或 IDE 元数据，本文只标注用途，不作为研究代码展开。

## 顶层目录树

```text
tactile_topo_seg/
├── .git/                         # Git 版本库元数据，不属于实验代码
├── .gitignore                    # Git 忽略规则，忽略数据、权重、work_dirs、缓存等大文件
├── .idea/                        # JetBrains/PyCharm IDE 配置
├── .ipynb_checkpoints/           # Jupyter 自动检查点缓存
├── .venv/                        # 本地 Python 虚拟环境
├── configs/                      # 训练配置，是实验组织的核心入口
├── data/                         # 数据集目录，包含 TP-Dataset 和 YellowBlock 数据
├── docs/                         # 实验表格、环境快照、结果文档
├── experiments/                  # 实验记录占位/分组目录
├── logs/                         # 训练、消融、拓扑评估日志
├── mmseg_ext/                    # 基于 MMSegmentation 的自定义数据集、transform、head、loss
├── outputs/                      # 汇总输出，如主表、关键训练日志副本
├── requirements.txt              # 当前 Python 环境依赖冻结列表
├── third_party/                  # 第三方模型与 MMSegmentation 源码
├── tools/                        # 训练、评估、导出、消融、表格构建脚本
└── work_dirs/                    # 训练产物与预测产物，本文不展开
```

## 根目录文件

```text
.gitignore
```

- 忽略 `.venv/`、`__pycache__/`、`.ipynb_checkpoints/`、`.idea/` 等本地环境/缓存。
- 默认忽略 `data/**`，但保留 `data/**/Index/*.txt` 和 `data/**/samples/**`。
- 忽略 `work_dirs/**`、模型权重 `*.pth`/`*.pt`/`*.ckpt`、数组/序列化文件 `*.npy`/`*.pkl` 等大文件。
- 当前文件存在中文注释编码显示异常的问题，但实际意图是“大数据和训练产物不入库”。

```text
requirements.txt
```

- 当前环境完整依赖冻结列表。
- 与本项目强相关的核心依赖包括 `torch`、`torchvision`、`mmengine`、`mmcv`、`mmsegmentation`、`opencv-python`、`numpy`、`Pillow`、`matplotlib`、`timm` 等。
- 文件里也包含较多大模型/服务相关包，像 `vllm`、`transformers`、`deepspeed`、`openai` 等，不一定都是本分割项目运行的最小依赖。

## configs/

`configs/` 是实验入口。`tp_dataset/` 针对原始盲道 TP-Dataset，`yellowblock/` 针对扩展 YellowBlock 数据集。

```text
configs/
├── tp_dataset/
│   ├── segformer_b2_tp.py
│   ├── segformer_b2_tp_skel.py
│   ├── segformer_b2_tp_skel_clDice.py
│   ├── segformer_b2_tp_skel_clDice_v2.py
│   ├── segformer_b2_tp_skel_clDice_v2_4090_bs4.py
│   ├── segformer_b2_tp_skel_clDice_v2_4090_bs4_F1.py
│   ├── segformer_b2_tp_skel_clDice_v2_4090_bs4_F3.py
│   ├── segformer_b2_tp_skel_clDice_v2_4090_bs4_F4.py
│   ├── segformer_b2_tp_skel_stage2_4090_bs4.py
│   ├── segformer_b2_tp_skel_stage2_4090_bs4_stable.py
│   ├── deeplabv3plus_r50_tp_main_table1_B.py
│   ├── pspnet_r50_tp_main_table1_B.py
│   ├── upernet_r50_tp_main_table1_B.py
│   ├── bevanet_s_tp_revise.py
│   ├── bevanet_s_tp_revise_v2_pretrain.py
│   ├── emcad_tp_revise.py
│   ├── emcad_tp_revise_v2_pretrain.py
│   ├── vmunet_tp_revise.py
│   ├── vmunet_tp_revise_v2_pretrain.py
│   ├── tp_dataset.py
│   └── ablation_generated/
└── yellowblock/
    ├── segformer_b2_yellowblock.py
    ├── segformer_b2_skeleton_yellowblock_stage2_4090_bs4_stable.py
    ├── segformer_b2_skeleton_yellowblock_stage2_4090_bs4_stable_nocrop.py
    ├── segformer_b2_skeleton_cldice_v2_yellowblock_4090_bs4.py
    ├── segformer_b2_skeleton_cldice_v2_yellowblock_4090_bs4_nocrop.py
    ├── deeplabv3plus_r50_yellowblock_main_table1_B.py
    ├── pspnet_r50_yellowblock_main_table1_B.py
    ├── upernet_r50_yellowblock_main_table1_B.py
    ├── bevanet_s_yellowblock_revise_v2_pretrain.py
    ├── bevanet_s_yellowblock_revise_v2_pretrain_smoke500.py
    ├── emcad_yellowblock_revise_v2_pretrain.py
    ├── emcad_yellowblock_revise_v2_pretrain_smoke500.py
    ├── vmunet_yellowblock_revise_v2_pretrain.py
    └── vmunet_yellowblock_revise_v2_pretrain_smoke500.py
```

### configs/tp_dataset/

```text
tp_dataset.py
```

- 一个较基础的数据加载配置片段。
- 指向 `data/TP-Dataset`，使用 `Index/train.txt` 与 `Index/val.txt`。
- 可作为理解 TP 数据集路径结构的最小配置参考。

```text
segformer_b2_tp.py
```

- 主线 baseline：`SegFormer-B2`。
- 继承 MMSegmentation 官方 SegFormer MIT-B2 Cityscapes 配置。
- 自定义导入 `mmseg_ext.datasets`、`mmseg_ext.transforms`、`mmseg_ext.models`。
- 数据集类型为 `TPDataset`，类别数改为 2：`background` / `tactile_paving`。
- 训练 pipeline：读图、读标注、`MapTPLabels(255->1)`、随机缩放、随机裁剪、随机翻转、光度扰动、`PackSegInputs`。
- 验证 pipeline：读图、读标注、标签映射、resize、打包。
- 训练 10000 iter，每 500 iter 验证/保存，指标为 `mIoU`、`mDice`、`mFscore`。
- 输出目录：`work_dirs/segformer_b2_tp`。

```text
segformer_b2_tp_skel.py
```

- 创新点 1 配置：`SegFormer-B2 + Skeleton Head`。
- 在 baseline 基础上增加 `GenerateTPSkeleton` 和 `PackSegInputsWithSkeleton`。
- 模型增加 `auxiliary_head=dict(type='TPSkeletonHead')`。
- skeleton head 默认挂在 SegFormer 第 1 个特征层：`in_index=0`，`in_channels=64`。
- skeleton 辅助损失：sigmoid BCE + Dice。
- 输出目录：`work_dirs/segformer_b2_tp_skel`，checkpoint 额外放在 `work_dirs/segformer_b2_tp_skel/ckpt`。

```text
segformer_b2_tp_skel_clDice.py
```

- 创新点 2 的早期版本：`Skeleton Head + clDice`。
- 主 decode head 损失变为 CE + Dice + `SoftCLDiceLoss`。
- 辅助 skeleton head 变为 `TPSkeletonHeadClDice`，损失为 BCE + Dice + clDice。
- clDice 权重初始为 `0.3`，用于同时约束主分割和骨架分支。
- 输出目录：`work_dirs/segformer_b2_tp_skel_cldice`。

```text
segformer_b2_tp_skel_clDice_v2.py
```

- clDice 改进版 V2。
- 目标是降低 clDice 对面积分割的过约束，降低 aux 分支支配性，并将 skeleton head 移到更深特征层。
- 主 head clDice 权重从 `0.3` 降到 `0.1`。
- 辅助 head 改为 `in_index=1`、`in_channels=128`，BCE/Dice 权重降为 `0.5`，aux clDice 权重为 `0.1`。
- 输出目录：`work_dirs/segformer_b2_tp_skel_cldice_v2`。

```text
segformer_b2_tp_skel_clDice_v2_4090_bs4.py
```

- 创新点 2 的主要 4090 实验配置。
- 继承 V2，训练 batch size 提到 4，开启 `AmpOptimWrapper` 动态 loss scale。
- 学习率设为 `1.2e-4`，用于 4090 24G 显存吞吐优化。
- 验证/测试仍用 batch size 1，适配原始 TP 验证图像变尺寸。
- 输出目录：`work_dirs/segformer_b2_tp_skel_cldice_v2_4090_bs4`。

```text
segformer_b2_tp_skel_clDice_v2_4090_bs4_F1.py
segformer_b2_tp_skel_clDice_v2_4090_bs4_F3.py
segformer_b2_tp_skel_clDice_v2_4090_bs4_F4.py
```

- skeleton head 挂载层消融配置。
- 共同继承 `segformer_b2_tp_skel_clDice_v2_4090_bs4.py`。
- 用于比较 aux skeleton head 接入不同 SegFormer feature stage 的效果。
- `F2` 对应原始 `segformer_b2_tp_skel_clDice_v2_4090_bs4.py`。
- 汇总脚本为 `tools/revise/build_fpn_attach_ablation_summary.py`。

```text
segformer_b2_tp_skel_stage2_4090_bs4.py
segformer_b2_tp_skel_stage2_4090_bs4_stable.py
```

- Skeleton-only 分支的 4090 batch size 4 训练配置。
- stable 版将 skeleton head 移到更深层 `in_index=1`、`in_channels=128`，使用 AMP、较保守学习率 `8e-5`、梯度裁剪和 10k 对齐 scheduler。
- 主要用于创新点 1 的稳定复现实验。

```text
deeplabv3plus_r50_tp_main_table1_B.py
pspnet_r50_tp_main_table1_B.py
upernet_r50_tp_main_table1_B.py
```

- MMSegmentation 内置模型对比配置。
- 分别继承官方 DeepLabV3+ R50、PSPNet R50、UPerNet R50 Cityscapes 配置。
- 统一改为 TP-Dataset、2 类输出、10000 iter、500 iter 验证。
- 优化器统一使用 AdamW + AMP，学习率 `6e-5`，seed `3407`。
- UPerNet 配置使用 slide test：`crop_size=(512,512)`、`stride=(341,341)`。
- 输出目录分别为 `work_dirs/deeplabv3plus_r50_tp_main_table1_B`、`work_dirs/pspnet_r50_tp_main_table1_B`、`work_dirs/upernet_r50_tp_main_table1_B`。

```text
bevanet_s_tp_revise.py
emcad_tp_revise.py
vmunet_tp_revise.py
```

- 第三方对比模型的早期训练配置。
- 不走 MMSeg Runner，而由 `tools/revise/train_revise_external.py` 读取。
- 配置字段包括 `experiment_name`、`model_name`、`repo_root`、`model_kwargs`、`data`、`train`、`optimizer`。

```text
bevanet_s_tp_revise_v2_pretrain.py
emcad_tp_revise_v2_pretrain.py
vmunet_tp_revise_v2_pretrain.py
```

- 第三方对比模型的 V2 预训练配置。
- `BEVANet-S` 使用 ImageNet 预训练权重，batch size 8，lr `3e-4`。
- `EMCAD` 使用 PVTv2-B2 encoder 预训练，batch size 6，lr `2e-4`。
- `VM-UNet` 使用 VMamba small 预训练权重，batch size 4，lr `1e-4`。
- 都使用 `data/TP-Dataset`，统一 image size `512x512`，训练 10000 iter，500 iter 验证。

### configs/tp_dataset/ablation_generated/

```text
core_none.py
core_main_only.py
core_aux_only.py
core_both.py
w_0p05.py
w_0p10.py
w_0p20.py
w_0p30.py
```

- clDice 消融实验自动/半自动生成配置。
- 均继承 `segformer_b2_tp_skel_clDice_v2_4090_bs4.py`。
- `core_*` 用于主分支 clDice 与 aux 分支 clDice 的开关消融：
  - `core_none`：主分支和 aux 分支 clDice 都关。
  - `core_main_only`：只开主分支 clDice。
  - `core_aux_only`：只开 aux skeleton 分支 clDice。
  - `core_both`：两者都开。
- `w_*` 用于主分支和 aux 分支 clDice 同权重扫描：`0.05`、`0.10`、`0.20`、`0.30`。
- 对应汇总文档在 `docs/tables/ablation_tables_detailed.md`。

### configs/yellowblock/

YellowBlock 配置整体是把 TP-Dataset 实验迁移到 `data/YellowBlock-TP`。

```text
segformer_b2_yellowblock.py
```

- YellowBlock 上的 SegFormer-B2 baseline。
- 继承 TP baseline，只替换数据根目录、索引文件和输出目录。

```text
segformer_b2_skeleton_yellowblock_stage2_4090_bs4_stable.py
```

- YellowBlock 上的 `SegFormer-B2 + Skeleton Head` 稳定版。
- 继承 TP skeleton stable 配置，替换为 YellowBlock 数据。

```text
segformer_b2_skeleton_yellowblock_stage2_4090_bs4_stable_nocrop.py
```

- YellowBlock skeleton 版的 no-crop 稳定/冒烟配置。
- 去掉随机 crop/resize jitter，保留 resize、flip、光度扰动和在线 skeleton 生成。

```text
segformer_b2_skeleton_cldice_v2_yellowblock_4090_bs4.py
```

- YellowBlock 上的 `SegFormer-B2 + Skeleton + clDice-v2`。
- 继承 TP clDice-v2 4090 bs4 配置，替换 YellowBlock 数据和输出目录。

```text
segformer_b2_skeleton_cldice_v2_yellowblock_4090_bs4_nocrop.py
```

- clDice YellowBlock 的 no-crop 稳定/冒烟配置。
- 用于排除随机裁剪导致的小数据集训练波动。

```text
deeplabv3plus_r50_yellowblock_main_table1_B.py
pspnet_r50_yellowblock_main_table1_B.py
upernet_r50_yellowblock_main_table1_B.py
```

- YellowBlock 上的 MMSeg 对比模型配置。
- 继承对应 TP 对比配置，仅替换数据集路径、索引文件、test split 和输出目录。

```text
bevanet_s_yellowblock_revise_v2_pretrain.py
emcad_yellowblock_revise_v2_pretrain.py
vmunet_yellowblock_revise_v2_pretrain.py
```

- YellowBlock 上的第三方模型对比配置。
- 由 `tools/yellowblock/train_revise_external_yellowblock.py` 读取。
- 字段结构与 TP 第三方配置一致，但 `data_root='data/YellowBlock-TP'`。

```text
bevanet_s_yellowblock_revise_v2_pretrain_smoke500.py
emcad_yellowblock_revise_v2_pretrain_smoke500.py
vmunet_yellowblock_revise_v2_pretrain_smoke500.py
```

- YellowBlock 第三方模型 500 iter 冒烟测试配置。
- 用于快速验证环境、数据格式、模型 forward/backward 和日志输出是否正常。

## mmseg_ext/

`mmseg_ext/` 是本项目最核心的自定义代码目录，负责把 TP 数据、skeleton 监督和 clDice loss 接入 MMSegmentation。

```text
mmseg_ext/
├── __init__.py
├── datasets/
│   ├── __init__.py
│   └── tp_dataset.py
├── models/
│   ├── __init__.py
│   ├── tp_cldice_loss.py
│   ├── tp_skeleton_head.py
│   └── tp_skeleton_head_clDice.py
└── transforms/
    ├── __init__.py
    ├── generate_skeleton.py
    ├── map_tp_labels.py
    └── pack_seg_inputs_ext.py
```

```text
mmseg_ext/__init__.py
mmseg_ext/datasets/__init__.py
mmseg_ext/models/__init__.py
mmseg_ext/transforms/__init__.py
```

- 包初始化文件。
- 让 MMSeg 的 registry 能通过 `custom_imports` 找到自定义 dataset、transform、model、loss。

```text
mmseg_ext/datasets/tp_dataset.py
```

- 定义 `TPDataset(BaseSegDataset)`。
- 类别：`background` 与 `tactile_paving`，palette 为黑色背景和黄色盲道。
- 图像后缀 `.jpg`，标注后缀 `.png`。
- 按 `Index/*.txt` 中的相对路径读取样本，例如 `Part02/0416`。
- 图像路径为 `JPEGImages/<Part>/<id>.jpg`，标注路径为 `GroundTruth/<Part>/<id>.png`。
- `get_gt_seg_map_by_idx` 会把标注中的 `255` 映射为类别 `1`。

```text
mmseg_ext/transforms/map_tp_labels.py
```

- 定义 `MapTPLabels`。
- 在线标签映射 transform，默认把 `255 -> 1`。
- 解决原始二值 mask 使用 255 表示前景，而 MMSeg 训练需要类别 id 1 的问题。

```text
mmseg_ext/transforms/generate_skeleton.py
```

- 定义 `GenerateTPSkeleton` 和函数 `morph_skeleton`。
- 使用 OpenCV 形态学细化，从前景 mask 中生成骨架监督图 `gt_skeleton`。
- 输入语义 mask 中的 `target_label=1`，输出 `{0,1}` 的 skeleton 二值图。
- 将 `gt_skeleton` 加入 `seg_fields`，保证后续几何变换/打包时能被识别。

```text
mmseg_ext/transforms/pack_seg_inputs_ext.py
```

- 定义 `PackSegInputsExt` 和兼容别名 `PackSegInputsWithSkeleton`。
- 继承 MMSeg 的 `PackSegInputs`，额外把 `gt_skeleton` 写入 `data_samples.gt_skeleton`。
- 同时预留了 `gt_skel_dist`、`gt_edge_dist`、`gt_mask_dist` 等距离图监督字段。

```text
mmseg_ext/models/tp_skeleton_head.py
```

- 创新点 1 的模型实现：`TPSkeletonHead`。
- 结构为 `Conv3x3(in_channels -> channels)` + ReLU + `Conv1x1(channels -> 1)`。
- 输出单通道 skeleton logits。
- 损失包含 sigmoid BCE 和 Dice。
- 从 `batch_data_samples` 中读取 `gt_skeleton`，必要时 resize 到预测分辨率。

```text
mmseg_ext/models/tp_skeleton_head_clDice.py
```

- 创新点 2 的 aux head：`TPSkeletonHeadClDice`。
- 结构与 `TPSkeletonHead` 类似，但损失增加 `SoftCLDiceLoss`。
- 返回 `loss_skel_bce`、`loss_skel_dice`、`loss_skel_cldice`。
- 与 `segformer_b2_tp_skel_clDice*.py` 系列配置配合使用。

```text
mmseg_ext/models/tp_cldice_loss.py
```

- 定义 `SoftCLDiceLoss`，用于可微分拓扑/中心线约束。
- 支持二分类 logits `[N,1,H,W]` 和多类 logits `[N,C,H,W]`。
- 使用 soft erosion、soft dilation、soft opening 近似 skeletonize。
- 通过 topological precision 与 topological sensitivity 计算 clDice loss。
- 在主 decode head 中通常使用 `foreground_index=1`，在 skeleton head 单通道输出中使用 `foreground_index=0`。

## data/

```text
data/
├── TP-Dataset/
│   ├── GroundTruth/
│   ├── Index/
│   ├── JPEGImages/
│   └── SegmentationClassPNG/
├── YellowBlock.v1i.coco-segmentation/
└── YellowBlock-TP/
```

### data/TP-Dataset/

```text
GroundTruth/
```

- 原始 TP-Dataset 标注 mask。
- 按 `Part01` 到 `Part09` 分目录。
- 当前统计：`Part01=20`、`Part02=518`、`Part03=160`、`Part04=25`、`Part05=22`、`Part06=102`、`Part07=151`、`Part08=369`、`Part09=25`。
- mask 值主要为 `0/255`，训练时通过 `MapTPLabels` 或 `TPDataset.get_gt_seg_map_by_idx` 映射为 `0/1`。

```text
JPEGImages/
```

- 原始 RGB 图像。
- 按 `Part01` 到 `Part09` 分目录。
- 当前统计基本与 GroundTruth 对齐，但 `Part08` 图像统计为 368，GroundTruth 为 369，建议需要正式训练前做一次 dataset sanity 检查。

```text
SegmentationClassPNG/
```

- 另一份语义分割 PNG 标注目录。
- 按 Part 分目录，其中 `part02` 使用小写目录名。
- 当前训练配置主要使用 `GroundTruth/`，此目录更像历史/兼容标注副本。

```text
Index/
├── train.txt
├── val.txt
└── predict.txt
```

- 数据划分索引文件。
- 每行是无后缀相对 id，例如 `Part02/0416`。
- `train.txt` 用于训练，`val.txt` 用于验证/论文表格，`predict.txt` 用于预测集或额外导出。
- 当前没有 `test.txt`，TP 主实验的 `test_dataloader` 实际复用 `val_dataloader`。

### data/YellowBlock.v1i.coco-segmentation/

```text
train/
valid/
test/
```

- Roboflow COCO segmentation 格式原始导出。
- 每个 split 下包含图像和 `_annotations.coco.json`。
- 当前统计：`train` 约 1039 个文件，`valid` 约 242 个文件，`test` 约 124 个文件。
- 通过 `tools/yellowblock/convert_coco_to_tp_format.py` 转成 TP 风格目录。

### data/YellowBlock-TP/

```text
JPEGImages/
GroundTruth/
Index/
conversion_summary.json
```

- YellowBlock 转换后的 TP 风格数据集。
- `JPEGImages/`：转换/复制后的图像，共 1402 张。
- `GroundTruth/`：COCO polygon 栅格化后的二值 mask，共 1402 张。
- `Index/train.txt`、`Index/val.txt`、`Index/test.txt`：转换后的划分文件。
- `conversion_summary.json`：转换统计，包括每个 split 的样本数、非空 mask 数和比例。

## tools/

`tools/` 是实验自动化入口，覆盖训练、评估、预测导出、拓扑指标、消融补跑、表格构建和可视化。

```text
tools/
├── run_tp_experiment.py
├── train_iter_resume_noskip.py
├── check_dataset_sanity.py
├── export_preds_unique_by_index.py
├── eval_tp_topology.py
├── build_topo_v2_metrics.py
├── build_table2_topology.py
├── build_table2_visual_compare.py
├── build_table2_selected_montage.py
├── compare_seg_metrics_from_png.py
├── compare_official_eval_metrics.py
├── monitor_and_compare_cldice.py
├── run_ablation_suite.py
├── run_ablation_missing_only.py
├── ablation_table2_guard.py
├── rebuild_ablation_summary.py
├── run_table2_topology_once.sh
├── revise/
└── yellowblock/
```

```text
run_tp_experiment.py
```

- SegFormer baseline 与 skeleton 版本的统一训练/评估入口。
- 支持 `train`、`eval`、`train-eval` 三个子命令。
- 自动设置 `PYTHONPATH`，调用 `third_party/mmsegmentation/tools/train.py` 和 `test.py`。
- 可选调用 `tools/eval_tp_topology.py` 生成拓扑指标 CSV。

```text
train_iter_resume_noskip.py
```

- MMEngine iter-based resume 的本地补丁训练脚本。
- 解决 resume 时 `Advance dataloader N steps` 过慢或卡住的问题。
- 保留模型、优化器、scheduler、iter 状态恢复，但跳过 dataloader fast-forward。

```text
check_dataset_sanity.py
```

- 用真实 config pipeline 构建 dataset，检查 `gt_sem_seg`、`gt_skeleton` 是否存在、shape 是否匹配、unique value 是否合理。
- 可保存语义 mask、skeleton、overlay 可视化图。
- 用于验证 `MapTPLabels + GenerateTPSkeleton + PackSegInputsWithSkeleton` 是否正确串联。

```text
export_preds_unique_by_index.py
```

- 使用 MMSeg `init_model`/`inference_model` 对 split 导出预测 PNG。
- 输出文件名统一为索引形式，如 `0000.png`，避免不同 Part 下同名图片冲突。
- 同时生成 `pred_mapping.csv`，记录 `idx`、`part_id`、`pred_file`、`img_file`、`gt_file`。
- 是拓扑指标和可视化对比的基础工具。

```text
eval_tp_topology.py
```

- 对预测 PNG 与 GT mask 计算拓扑风格指标。
- 输出 per-sample CSV 和 summary CSV。
- 指标包括 Dice、IoU、clDice、预测/GT 连通域数、最大连通域占比、洞数量、前景像素数。

```text
build_topo_v2_metrics.py
```

- 构建新版拓扑指标。
- 重点输出 `T_prec`、`T_sens`、`clDice`、`Betti_Error`、`abs_dBeta0`、`abs_dBeta1`。
- 支持 resume，按模型逐个生成 per-model CSV 和总 summary。
- 默认读取 `work_dirs/table2_topology/preds` 和 `docs/tables/table2_topology_metrics.csv`。

```text
build_table2_topology.py
```

- 自动构建论文 Table 2 拓扑对比。
- 内置模型包括 baseline、skeleton、skeleton+clDice-v2、DeepLabV3+、PSPNet、UPerNet。
- 流程：导出预测 -> 计算拓扑指标 -> 写 `docs/tables/table2_topology_metrics.csv/md`。

```text
build_table2_visual_compare.py
```

- 生成每个样本的多模型可视化对比 panel。
- 列包含原图、GT、SegFormer-B2、Skeleton、Skeleton+clDice-v2、DeepLabV3+、PSPNet、UPerNet。
- 默认读取 `work_dirs/table2_topology/preds`。

```text
build_table2_selected_montage.py
```

- 从指定样本 index 中生成多行 montage 图。
- 适合论文/答辩中挑选代表性样本展示。

```text
compare_seg_metrics_from_png.py
```

- 不跑模型，仅根据已有预测 PNG 重新计算二分类 segmentation 指标。
- 支持 mapping CSV 对齐或 index 文件对齐。
- 输出 baseline 与 skel 的 mIoU、mDice、mFscore 等差异 CSV。

```text
compare_official_eval_metrics.py
```

- 从 MMSeg 官方 eval JSON/log 中提取指标。
- 适合论文主表使用官方 evaluator 输出，而不是从 PNG 重算。
- 会解析 `tactile_paving` 类别行，输出 baseline 与 skel 的指标差异。

```text
monitor_and_compare_cldice.py
```

- 监控 clDice 训练日志，在 1000/5000/10000 iter 捕获训练 loss 与验证指标。
- 训练完成后自动运行 baseline、skel、skel+clDice 三模型对比。
- 生成官方指标、预测图、拓扑指标和最终 summary CSV/JSON。

```text
run_ablation_suite.py
```

- clDice 消融实验批量运行脚本。
- 自动生成 `configs/tp_dataset/ablation_generated/*.py`，运行训练，并解析 best metrics。
- 支持 core 消融和权重扫描。

```text
run_ablation_missing_only.py
```

- 只补跑缺失/未完成的 ablation case。
- 会从已有 CSV、scalars、checkpoint 中判断是否已经完成。
- 输出 `work_dirs/ablation/ablation_results_completed.csv`。

```text
ablation_table2_guard.py
```

- 消融实验守护脚本。
- 监控指定 case 是否达到 10000 iter，必要时自动 resume。
- 完成后可触发 Table 2 拓扑评估。

```text
rebuild_ablation_summary.py
```

- 从 `work_dirs/ablation/*` 的 checkpoint 和 scalars 重建消融结果 master CSV。
- 输出 `work_dirs/ablation/ablation_results_master.csv` 和日志索引 CSV。

```text
run_table2_topology_once.sh
```

- 一次性运行 Table 2 拓扑评估的 shell 脚本。
- 注意：当前脚本参数名看起来与新版 `export_preds_unique_by_index.py`、`eval_tp_topology.py` 不完全一致，例如脚本里出现 `--output-dir`、`--gt-dir`、`--index-file`，新版 Python 脚本使用 `--out-dir`、`--gt-root`、`--split-file`。如果继续使用，建议先同步参数。

### tools/revise/

```text
tools/revise/
├── train_revise_external.py
├── build_revise_tables.py
├── build_fpn_attach_ablation_summary.py
├── run_all_revise_10k.sh
├── run_revise_v2_pretrain_queue.sh
└── run_skel_head_attach_ablation_f1f2f3f4.sh
```

```text
train_revise_external.py
```

- 第三方模型统一训练脚本，用于 VM-UNet、EMCAD、BEVANet。
- 自定义 `TPIndexDataset`，按 `Index/*.txt` 读取图像/mask。
- 统一实现随机缩放、裁剪、翻转、ImageNet normalize、CE loss、AMP、poly-like LR、验证、checkpoint 保存。
- 适配不同第三方模型输出格式：
  - VM-UNet：list 输出时最后一个作为 main。
  - EMCAD：list 输出时最后一个作为 main。
  - BEVANet：使用 semantic main，忽略 boundary aux 的 CE。
- 训练结束后导出 best 预测，并调用拓扑评估，生成 `metrics/final_summary.json`。

```text
build_revise_tables.py
```

- 将第三方对比模型的 final summary 合并进修订版 Table 1/2。
- 默认读取 `outputs/table1.txt`、`docs/tables/topo_v2/topology_v2_summary.csv` 和 `work_dirs/revise/*/metrics/final_summary.json`。
- 输出到 `docs/revise/table1_revise.*` 和 `docs/revise/table2_revise.*`。

```text
build_fpn_attach_ablation_summary.py
```

- 汇总 skeleton head 挂载位置 F1/F2/F3/F4 消融日志。
- 解析 best iter、mIoU、mDice、mFscore、mPrecision、mRecall。
- 输出 `docs/revise/fpn_attach_ablation_table.csv/md`。

```text
run_all_revise_10k.sh
```

- 按顺序训练 VM-UNet、EMCAD、BEVANet 的旧 revise 配置，并构建 revise 表格。
- 使用服务器 conda 环境 `/root/miniconda3/envs/tpseg/bin/python`。

```text
run_revise_v2_pretrain_queue.sh
```

- 服务器队列脚本，顺序运行 EMCAD、BEVANet、VM-UNet 的 V2 pretrain 配置。
- VM-UNet 使用单独 conda 环境 `vmunet`。

```text
run_skel_head_attach_ablation_f1f2f3f4.sh
```

- 顺序训练 skeleton head 接入层 F1/F2/F3/F4 消融。
- 训练完成后调用 `build_fpn_attach_ablation_summary.py` 生成汇总。

### tools/yellowblock/

```text
tools/yellowblock/
├── convert_coco_to_tp_format.py
├── train_revise_external_yellowblock.py
├── run_all_6_mmseg_10k_queue.sh
├── run_all_9_models_queue.sh
└── run_revise_v2_pretrain_queue.sh
```

```text
convert_coco_to_tp_format.py
```

- 将 Roboflow COCO segmentation 数据转换为 TP 风格目录。
- 输入：`data/YellowBlock.v1i.coco-segmentation/{train,valid,test}`。
- 输出：`data/YellowBlock-TP/JPEGImages`、`GroundTruth`、`Index/train.txt`、`Index/val.txt`、`Index/test.txt`。
- 支持 `merge` 或 `yellow-only` 类别映射。

```text
train_revise_external_yellowblock.py
```

- YellowBlock 版第三方模型统一训练脚本。
- 主体逻辑与 `tools/revise/train_revise_external.py` 相同。
- 增加 `topology_split`、`topology_gt_subdir`、`log_subdir`、`work_subdir` 等 YellowBlock 专用配置字段。

```text
run_all_6_mmseg_10k_queue.sh
```

- YellowBlock 上 6 个 MMSeg 模型/变体的 10k 队列训练脚本。
- 覆盖 baseline、skeleton、clDice 和 DeepLabV3+/PSPNet/UPerNet 等。

```text
run_all_9_models_queue.sh
```

- YellowBlock 上 9 模型队列脚本。
- 通常包含 6 个 MMSeg 变体加 3 个第三方 revise 模型。

```text
run_revise_v2_pretrain_queue.sh
```

- YellowBlock 上第三方 V2 pretrain 对比模型队列训练脚本。

## docs/

```text
docs/
├── env_snapshot.txt
├── project_structure_and_cleanup.md
├── revise/
└── tables/
```

```text
env_snapshot.txt
```

- 环境快照记录，用于复现实验环境。

```text
project_structure_and_cleanup.md
```

- 早期项目结构与清理记录。
- 其中中文部分当前显示为乱码，可能是历史编码问题。
- 记录了 baseline、skeleton 主线和已清理的旧 morph/thick 分支。

### docs/tables/

```text
ablation_tables_detailed.md
```

- clDice 消融实验的详细解释文档。
- 包含 core 组件消融、clDice 权重扫描、loss 公式、实验解释。

```text
ablation_table_core_components.csv
ablation_table_cldice_weights.csv
```

- 消融实验核心 CSV。
- `core_components` 对应 main/aux clDice 开关。
- `cldice_weights` 对应 clDice 权重扫描。

```text
add_mAcc/ablation_table_core_components_with_mAcc.csv
add_mAcc/ablation_table_cldice_weights_with_mAcc.csv
```

- 在原消融表基础上补充 `mAcc` 指标的版本。

```text
table2_topology_metrics.md
table2_topology_metrics.csv
```

- Table 2 拓扑对比结果。
- 包含 Dice、IoU、clDice、连通域数、最大连通域占比、洞数量、前景面积等。
- 对比模型包括 baseline、skeleton、skeleton+clDice-v2、DeepLabV3+、PSPNet、UPerNet。

```text
topo_v2/
├── topology_v2_per_sample.csv
├── topology_v2_summary.csv
├── topology_v2_summary.md
├── topology_v2_summary_no_outliers_iqr.csv
├── topology_v2_summary_no_outliers_iqr.md
└── per_model/
```

- 新版拓扑指标结果。
- 汇总 `T_prec`、`T_sens`、`clDice`、`Betti_Error` 等。
- `per_model/` 中每个模型有一份 per-sample CSV。
- `no_outliers_iqr` 是去除 IQR 离群样本后的统计版本。

### docs/revise/

```text
fpn_attach_ablation_table.csv
fpn_attach_ablation_table.md
```

- skeleton head 接入层 F1/F2/F3/F4 消融汇总。
- 当前结果中 F1、F3、F4 有日志，F2 标记为 missing log。

## outputs/

```text
outputs/
├── table1.txt
└── train_skel_cldice_v2_4090_bs4_10k_20260326_150548.log
```

```text
table1.txt
```

- 主实验 Table 1 文本版。
- 包含 SegFormer-B2 baseline、Skeleton、Skeleton+clDice-v2、DeepLabV3+、PSPNet、UPerNet 的 best-val 指标。
- 指标包括 aAcc、mIoU、mDice、mFscore、mPrecision、mRecall。

```text
train_skel_cldice_v2_4090_bs4_10k_20260326_150548.log
```

- `SegFormer-B2 + Skeleton + clDice-v2` 4090 bs4 训练日志副本。
- 用于追溯 Table 1 中 clDice-v2 的训练过程。

## logs/

```text
logs/
├── ablation/
├── revise/
├── table2_topology/
├── yellowblock/
└── train_skel_stage2_*.log 等根目录日志
```

- `logs/ablation/`：clDice 消融训练、守护、补跑和重建日志。
- `logs/revise/`：第三方对比模型和 F1/F2/F3/F4 挂载层消融日志。
- `logs/table2_topology/`：Table 2 预测导出与拓扑评估日志。
- `logs/yellowblock/`：YellowBlock 数据集上的 mmseg/revise 训练日志。
- 根目录下若干 `train_skel_stage2_*`、`topo_v2_resume_*` 日志是 skeleton-only 和拓扑指标重跑记录。

## experiments/

```text
experiments/
├── baseline_segformer_b2/
└── tp/
    └── segformer_b2_baseline/
```

- 实验分组/记录目录。
- 当前主要是目录占位，实际训练配置、日志、结果分别位于 `configs/`、`logs/`、`work_dirs/`、`docs/`。

## third_party/

```text
third_party/
├── mmsegmentation/
└── revise_models/
    ├── bevanet/
    ├── emcad/
    ├── VM-UNet/
    ├── emcad.tar.gz
    └── emcad.zip
```

### third_party/mmsegmentation/

- 上游 MMSegmentation 源码。
- 本项目通过配置 `_base_` 继承其中的 SegFormer、DeepLabV3+、PSPNet、UPerNet 官方配置。
- 训练和测试入口使用：
  - `third_party/mmsegmentation/tools/train.py`
  - `third_party/mmsegmentation/tools/test.py`
- 主要子目录：
  - `configs/`：官方模型配置库。
  - `mmseg/`：MMSeg 核心源码，包括 datasets、models、engine、evaluation、transforms 等。
  - `tools/`：官方训练、测试、转换、部署脚本。
  - `docs/`、`demo/`、`tests/`：上游文档、示例和测试。

### third_party/revise_models/bevanet/

- BEVANet 官方/第三方实现，用于论文对比模型。
- 关键文件/目录：
  - `README.md`、`LICENSE`、`requirements.txt`：来源说明、许可证和依赖。
  - `models/BEVANet.py`：BEVANet 主模型。
  - `models/EVAN.py`、`FB.py`、`FS.py`、`LSKA.py`、`PPM.py`：BEVANet 组成模块。
  - `configs/default.py`：BEVANet 原始配置对象，训练脚本会动态改写其中的模型超参。
  - `pretrained_models/imagenet/`：ImageNet 预训练权重目录。
  - `tools/train.py`、`tools/eval.py`、`tools/pretrain.py`：原项目脚本，本项目主要通过 `tools/revise/train_revise_external.py` 调用模型本体。

### third_party/revise_models/emcad/

- EMCAD 官方/第三方实现，用于论文对比模型。
- 关键文件/目录：
  - `lib/networks.py`：`EMCADNet` 主网络入口。
  - `lib/decoders.py`、`lib/pvtv2.py`、`lib/resnet.py`：decoder 与 encoder/backbone 实现。
  - `pretrained_pth/pvt/`：PVTv2 预训练权重目录。
  - `trainer.py`、`train_polyp.py`、`train_synapse.py`、`test_*.py`：原项目训练/测试脚本。
  - `utils/`：原项目 dataloader、transform、misc、数据预处理工具。
  - `EMCAD_architecture.jpg`、`avg_dice_*.png`、`qualitative_results_*.png`：原项目图示和结果图片。

### third_party/revise_models/VM-UNet/

- VM-UNet 官方/第三方实现，用于论文对比模型。
- 关键文件/目录：
  - `models/vmunet/vmunet.py`：VM-UNet 主模型。
  - `models/vmunet/vmamba.py`：VMamba backbone/模块。
  - `pre_trained_weights/vmamba_small_e238_ema.pth`：预训练权重。
  - `datasets/dataset.py`、`engine.py`、`train.py`、`utils.py`：原项目数据、训练和工具代码。
  - 本项目通过 `tools/revise/train_revise_external.py` 加载其模型并统一训练/评估。

## work_dirs/

```text
work_dirs/
├── segformer_b2_tp/
├── segformer_b2_tp_skel/
├── segformer_b2_tp_skel_stage2_4090_bs4_stable/
├── segformer_b2_tp_skel_cldice_v2_4090_bs4/
├── ablation/
├── revise/
├── table2_topology/
├── yellowblock/
└── 若干 debug/vis/topk 临时目录
```

- 训练产物目录，包含 checkpoint、`last_checkpoint`、vis_data、scalars、预测 PNG、mapping CSV、拓扑 CSV、临时可视化等。
- 本目录体量较大，并且你说明大部分内容在服务器上，因此本文不逐文件展开。
- 与论文表格相关的产物通常会被脚本复制或汇总到 `docs/`、`outputs/`、`logs/`。

## 缓存和本地环境目录

```text
.venv/
```

- 本地 Python 虚拟环境，不属于项目源码。

```text
__pycache__/
```

- Python 运行产生的字节码缓存。
- 多个目录下都有，不影响实验逻辑，可忽略。

```text
.ipynb_checkpoints/
```

- Jupyter 自动保存检查点。
- 根目录下当前有 `ENVIRONMENT-checkpoint.md` 和 `requirements_freeze-checkpoint.txt`。
- `configs/`、`docs/`、`mmseg_ext/`、`logs/` 下也有少量 checkpoint 缓存，均非主线文件。

## 实验主线速查

```text
Baseline:
  config: configs/tp_dataset/segformer_b2_tp.py
  model: SegFormer-B2
  output: work_dirs/segformer_b2_tp

Innovation 1:
  config: configs/tp_dataset/segformer_b2_tp_skel.py
  model: SegFormer-B2 + TPSkeletonHead
  code: mmseg_ext/models/tp_skeleton_head.py
  skeleton target: mmseg_ext/transforms/generate_skeleton.py

Innovation 2:
  config: configs/tp_dataset/segformer_b2_tp_skel_clDice_v2_4090_bs4.py
  model: SegFormer-B2 + TPSkeletonHeadClDice + SoftCLDiceLoss
  code: mmseg_ext/models/tp_skeleton_head_clDice.py
  loss: mmseg_ext/models/tp_cldice_loss.py

MMSeg comparison:
  configs/tp_dataset/deeplabv3plus_r50_tp_main_table1_B.py
  configs/tp_dataset/pspnet_r50_tp_main_table1_B.py
  configs/tp_dataset/upernet_r50_tp_main_table1_B.py

External comparison:
  configs/tp_dataset/bevanet_s_tp_revise_v2_pretrain.py
  configs/tp_dataset/emcad_tp_revise_v2_pretrain.py
  configs/tp_dataset/vmunet_tp_revise_v2_pretrain.py
  runner: tools/revise/train_revise_external.py

Ablation:
  configs/tp_dataset/ablation_generated/*.py
  runner: tools/run_ablation_suite.py
  repair/summary: tools/run_ablation_missing_only.py, tools/rebuild_ablation_summary.py
  docs: docs/tables/ablation_tables_detailed.md

Topology evaluation:
  export: tools/export_preds_unique_by_index.py
  eval v1: tools/eval_tp_topology.py
  eval v2: tools/build_topo_v2_metrics.py
  table: tools/build_table2_topology.py
  docs: docs/tables/table2_topology_metrics.md, docs/tables/topo_v2/
```
