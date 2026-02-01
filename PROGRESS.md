# Tactile Topo Seg Progress (保存点)

## 环境
- Platform: Alibaba Cloud DSW
- Python: 3.11
- Torch: 2.9.1 + CUDA 12.8
- mmcv: 2.1.0 (built)
- mmseg: 1.2.2
- Dataset: TP-Dataset (GroundTruth with {0,255})

## 已完成
- 数据集目录整理完成
- 自定义 TPDataset 已注册 (mmseg_ext.datasets)
- segformer_b2_tp.py 配置可加载
- 50 iter smoke train 成功跑通
- GroundTruth label: {0,255}

## 当前问题
- BaseSegDataset 不支持 label_map
- 需要在 pipeline 中做 255 -> 1 映射
- tp_dataset.py 中修复 ann_file 路径拼接（避免 data_root 重复）

## 下一步
1. 修改 tp_dataset.py 的 load_data_list
2. 在 pipeline 中加 LabelMap / custom transform
3. 重新跑 train smoke test
