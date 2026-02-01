from mmseg.registry import DATASETS
from mmseg.datasets import BaseSegDataset
import os.path as osp
import mmcv
import numpy as np
from pathlib import Path

@DATASETS.register_module()
class TPDataset(BaseSegDataset):
    """TP-Dataset: images in JPEGImages, masks in GroundTruth.
    Split file lines look like: Part02/0416
    Mask png values: {0, 255} -> we map 255 to 1 online.
    """
    METAINFO = dict(
        classes=('background', 'tactile_paving'),
        palette=[[0, 0, 0], [255, 255, 0]],
    )

    def __init__(self, **kwargs):
        super().__init__(
            img_suffix='.jpg',
            seg_map_suffix='.png',
            reduce_zero_label=False,
            **kwargs
        )

    def load_data_list(self):
        data_list = []
        ann_path = self.ann_file   # 关键：不要自己拼 data_root

        with open(ann_path, 'r') as f:
            lines = [ln.strip() for ln in f.readlines() if ln.strip()]

        for rel in lines:
            # rel: Part02/0416
            img_path = osp.join(self.data_prefix['img_path'], rel + self.img_suffix)
            seg_path = osp.join(self.data_prefix['seg_map_path'], rel + self.seg_map_suffix)

            data_list.append(dict(
                img_path=img_path,
                seg_map_path=seg_path,
                seg_fields=[],
                reduce_zero_label=self.reduce_zero_label,
            ))
        return data_list


    def get_gt_seg_map_by_idx(self, idx):
        """Read seg map and do online mapping: 255 -> 1."""
        seg_map_path = self.data_list[idx]['seg_map_path']
        gt = mmcv.imread(seg_map_path, flag='unchanged')  # HxW or HxWxC

        # 若读到 3 通道，取单通道
        if gt.ndim == 3:
            gt = gt[..., 0]

        gt = gt.astype(np.uint8)
        gt[gt == 255] = 1
        return gt
