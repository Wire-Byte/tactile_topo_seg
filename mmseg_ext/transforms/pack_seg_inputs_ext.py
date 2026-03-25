import numpy as np
import torch
from mmseg.registry import TRANSFORMS
from mmseg.datasets.transforms import PackSegInputs
from mmengine.structures import PixelData


def _to_1chw(x, dtype=None):
    """Convert numpy/torch HxW or 1xHxW to torch 1xHxW."""
    if x is None:
        return None
    if isinstance(x, np.ndarray):
        t = torch.from_numpy(x)
    elif torch.is_tensor(x):
        t = x
    else:
        raise TypeError(f"Unsupported type: {type(x)}")

    if t.ndim == 2:
        t = t.unsqueeze(0)
    elif t.ndim == 3:
        # assume already (1,H,W) or (C,H,W)
        pass
    else:
        raise ValueError(f"Unexpected ndim={t.ndim}, expect 2 or 3")

    if dtype is not None:
        t = t.to(dtype)
    return t


@TRANSFORMS.register_module()
class PackSegInputsExt(PackSegInputs):
    """PackSegInputs + pack extra supervision maps into data_samples."""

    def transform(self, results: dict) -> dict:
        packed = super().transform(results)
        data_samples = packed["data_samples"]

        # gt_skeleton: (H,W) -> (1,H,W) long
        if "gt_skeleton" in results and results["gt_skeleton"] is not None:
            skel = _to_1chw(results["gt_skeleton"], dtype=torch.long)
            data_samples.gt_skeleton = PixelData(data=skel)

        # gt_skel_dist: (H,W) -> (1,H,W) float
        if "gt_skel_dist" in results and results["gt_skel_dist"] is not None:
            dist = _to_1chw(results["gt_skel_dist"], dtype=torch.float32)
            data_samples.gt_skel_dist = PixelData(data=dist)

        # gt_edge_dist: (H,W) -> (1,H,W) float
        if "gt_edge_dist" in results and results["gt_edge_dist"] is not None:
            edge = _to_1chw(results["gt_edge_dist"], dtype=torch.float32)
            data_samples.gt_edge_dist = PixelData(data=edge)

        # gt_mask_dist: (H,W) -> (1,H,W) float
        if "gt_mask_dist" in results and results["gt_mask_dist"] is not None:
            md = _to_1chw(results["gt_mask_dist"], dtype=torch.float32)
            data_samples.gt_mask_dist = PixelData(data=md)

        return packed


# 兼容旧名字（如果你别处还在用）
@TRANSFORMS.register_module()
class PackSegInputsWithSkeleton(PackSegInputsExt):
    pass
