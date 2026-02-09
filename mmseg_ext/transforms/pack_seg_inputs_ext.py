import numpy as np
import torch
from mmseg.registry import TRANSFORMS
from mmseg.datasets.transforms import PackSegInputs
from mmengine.structures import PixelData

@TRANSFORMS.register_module()
class PackSegInputsWithSkeleton(PackSegInputs):
    """PackSegInputs + additionally pack gt_skeleton and gt_skel_dist into data_samples."""

    def transform(self, results: dict) -> dict:
        packed = super().transform(results)
        data_samples = packed['data_samples']

        # ---- pack gt_skeleton (as you already did) ----
        if 'gt_skeleton' in results:
            skel = results['gt_skeleton']
            if isinstance(skel, np.ndarray):
                # HxW -> 1xHxW
                skel = torch.from_numpy(skel).long().unsqueeze(0)
            elif torch.is_tensor(skel):
                if skel.ndim == 2:
                    skel = skel.long().unsqueeze(0)
                else:
                    skel = skel.long()
            else:
                raise TypeError(f"Unsupported gt_skeleton type: {type(skel)}")

            data_samples.gt_skeleton = PixelData(data=skel)

        # ---- NEW: pack gt_skel_dist (distance-to-skeleton map) ----
        if 'gt_skel_dist' in results and results['gt_skel_dist'] is not None:
            dist = results['gt_skel_dist']  # expected (H,W) float32
            if isinstance(dist, np.ndarray):
                # HxW -> 1xHxW
                if dist.ndim == 2:
                    dist = torch.from_numpy(dist).float().unsqueeze(0)
                else:
                    dist = torch.from_numpy(dist).float()
            elif torch.is_tensor(dist):
                if dist.ndim == 2:
                    dist = dist.float().unsqueeze(0)
                else:
                    dist = dist.float()
            else:
                raise TypeError(f"Unsupported gt_skel_dist type: {type(dist)}")

            data_samples.gt_skel_dist = PixelData(data=dist)

        return packed
