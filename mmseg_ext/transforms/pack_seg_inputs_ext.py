import numpy as np
import torch
from mmseg.registry import TRANSFORMS
from mmseg.datasets.transforms import PackSegInputs
from mmengine.structures import PixelData

@TRANSFORMS.register_module()
class PackSegInputsWithSkeleton(PackSegInputs):
    """PackSegInputs + additionally pack gt_skeleton into data_samples.gt_skeleton."""

    def transform(self, results: dict) -> dict:
        packed = super().transform(results)
        data_samples = packed['data_samples']

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
        return packed
