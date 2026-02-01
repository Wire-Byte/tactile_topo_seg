from mmcv.transforms import BaseTransform
from mmseg.registry import TRANSFORMS
import numpy as np


@TRANSFORMS.register_module()
class MapTPLabels(BaseTransform):
    """Map TP-Dataset mask values online, e.g. 255->1."""

    def __init__(self, mapping=None):
        # default: 255 -> 1
        self.mapping = {255: 1} if mapping is None else {int(k): int(v) for k, v in mapping.items()}

    def transform(self, results: dict) -> dict:
        gt = results.get('gt_seg_map', None)
        if gt is None:
            return results

        # ensure numpy array
        if not isinstance(gt, np.ndarray):
            gt = np.array(gt)

        for src, dst in self.mapping.items():
            gt[gt == src] = dst

        results['gt_seg_map'] = gt
        return results

    def __repr__(self):
        return f'{self.__class__.__name__}(mapping={self.mapping})'
