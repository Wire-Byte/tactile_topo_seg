import numpy as np
import cv2
from mmcv.transforms.base import BaseTransform
from mmseg.registry import TRANSFORMS

def morph_skeleton(binary: np.ndarray) -> np.ndarray:
    """Binary skeletonization using morphological operations.
    Input: binary {0,1} or {0,255}, uint8.
    Output: skeleton {0,1}, uint8.
    """
    img = (binary > 0).astype(np.uint8) * 255
    skel = np.zeros_like(img, dtype=np.uint8)

    element = cv2.getStructuringElement(cv2.MORPH_CROSS, (3, 3))
    while True:
        opened = cv2.morphologyEx(img, cv2.MORPH_OPEN, element)
        temp = cv2.subtract(img, opened)
        eroded = cv2.erode(img, element)
        skel = cv2.bitwise_or(skel, temp)
        img = eroded.copy()
        if cv2.countNonZero(img) == 0:
            break

    # binarize to {0,1}
    return (skel > 0).astype(np.uint8)

@TRANSFORMS.register_module()
class GenerateTPSkeleton(BaseTransform):
    """Generate skeleton target from gt_seg_map for tactile_paving class.

    Assumes gt_seg_map is {0,1} after MapTPLabels (255->1) or similar.
    Adds:
      results['gt_skeleton'] : uint8 {0,1} with same HxW
    """
    def __init__(self, target_label: int = 1):
        self.target_label = target_label

    def transform(self, results: dict) -> dict:
        gt = results.get('gt_seg_map', None)
        if gt is None:
            raise KeyError("gt_seg_map not found. Put GenerateTPSkeleton after LoadAnnotations/MapTPLabels.")
        # gt could be HxW int array
        binary = (gt == self.target_label).astype(np.uint8)
        skel = morph_skeleton(binary)
        results['gt_skeleton'] = skel  # HxW, {0,1}
        
        if 'seg_fields' not in results:
            results['seg_fields'] = []
        if 'gt_skeleton' not in results['seg_fields']:
            results['seg_fields'].append('gt_skeleton')

        return results
