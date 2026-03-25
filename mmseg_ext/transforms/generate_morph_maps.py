# mmseg_ext/transforms/generate_morph_maps.py
import numpy as np
from mmseg.registry import TRANSFORMS
from mmcv.transforms import BaseTransform

@TRANSFORMS.register_module()
class GenerateTPMorphMaps(BaseTransform):
    """Generate morphological distance maps for structure-aware regularization.

    Requires a foreground mask in results (after LoadAnnotations/MapTPLabels):
      - results['gt_seg_map'] or results['gt_sem_seg'] or results['gt_semantic_seg']

    Produces:
      - results['gt_edge_dist']: float32 (H,W)
          distance-to-boundary INSIDE GT foreground; 0 outside.
      - results['gt_mask_dist']: float32 (H,W)
          distance-to-foreground for background pixels; 0 inside GT.
    """

    def __init__(self, clip_edge: float = 64.0, clip_mask: float = 128.0, use_cv2: bool = True):
        self.clip_edge = float(clip_edge)
        self.clip_mask = float(clip_mask)
        self.use_cv2 = bool(use_cv2)

    @staticmethod
    def _get_fg_mask(results: dict) -> np.ndarray:
        for k in ["gt_seg_map", "gt_sem_seg", "gt_semantic_seg"]:
            if k in results:
                m = np.asarray(results[k])
                if m.ndim == 3 and m.shape[0] == 1:
                    m = m[0]
                return (m > 0)
        raise KeyError("Foreground mask not found. Put GenerateTPMorphMaps after LoadAnnotations/MapTPLabels.")

    @staticmethod
    def _dt_to_zeros(img_u8: np.ndarray, use_cv2: bool) -> np.ndarray:
        # distanceTransform computes distance to zero pixels
        if use_cv2:
            try:
                import cv2
                return cv2.distanceTransform(img_u8, cv2.DIST_L2, 3).astype(np.float32)
            except Exception:
                pass
        try:
            from scipy.ndimage import distance_transform_edt
            # distance to zeros: edt on (img!=0)
            return distance_transform_edt(img_u8 != 0).astype(np.float32)
        except Exception as e:
            raise RuntimeError("GenerateTPMorphMaps needs opencv-python or scipy.") from e

    def transform(self, results: dict) -> dict:
        fg = self._get_fg_mask(results).astype(bool)  # GT foreground mask

        # 1) edge_dist: inside GT, distance to boundary (background pixels are zeros)
        # zeros = background; non-zero = foreground
        img_edge = np.where(fg, 255, 0).astype(np.uint8)
        edge_dist = self._dt_to_zeros(img_edge, self.use_cv2)
        edge_dist[~fg] = 0.0
        if self.clip_edge > 0:
            edge_dist = np.clip(edge_dist, 0.0, self.clip_edge)

        # 2) mask_dist: outside GT, distance to foreground (foreground pixels are zeros)
        img_mask = np.where(fg, 0, 255).astype(np.uint8)
        mask_dist = self._dt_to_zeros(img_mask, self.use_cv2)
        mask_dist[fg] = 0.0
        if self.clip_mask > 0:
            mask_dist = np.clip(mask_dist, 0.0, self.clip_mask)

        results["gt_edge_dist"] = edge_dist.astype(np.float32)
        results["gt_mask_dist"] = mask_dist.astype(np.float32)
        return results

    def __repr__(self):
        return (f"{self.__class__.__name__}(clip_edge={self.clip_edge}, "
                f"clip_mask={self.clip_mask}, use_cv2={self.use_cv2})")
