import numpy as np
from mmseg.registry import TRANSFORMS
from mmcv.transforms import BaseTransform

@TRANSFORMS.register_module()
class GenerateSkeletonDistMap(BaseTransform):
    """Generate a thickness-like distance map from gt_skeleton (and foreground mask).

    Expects:
        results['gt_skeleton'] with shape (H, W), values {0,1} or {0,255}
        results contains a foreground semantic mask:
            - results['gt_seg_map'] (mmseg LoadAnnotations output), OR
            - results['gt_sem_seg'] (after Pack), OR
            - results['gt_semantic_seg'] (some pipelines)
    Produces:
        results['gt_skel_dist'] float32 (H, W)
            background = 0
            inside foreground: larger near skeleton, smaller near boundary
    """

    def __init__(self, clip_max: float = 64.0, use_cv2: bool = True):
        self.clip_max = float(clip_max)
        self.use_cv2 = bool(use_cv2)

    def _dist_to_skeleton(self, skel_bin: np.ndarray) -> np.ndarray:
        # distance-to-skeleton, skeleton pixels distance=0
        img = np.where(skel_bin > 0, 0, 255).astype(np.uint8)

        if self.use_cv2:
            try:
                import cv2
                return cv2.distanceTransform(img, cv2.DIST_L2, 3).astype(np.float32)
            except Exception:
                pass

        try:
            from scipy.ndimage import distance_transform_edt
            inv = (skel_bin == 0).astype(np.uint8)
            return distance_transform_edt(inv).astype(np.float32)
        except Exception as e:
            raise RuntimeError("GenerateSkeletonDistMap needs opencv-python or scipy.") from e

    @staticmethod
    def _get_fg_mask(results: dict) -> np.ndarray:
        # Try common keys produced by mmseg pipelines
        for k in ["gt_seg_map", "gt_sem_seg", "gt_semantic_seg"]:
            if k in results:
                m = np.asarray(results[k])
                # LoadAnnotations gives (H,W), sometimes (1,H,W)
                if m.ndim == 3 and m.shape[0] == 1:
                    m = m[0]
                return (m > 0)
        raise KeyError(
            "Foreground mask not found. Put GenerateSkeletonDistMap after LoadAnnotations/MapTPLabels "
            "so results contains gt_seg_map (or gt_sem_seg)."
        )

    def transform(self, results: dict) -> dict:
        if "gt_skeleton" not in results:
            raise KeyError("`gt_skeleton` not found. Put GenerateTPSkeleton before this.")

        skel = np.asarray(results["gt_skeleton"])
        skel_bin = (skel > 0).astype(np.uint8)

        fg = self._get_fg_mask(results).astype(bool)

        # 1) compute distance-to-skeleton on full image
        dist = self._dist_to_skeleton(skel_bin)

        # 2) keep only inside foreground
        dist = dist * fg.astype(np.float32)

        # 3) clip (optional) BEFORE invert is fine; we will invert within fg only
        if self.clip_max > 0:
            dist = np.clip(dist, 0.0, self.clip_max)

        # 4) invert within foreground: skeleton becomes large, boundary becomes small
        max_val = float(dist[fg].max()) if fg.any() else 0.0
        if max_val > 0:
            dist_inv = np.zeros_like(dist, dtype=np.float32)
            dist_inv[fg] = max_val - dist[fg]
            dist = dist_inv

        # 5) background hard zero
        dist[~fg] = 0.0

        results["gt_skel_dist"] = dist.astype(np.float32)
        return results

    def __repr__(self):
        return f"{self.__class__.__name__}(clip_max={self.clip_max}, use_cv2={self.use_cv2})"
