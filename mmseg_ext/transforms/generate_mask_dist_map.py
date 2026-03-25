import numpy as np
from mmcv.transforms import BaseTransform
from mmseg.registry import TRANSFORMS


def _edt_2d_bg_to_fg(fg_mask: np.ndarray) -> np.ndarray:
    """
    Compute background-to-foreground distance transform.
    Returns dist for all pixels, but we'll later zero it inside FG.
    fg_mask: bool array (H, W), True for foreground.
    """
    H, W = fg_mask.shape
    bg = ~fg_mask

    # If no FG, distance is all zeros (avoid weirdness)
    if fg_mask.sum() == 0:
        return np.zeros((H, W), dtype=np.float32)

    # If scipy is available, use it (fast, correct)
    try:
        from scipy.ndimage import distance_transform_edt
        # distance_transform_edt computes distance to nearest zero.
        # We want distance from BG pixels to nearest FG pixel.
        # So set zeros at FG, ones at BG, then EDT gives BG->FG distances.
        dist = distance_transform_edt(bg.astype(np.uint8)).astype(np.float32)
        return dist
    except Exception:
        # Fallback: simple BFS-based distance (Manhattan) if scipy missing
        # (slower but works). We'll do 4-neighborhood.
        from collections import deque
        dist = np.full((H, W), fill_value=np.inf, dtype=np.float32)
        q = deque()

        # Initialize queue with all FG pixels at dist=0
        ys, xs = np.where(fg_mask)
        for y, x in zip(ys.tolist(), xs.tolist()):
            dist[y, x] = 0.0
            q.append((y, x))

        dirs = [(1, 0), (-1, 0), (0, 1), (0, -1)]
        while q:
            y, x = q.popleft()
            d0 = dist[y, x]
            for dy, dx in dirs:
                ny, nx = y + dy, x + dx
                if 0 <= ny < H and 0 <= nx < W:
                    nd = d0 + 1.0
                    if nd < dist[ny, nx]:
                        dist[ny, nx] = nd
                        q.append((ny, nx))
        # convert inf -> 0 (shouldn't happen if FG exists)
        dist[~np.isfinite(dist)] = 0.0
        return dist.astype(np.float32)


@TRANSFORMS.register_module()
class GenerateMaskDistMap(BaseTransform):
    """
    Generate gt_mask_dist: distance-from-background-to-foreground map.
    - Input: results['gt_seg_map'] (H,W) with fg_label=1, bg_label=0 (your TP dataset)
    - Output: results['gt_mask_dist'] (H,W) float32
      where dist is >0 on background, 0 inside foreground.
    """
    def __init__(self, fg_label=1, clip_max=None):
        self.fg_label = int(fg_label)
        self.clip_max = None if clip_max is None else float(clip_max)

    def transform(self, results: dict) -> dict:
        if "gt_seg_map" not in results:
            # LoadAnnotations must be before this transform
            raise KeyError("GenerateMaskDistMap requires 'gt_seg_map' in results. Put it after LoadAnnotations.")

        gt = results["gt_seg_map"]
        if gt.ndim == 3:
            gt = gt.squeeze(-1)

        # 原来可能是 fg = (gt == self.fg_label)
        # 改成：
        fg = (gt > 0)

        dist = _edt_2d_bg_to_fg(fg)  # (H,W) float32

        # Zero inside FG (we only need BG distances for FP suppression)
        dist = dist * (~fg).astype(np.float32)

        if self.clip_max is not None:
            dist = np.clip(dist, 0.0, self.clip_max)

        results["gt_mask_dist"] = dist.astype(np.float32)
        return results
