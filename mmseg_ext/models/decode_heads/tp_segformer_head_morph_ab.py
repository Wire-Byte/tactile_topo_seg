import torch
import torch.nn.functional as F
from mmseg.registry import MODELS
from mmseg.models.decode_heads import SegformerHead


@MODELS.register_module()
class TPSegformerHeadMorphAB(SegformerHead):
    """
    SegformerHead + (A) background-only distance-aware BCE + (B) anti-island hard-opening suppression.

    Requires data_samples contain:
      - gt_sem_seg   (PixelData): (1,H,W) {0,1}
      - gt_mask_dist (PixelData): (1,H,W) float (outside-GT distance-to-foreground), inside-GT usually 0
    """

    def __init__(self,
                 loss_bg_suppress=None,
                 loss_anti_island=None,
                 **kwargs):
        # ---- tolerate leftover keys from other heads/configs ----
        # These keys belong to your old Morph head; SegformerHead/BaseDecodeHead doesn't accept them.
        legacy_keys = [
            # tube / morph related
            "tube_mode", "tube_ratio", "tube_quantile", "tube_q", "tube_q_low", "tube_q_high",
            "tau_fixed", "tau_fp", "tau_near", "tau_far",
            "tube_kernel", "tube_ks", "tube_dilate", "tube_erode",

            # weights / lambdas
            "lambda_tube", "lambda_fp", "lambda_island",
            "w_tube", "w_fp", "w_island",

            # dist switches / names
            "use_mask_dist", "use_edge_dist", "dist_key", "edge_key", "mask_key",

            # debug / misc
            "debug", "dbg", "save_debug", "vis_debug",
        ]
        for k in legacy_keys:
            if k in kwargs:
                kwargs.pop(k)

        super().__init__(**kwargs)
        self.loss_bg_suppress = MODELS.build(loss_bg_suppress) if loss_bg_suppress is not None else None
        self.loss_anti_island = MODELS.build(loss_anti_island) if loss_anti_island is not None else None

    @staticmethod
    def _stack_map(batch_data_samples, attr: str, device, size_hw):
        """Stack PixelData map from data_samples to (B,1,H,W) with resize."""
        H, W = size_hw
        maps = []
        for ds in batch_data_samples:
            if not hasattr(ds, attr):
                return None
            m = getattr(ds, attr)
            if hasattr(m, "data"):
                m = m.data
            if m.dim() == 2:
                m = m.unsqueeze(0)
            m = m.unsqueeze(0).float().to(device)  # (1,1,h,w)
            if m.shape[-2:] != (H, W):
                m = F.interpolate(m, size=(H, W), mode="nearest")
            maps.append(m)
        return torch.cat(maps, dim=0)  # (B,1,H,W)

    def loss_by_feat(self, seg_logits, batch_data_samples):
        losses = super().loss_by_feat(seg_logits, batch_data_samples)

        B, _, H, W = seg_logits.shape
        device = seg_logits.device

        gt = self._stack_map(batch_data_samples, "gt_sem_seg", device, (H, W))
        if gt is None:
            return losses

        if self.loss_bg_suppress is not None:
            mdist = self._stack_map(batch_data_samples, "gt_mask_dist", device, (H, W))
            if mdist is None:
                raise ValueError("loss_bg_suppress requires gt_mask_dist in data_samples!")
            losses["loss_bg_dist_bce"] = self.loss_bg_suppress(seg_logits, gt, mdist)

        if self.loss_anti_island is not None:
            losses["loss_anti_island"] = self.loss_anti_island(seg_logits, gt)

        return losses
