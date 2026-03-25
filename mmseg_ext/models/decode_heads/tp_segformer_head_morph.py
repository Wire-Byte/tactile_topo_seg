# mmseg_ext/models/decode_heads/tp_segformer_head_morph.py
import torch
import torch.nn.functional as F
from mmseg.registry import MODELS
from mmseg.models.decode_heads import SegformerHead

@MODELS.register_module()
class TPSegformerHeadMorph(SegformerHead):
    """SegformerHead + Morphological Consistency Regularization (C + small B).

    Requires data_samples contain:
      - gt_edge_dist (PixelData): (1,H,W) float, inside-GT dist-to-boundary, 0 outside
      - gt_mask_dist (PixelData): (1,H,W) float, outside-GT dist-to-foreground, 0 inside
      - gt_sem_seg  (PixelData): (1,H,W) 0/1 (after MapTPLabels)
    """

    def __init__(self,
                 lambda_tube: float = 0.05,
                 lambda_fp: float = 0.01,
                 # C fixed-tau tube
                 use_fixed_tube: bool = True,
                 tau_fixed: float = 8.0,
                 # C adaptive (DT gradient) alternative
                 use_adaptive_center: bool = True,
                 center_power: float = 1.0,
                 # B fp outside weighting
                 tau_fp: float = 12.0,
                 **kwargs):
        super().__init__(**kwargs)
        self.lambda_tube = float(lambda_tube)
        self.lambda_fp = float(lambda_fp)

        self.use_fixed_tube = bool(use_fixed_tube)
        self.tau_fixed = float(tau_fixed)

        self.use_adaptive_center = bool(use_adaptive_center)
        self.center_power = float(center_power)

        self.tau_fp = float(tau_fp)

    @staticmethod
    def _fg_prob(seg_logits: torch.Tensor) -> torch.Tensor:
        # (N,C,H,W) -> (N,1,H,W)
        if seg_logits.size(1) == 1:
            return torch.sigmoid(seg_logits)
        probs = F.softmax(seg_logits, dim=1)
        return probs[:, 1:2, ...]

    def _stack_map(self, batch_data_samples, attr: str, device, size_hw):
        """Stack PixelData map from data_samples to (N,1,H,W) with resize."""
        H, W = size_hw
        maps = []
        for ds in batch_data_samples:
            if not hasattr(ds, attr):
                return None
            m = getattr(ds, attr).data  # (1,h,w) tensor
            if m.dim() == 2:
                m = m.unsqueeze(0)
            m = m.unsqueeze(0).float().to(device)  # (1,1,h,w)
            if m.shape[-2:] != (H, W):
                m = F.interpolate(m, size=(H, W), mode='nearest')
            maps.append(m)
        return torch.cat(maps, dim=0)  # (N,1,H,W)

    def loss_by_feat(self, seg_logits, batch_data_samples):
        # 1) normal segformer losses
        losses = super().loss_by_feat(seg_logits, batch_data_samples)

        # 2) morphological regularization needs extra maps
        N, _, H, W = seg_logits.shape
        device = seg_logits.device

        fg_prob = self._fg_prob(seg_logits)

        # gt (0/1)
        gt = self._stack_map(batch_data_samples, "gt_sem_seg", device, (H, W))
        if gt is None:
            return losses
        gt_fg = (gt > 0.5).float()
        gt_bg = 1.0 - gt_fg

        edge = self._stack_map(batch_data_samples, "gt_edge_dist", device, (H, W))
        mdist = self._stack_map(batch_data_samples, "gt_mask_dist", device, (H, W))

        # ---- C: MorphTube / center consistency ----
        tube_loss = None
        if edge is not None and self.lambda_tube > 0:
            eps = 1e-6

            if self.use_fixed_tube:
                # tube = eroded foreground region: edge_dist >= tau_fixed
                tube = (gt_fg > 0.5) * (edge >= self.tau_fixed).float()
                denom = tube.sum().clamp_min(1.0)
                # if tube too small (thin region), fallback to weighted center loss
                if denom.item() < 32:
                    tube = None
                else:
                    tube_loss = ((1.0 - fg_prob) * tube).sum() / (denom + eps)

            if tube_loss is None and self.use_adaptive_center:
                # adaptive center weighting in GT foreground:
                # w_center = (edge / max(edge))^p  (per-image normalization)
                # This avoids fixed tau and adapts to width variations.
                # Compute per-sample max over GT foreground
                edge_fg = edge * gt_fg
                # max over spatial for each sample
                mx = edge_fg.flatten(2).max(dim=2).values.view(N, 1, 1, 1).clamp_min(eps)
                w = (edge_fg / mx).clamp(0.0, 1.0) ** max(self.center_power, 1e-6)
                denom = (w.sum()).clamp_min(1.0)
                tube_loss = ((1.0 - fg_prob) * w).sum() / denom

        if tube_loss is not None:
            losses["loss_morph_tube"] = tube_loss * self.lambda_tube

        # ---- B: FP outside (distance-weighted) ----
        if mdist is not None and self.lambda_fp > 0:
            # weight grows with distance from GT (only meaningful on background)
            w_far = torch.clamp(mdist / max(self.tau_fp, 1e-6), 0.0, 1.0)
            bg_far = gt_bg * w_far
            denom = bg_far.sum().clamp_min(1.0)
            fp_loss = (fg_prob * bg_far).sum() / denom
            losses["loss_morph_fp"] = fp_loss * self.lambda_fp

        return losses
