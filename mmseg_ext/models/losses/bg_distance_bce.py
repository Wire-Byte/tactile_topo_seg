import torch
import torch.nn as nn
import torch.nn.functional as F
from mmseg.registry import MODELS


@MODELS.register_module()
class BackgroundDistanceBCE(nn.Module):
    """
    Background-only, distance-aware BCE (FP suppression).

    Enhancements:
      - p^alpha gating (default detach): focus on suppressing confident false positives
      - smooth w_dist via smoothstep to avoid over-sharp weighting
      - warmup on loss_weight using internal step counter

    Inputs:
      seg_logits: (B, 2, H, W)
      gt_semantic_seg: (B, 1, H, W) values {0,255}
      gt_mask_dist: (B, 1, H, W) float, 0 in FG, >=1 in BG, clipped to max_dist(64)

    Weight design:
      is_bg = (gt==0)
      d = clamp(gt_mask_dist - r0, >=0)
      x = clamp(d/tau, [0,1])
      w_dist = smoothstep(x) ^ gamma
      gate = (p_fg)^alpha  (optionally detach)
      w = is_bg * w_dist * gate
    """

    def __init__(self,
                 # distance weight
                 tau=64.0,
                 gamma=1.0,
                 r0=8.0,
                 smooth_w=True,

                 # p^alpha gating
                 alpha=2.0,
                 gate_detach=True,

                 # loss weight + warmup
                 loss_weight=0.05,
                 warmup_iters=1000,
                 warmup_start=0.0,   # 0 -> from 0 to 1, 0.2 -> from 0.2 to 1
                 eps=1e-6,
                 loss_name='loss_bg_dist_bce'):
        super().__init__()
        self.tau = float(tau)
        self.gamma = float(gamma)
        self.r0 = float(r0)
        self.smooth_w = bool(smooth_w)

        self.alpha = float(alpha)
        self.gate_detach = bool(gate_detach)

        self.loss_weight = float(loss_weight)
        self.warmup_iters = int(warmup_iters)
        self.warmup_start = float(warmup_start)

        self.eps = float(eps)
        self.loss_name = loss_name

        # internal step counter for warmup (no runner dependency)
        self.register_buffer("_step", torch.zeros((), dtype=torch.long), persistent=False)

    @staticmethod
    def _smoothstep(x: torch.Tensor) -> torch.Tensor:
        # x in [0,1] -> smooth in [0,1], zero slope at endpoints
        return x * x * (3.0 - 2.0 * x)

    def _warmup_scale(self) -> float:
        if self.warmup_iters <= 0:
            return 1.0
        s = float(self._step.item())
        t = min(1.0, s / float(self.warmup_iters))
        # linear from warmup_start -> 1
        return self.warmup_start + (1.0 - self.warmup_start) * t

    def forward(self, seg_logits, gt_semantic_seg, gt_mask_dist):
        # update step only during training
        if self.training:
            self._step += 1

        # foreground logit (class 1)
        fg_logits = seg_logits[:, 1:2, :, :]  # (B,1,H,W)

        # resize gt to logits size
        if gt_semantic_seg.shape[-2:] != fg_logits.shape[-2:]:
            gt_semantic_seg = F.interpolate(
                gt_semantic_seg.float(), size=fg_logits.shape[-2:], mode='nearest'
            )
        if gt_mask_dist.shape[-2:] != fg_logits.shape[-2:]:
            gt_mask_dist = F.interpolate(
                gt_mask_dist.float(), size=fg_logits.shape[-2:], mode='nearest'
            )

        # GT is 0/255
        is_bg = (gt_semantic_seg <= 0.5).float()  # (B,1,H,W)

        # protect band: dist <= r0 => zero weight
        d = (gt_mask_dist - self.r0).clamp(min=0.0)

        # normalized distance in [0,1]
        x = (d / max(self.tau, self.eps)).clamp(0.0, 1.0)

        # smooth w_dist
        if self.smooth_w:
            w_dist = self._smoothstep(x)
        else:
            w_dist = x

        if self.gamma != 1.0:
            w_dist = w_dist.pow(self.gamma)

        # p^alpha gating (use sigmoid prob of FG channel)
        p = torch.sigmoid(fg_logits)
        if self.gate_detach:
            p = p.detach()
        gate = p.clamp(min=0.0, max=1.0).pow(self.alpha)

        # final weight (only background)
        w = is_bg * w_dist * gate

        # BCE pushing FG-logits down on background pixels (target=0)
        raw = F.binary_cross_entropy_with_logits(
            fg_logits, torch.zeros_like(fg_logits), reduction='none'
        )

        num = (raw * w).sum()
        den = w.sum().clamp_min(self.eps)

        warm = self._warmup_scale()
        return (self.loss_weight * warm) * (num / den)
