import torch
import torch.nn as nn
import torch.nn.functional as F
from mmseg.registry import MODELS


@MODELS.register_module()
class AntiIslandHardOpeningLoss(nn.Module):
    """
    Anti-island loss (hard, mask-based):
      1) bin_fg = (p_fg > thr)
      2) opened = morphological opening(bin_fg)
      3) island = bin_fg & (~opened)   # removed by opening => small islands / thin spurs
      4) only penalize islands that are in BG and (optionally) far from GT boundary via gt_mask_dist
      5) loss = mean( p_fg^(1+alpha) ) over island pixels (focus on confident islands)

    Requires:
      - gt_semantic_seg (0/255)
      - gt_mask_dist (distance-to-FG, clipped, 0 in FG)
    """

    def __init__(self,
                 kernel_size=3,
                 thr=0.3,
                 r_far=12.0,

                 # p^alpha focus
                 alpha=2.0,
                 gate_detach=False,  # for island loss, allowing grad through p is fine

                 loss_weight=0.02,
                 eps=1e-6,
                 loss_name='loss_anti_island'):
        super().__init__()
        self.kernel_size = int(kernel_size)
        self.thr = float(thr)
        self.r_far = float(r_far)

        self.alpha = float(alpha)
        self.gate_detach = bool(gate_detach)

        self.loss_weight = float(loss_weight)
        self.eps = float(eps)
        self.loss_name = loss_name
        self.pad = self.kernel_size // 2

    @torch.no_grad()
    def _hard_opening(self, bin_fg: torch.Tensor) -> torch.Tensor:
        # bin_fg: (B,1,H,W) {0,1}
        # erode via minpool = -maxpool(-x)
        er = -F.max_pool2d(-bin_fg, self.kernel_size, stride=1, padding=self.pad)
        op = F.max_pool2d(er, self.kernel_size, stride=1, padding=self.pad)
        return op

    def forward(self, seg_logits, gt_semantic_seg, gt_mask_dist=None):
        # FG prob
        probs = F.softmax(seg_logits, dim=1)[:, 1:2, :, :]  # (B,1,H,W)

        # align GT
        if gt_semantic_seg.shape[-2:] != probs.shape[-2:]:
            gt_semantic_seg = F.interpolate(
                gt_semantic_seg.float(), size=probs.shape[-2:], mode='nearest'
            )
        is_bg = (gt_semantic_seg <= 0.5)  # bool

        # distance gate (recommended)
        if gt_mask_dist is not None:
            if gt_mask_dist.shape[-2:] != probs.shape[-2:]:
                gt_mask_dist = F.interpolate(
                    gt_mask_dist.float(), size=probs.shape[-2:], mode='nearest'
                )
            far_bg = (gt_mask_dist > self.r_far)
        else:
            far_bg = torch.ones_like(probs, dtype=torch.bool)

        # hard island mask (no-grad)
        with torch.no_grad():
            bin_fg = (probs > self.thr).float()
            opened = self._hard_opening(bin_fg)
            island = (bin_fg > 0.5) & (opened < 0.5)
            island_mask = island & is_bg & far_bg  # bool

        if island_mask.sum() == 0:
            return probs.sum() * 0.0

        p = probs
        if self.gate_detach:
            p = p.detach()

        # p^alpha focusing: penalize confident islands more
        # mean(p^(1+alpha)) over island pixels
        loss = p[island_mask].clamp(min=0.0, max=1.0).pow(1.0 + self.alpha).mean()
        return self.loss_weight * loss
