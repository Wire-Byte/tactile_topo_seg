import torch
import torch.nn.functional as F
from mmseg.registry import MODELS
from mmseg.models import EncoderDecoder


@MODELS.register_module()
class TPEncoderDecoderThick(EncoderDecoder):
    """EncoderDecoder + thickness regularization loss.

    Thick loss (fixed):
      - Only penalize foreground probability on GT background (false positives),
        weighted by distance-to-skeleton map (farther -> higher penalty).
      - Normalize by number of background pixels to keep scale stable.
    """

    def __init__(self, lambda_thick: float = 0.05, thick_tau: float = 8.0, **kwargs):
        super().__init__(**kwargs)
        self.lambda_thick = float(lambda_thick)
        self.thick_tau = float(thick_tau)

    @staticmethod
    def _fg_prob(seg_logits: torch.Tensor) -> torch.Tensor:
        """Get foreground probability map (N,1,H,W)."""
        if seg_logits.size(1) == 1:
            return torch.sigmoid(seg_logits)
        probs = F.softmax(seg_logits, dim=1)
        # Binary task: MapTPLabels maps 255->1, so class-1 is tactile paving foreground
        return probs[:, 1:2, ...]

    def _thick_loss(self, fg_prob: torch.Tensor, data_samples) -> torch.Tensor:
        """Penalize false-positive foreground far from skeleton (GT background only).

        Args:
            fg_prob: (N,1,H,W) foreground probability from decode head logits.
            data_samples: list of SegDataSample, each contains:
                - gt_skel_dist: distance-to-skeleton map (in sample space)
                - gt_sem_seg: GT segmentation map (0/1 after MapTPLabels)
        """
        N, _, H, W = fg_prob.shape
        dist_list = []
        gt_list = []

        for ds in data_samples:
            # ---- dist map ----
            dist = ds.gt_skel_dist.data  # (1,h,w) or (h,w)
            if dist.dim() == 2:
                dist = dist.unsqueeze(0)  # (1,h,w)
            dist = dist.unsqueeze(0).float().to(fg_prob.device)  # (1,1,h,w)

            if dist.shape[-2:] != (H, W):
                dist = F.interpolate(dist, size=(H, W), mode="nearest")
            dist_list.append(dist)

            # ---- GT seg ----
            gt = ds.gt_sem_seg.data  # (1,h,w) typically
            if gt.dim() == 2:
                gt = gt.unsqueeze(0)  # (1,h,w)
            gt = gt.unsqueeze(0).float().to(fg_prob.device)  # (1,1,h,w)

            if gt.shape[-2:] != (H, W):
                gt = F.interpolate(gt, size=(H, W), mode="nearest")
            gt_list.append(gt)

        dist = torch.cat(dist_list, dim=0)  # (N,1,H,W)
        gt = torch.cat(gt_list, dim=0)      # (N,1,H,W), values should be 0/1

        # Weight: farther from skeleton -> larger penalty (clamped)
        w = torch.clamp(dist / max(self.thick_tau, 1e-6), 0.0, 1.0)

        # Only penalize on GT background (false positive suppression)
        bg = (gt < 0.5).float()  # (N,1,H,W)

        # Normalize by bg pixels to keep loss scale stable
        denom = bg.sum().clamp_min(1.0)
        loss = (fg_prob * w * bg).sum() / denom
        return loss

    def loss(self, inputs, data_samples):
        # 1) Original losses (seg loss + auxiliary skeleton head losses)
        losses = super().loss(inputs, data_samples)

        # 2) Extra thick loss computed from decode head logits
        feats = self.extract_feat(inputs)
        seg_logits = self.decode_head(feats)  # (N,C,H,W)
        fg_prob = self._fg_prob(seg_logits)

        loss_thick = self._thick_loss(fg_prob, data_samples)
        losses["loss_thick"] = loss_thick * self.lambda_thick
        return losses
