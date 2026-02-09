import torch
import torch.nn.functional as F
from mmseg.registry import MODELS
from mmseg.models import EncoderDecoder


@MODELS.register_module()
class TPEncoderDecoderThick(EncoderDecoder):
    """EncoderDecoder + thickness regularization loss."""

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
        # 你的任务是 2 类，MapTPLabels 把 255->1，所以 class-1 是盲道前景
        return probs[:, 1:2, ...]

    def _thick_loss(self, fg_prob, data_samples):
        """Thickness regularization (Scheme B).

        dist map semantic (after your fixed generator):
            - background: 0
            - inside foreground: larger near skeleton, smaller near boundary

        Scheme B: penalize foreground probability far from skeleton (near boundary),
        and only inside GT foreground region.
        """
        _, _, H, W = fg_prob.shape
        dist_list = []
        fgmask_list = []

        for ds in data_samples:
            # dist: (h,w) or (1,h,w)
            dist = ds.gt_skel_dist.data
            if dist.dim() == 2:
                dist = dist.unsqueeze(0)  # (1,h,w)
            dist = dist.unsqueeze(0).float().to(fg_prob.device)  # (1,1,h,w)
            if dist.shape[-2:] != (H, W):
                dist = F.interpolate(dist, size=(H, W), mode="nearest")
            dist_list.append(dist)

            # fgmask from gt_sem_seg: (h,w) or (1,h,w)
            gt = ds.gt_sem_seg.data
            if gt.dim() == 2:
                gt = gt.unsqueeze(0)  # (1,h,w)
            gt = gt.unsqueeze(0).float().to(fg_prob.device)  # (1,1,h,w)
            if gt.shape[-2:] != (H, W):
                gt = F.interpolate(gt, size=(H, W), mode="nearest")
            fgmask_list.append((gt > 0).float())

        dist = torch.cat(dist_list, dim=0)      # (N,1,H,W)
        fgmask = torch.cat(fgmask_list, dim=0)  # (N,1,H,W)

        # normalized weight near skeleton
        w = torch.clamp(dist / max(self.thick_tau, 1e-6), 0.0, 1.0)

        # Scheme B: penalize fg probability far from skeleton (near boundary)
        w_far = 1.0 - w
        loss_map = fg_prob * w_far * fgmask
        denom = fgmask.sum().clamp_min(1.0)
        return loss_map.sum() / denom

    def loss(self, inputs, data_samples):
        # 1) 原有 loss（包含 decode_head 的 seg loss + auxiliary skeleton head loss）
        losses = super().loss(inputs, data_samples)

        # 2) 额外计算 thick loss（用 decode_head 的 logits）
        feats = self.extract_feat(inputs)
        seg_logits = self.decode_head(feats)  # (N,C,H,W)
        fg_prob = self._fg_prob(seg_logits)
        loss_thick = self._thick_loss(fg_prob, data_samples)

        losses["loss_thick"] = loss_thick * self.lambda_thick
        return losses
