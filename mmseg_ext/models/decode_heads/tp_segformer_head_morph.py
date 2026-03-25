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
                 # weights
                 lambda_tube: float = 0.01,
                 lambda_fp: float = 0.05,

                 # tube mode:
                 #   "quantile": per-sample quantile tube (recommended, stable coverage)
                 #   "tau":      fixed tau tube (NOT recommended unless tau is huge)
                 #   "weight":   soft center weighting (edge normalized^p), no hard tube
                 tube_mode: str = "quantile",

                 # quantile tube
                 tube_ratio: float = 0.2,        # keep top 20% most-central pixels within FG
                 tube_min_pixels: int = 256,     # if tube too small -> fallback

                 # fixed tau tube (legacy)
                 tau_fixed: float = 32.0,

                 # soft center weighting (fallback / alternative)
                 center_power: float = 2.0,

                 # FP outside weighting (distance-to-foreground outside GT)
                 tau_fp: float = 24.0,
                 **kwargs):
        super().__init__(**kwargs)

        self.lambda_tube = float(lambda_tube)
        self.lambda_fp = float(lambda_fp)

        self.tube_mode = str(tube_mode)
        assert self.tube_mode in ("quantile", "tau", "weight"), \
            f"tube_mode must be in ['quantile','tau','weight'], got {self.tube_mode}"

        self.tube_ratio = float(tube_ratio)
        self.tube_min_pixels = int(tube_min_pixels)

        self.tau_fixed = float(tau_fixed)
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

    def _build_quantile_tube(self, edge: torch.Tensor, gt_fg: torch.Tensor):
        """Per-sample quantile tube: take top tube_ratio of edge within gt_fg."""
        # edge, gt_fg: (N,1,H,W)
        N = edge.shape[0]
        tube = torch.zeros_like(gt_fg)

        # take most-central pixels => large edge_dist
        keep = float(self.tube_ratio)
        keep = min(max(keep, 1e-3), 0.999)

        for n in range(N):
            fg = (gt_fg[n, 0] > 0.5)
            if fg.sum().item() == 0:
                continue
            vals = edge[n, 0][fg]  # 1D
            # If edge is all zeros in fg, tube will be empty -> fallback later
            # threshold so that approx keep% is selected: edge >= q_{1-keep}
            q = torch.quantile(vals, 1.0 - keep)
            t = fg & (edge[n, 0] >= q)
            if t.sum().item() >= self.tube_min_pixels:
                tube[n, 0] = t.float()
            # else: leave as zeros -> will trigger fallback

        return tube

    def _build_tau_tube(self, edge: torch.Tensor, gt_fg: torch.Tensor):
        """Fixed tau tube: edge >= tau_fixed within fg."""
        tube = (gt_fg > 0.5) * (edge >= self.tau_fixed).float()
        return tube

    def _center_weight(self, edge: torch.Tensor, gt_fg: torch.Tensor):
        """Soft center weighting: w = (edge/max(edge_in_fg))^p within fg."""
        eps = 1e-6
        N = edge.shape[0]
        edge_fg = edge * gt_fg
        mx = edge_fg.flatten(2).max(dim=2).values.view(N, 1, 1, 1).clamp_min(eps)
        w = (edge_fg / mx).clamp(0.0, 1.0) ** max(self.center_power, 1e-6)
        return w

    def loss_by_feat(self, seg_logits, batch_data_samples):
        # 1) normal segformer losses
        losses = super().loss_by_feat(seg_logits, batch_data_samples)

        # 2) morphological regularization needs extra maps
        N, _, H, W = seg_logits.shape
        device = seg_logits.device
        eps = 1e-6

        fg_prob = self._fg_prob(seg_logits)

        # gt (0/1)
        gt = self._stack_map(batch_data_samples, "gt_sem_seg", device, (H, W))
        if gt is None:
            return losses
        gt_fg = (gt > 0.5).float()
        gt_bg = 1.0 - gt_fg

        edge = self._stack_map(batch_data_samples, "gt_edge_dist", device, (H, W))
        mdist = self._stack_map(batch_data_samples, "gt_mask_dist", device, (H, W))

        # ---- C: tube / center consistency (ONLY inside GT foreground) ----
        if edge is not None and self.lambda_tube > 0:
            tube_loss = None

            if self.tube_mode == "quantile":
                tube = self._build_quantile_tube(edge, gt_fg)
                denom = tube.sum().clamp_min(1.0)
                if denom.item() >= self.tube_min_pixels:
                    tube_loss = ((1.0 - fg_prob) * tube).sum() / (denom + eps)
                else:
                    # fallback
                    w = self._center_weight(edge, gt_fg)
                    denom = w.sum().clamp_min(1.0)
                    tube_loss = ((1.0 - fg_prob) * w).sum() / (denom + eps)

            elif self.tube_mode == "tau":
                tube = self._build_tau_tube(edge, gt_fg)
                denom = tube.sum().clamp_min(1.0)
                if denom.item() >= self.tube_min_pixels:
                    tube_loss = ((1.0 - fg_prob) * tube).sum() / (denom + eps)
                else:
                    w = self._center_weight(edge, gt_fg)
                    denom = w.sum().clamp_min(1.0)
                    tube_loss = ((1.0 - fg_prob) * w).sum() / (denom + eps)

            elif self.tube_mode == "weight":
                w = self._center_weight(edge, gt_fg)
                denom = w.sum().clamp_min(1.0)
                tube_loss = ((1.0 - fg_prob) * w).sum() / (denom + eps)

            if tube_loss is not None:
                losses["loss_morph_tube"] = tube_loss * self.lambda_tube

        # ---- small B: FP outside (distance-weighted, only on GT background) ----
        if mdist is not None and self.lambda_fp > 0:
            # mdist: outside-GT dist-to-foreground, 0 inside (by your own comment)
            # penalize predicting fg far away from GT
            w_far = torch.clamp(mdist / max(self.tau_fp, 1e-6), 0.0, 1.0)
            bg_far = gt_bg * w_far
            denom = bg_far.sum().clamp_min(1.0)
            fp_loss = (fg_prob * bg_far).sum() / denom
            losses["loss_morph_fp"] = fp_loss * self.lambda_fp

        return losses
