import torch
import torch.nn as nn
import torch.nn.functional as F

from mmseg.registry import MODELS


@MODELS.register_module()
class SoftCLDiceLoss(nn.Module):
    """Differentiable clDice loss for thin, connected structures.

    Works for both:
    - multi-class logits: [N,C,H,W] with class index labels [N,H,W]
    - binary logits: [N,1,H,W] with labels [N,H,W] or [N,1,H,W]
    """

    def __init__(self,
                 iterations=10,
                 smooth=1.0,
                 loss_weight=1.0,
                 foreground_index=1,
                 from_logits=True,
                 loss_name='loss_cldice'):
        super().__init__()
        self.iterations = int(iterations)
        self.smooth = float(smooth)
        self.loss_weight = float(loss_weight)
        self.foreground_index = int(foreground_index)
        self.from_logits = bool(from_logits)
        self._loss_name = str(loss_name)

    @property
    def loss_name(self):
        return self._loss_name

    @staticmethod
    def _soft_erode(x):
        p1 = -F.max_pool2d(-x, kernel_size=(3, 1), stride=1, padding=(1, 0))
        p2 = -F.max_pool2d(-x, kernel_size=(1, 3), stride=1, padding=(0, 1))
        return torch.minimum(p1, p2)

    @staticmethod
    def _soft_dilate(x):
        return F.max_pool2d(x, kernel_size=3, stride=1, padding=1)

    @classmethod
    def _soft_open(cls, x):
        return cls._soft_dilate(cls._soft_erode(x))

    @classmethod
    def _soft_skeletonize(cls, x, iterations):
        # Differentiable skeleton approximation from iterative soft morphology.
        x = x.clamp(0.0, 1.0)
        opened = cls._soft_open(x)
        skel = F.relu(x - opened)
        for _ in range(max(0, iterations - 1)):
            x = cls._soft_erode(x)
            opened = cls._soft_open(x)
            delta = F.relu(x - opened)
            skel = skel + F.relu(delta - skel * delta)
        return skel

    def _extract_foreground_prob(self, pred):
        if pred.dim() != 4:
            raise ValueError(f"pred must be 4D [N,C,H,W], got shape={tuple(pred.shape)}")

        if pred.size(1) == 1:
            if self.from_logits:
                return torch.sigmoid(pred)
            return pred.clamp(0.0, 1.0)

        if self.foreground_index >= pred.size(1):
            raise ValueError(
                f"foreground_index={self.foreground_index} out of range for channels={pred.size(1)}")

        if self.from_logits:
            prob = torch.softmax(pred, dim=1)
        else:
            prob = pred
        return prob[:, self.foreground_index:self.foreground_index + 1]

    @staticmethod
    def _to_binary_target(target, ignore_index=255):
        if target.dim() == 4 and target.size(1) == 1:
            target = target[:, 0]
        if target.dim() != 3:
            raise ValueError(f"target must be [N,H,W] or [N,1,H,W], got shape={tuple(target.shape)}")
        valid_mask = (target != ignore_index).float().unsqueeze(1)
        target = (target > 0).float().unsqueeze(1)
        return target, valid_mask

    def forward(self,
                pred,
                target,
                weight=None,
                avg_factor=None,
                reduction_override=None,
                ignore_index=255,
                **kwargs):
        del weight, avg_factor, reduction_override
        del kwargs

        pred_fg = self._extract_foreground_prob(pred)
        target_fg, valid_mask = self._to_binary_target(target, ignore_index=ignore_index)
        target_fg = target_fg.to(pred_fg.device)
        valid_mask = valid_mask.to(pred_fg.device)

        if pred_fg.shape[-2:] != target_fg.shape[-2:]:
            target_fg = F.interpolate(target_fg, size=pred_fg.shape[-2:], mode='nearest')
            valid_mask = F.interpolate(valid_mask, size=pred_fg.shape[-2:], mode='nearest')

        pred_fg = pred_fg * valid_mask
        target_fg = target_fg * valid_mask

        skel_pred = self._soft_skeletonize(pred_fg, self.iterations)
        skel_gt = self._soft_skeletonize(target_fg, self.iterations)

        tprec = (skel_pred * target_fg).sum(dim=(1, 2, 3))
        tprec = (tprec + self.smooth) / (skel_pred.sum(dim=(1, 2, 3)) + self.smooth)

        tsens = (skel_gt * pred_fg).sum(dim=(1, 2, 3))
        tsens = (tsens + self.smooth) / (skel_gt.sum(dim=(1, 2, 3)) + self.smooth)

        cl_dice = (2.0 * tprec * tsens + self.smooth) / (tprec + tsens + self.smooth)
        loss = 1.0 - cl_dice
        return self.loss_weight * loss.mean()
