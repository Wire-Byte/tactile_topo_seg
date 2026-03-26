import torch
import torch.nn as nn
import torch.nn.functional as F

from mmseg.models.decode_heads.decode_head import BaseDecodeHead
from mmseg.registry import MODELS


@MODELS.register_module()
class TPSkeletonHeadClDice(BaseDecodeHead):
    """Binary skeleton auxiliary head with BCE + Dice + clDice losses."""

    def __init__(self,
                 in_channels,
                 channels=64,
                 loss_bce=dict(type='CrossEntropyLoss', use_sigmoid=True, loss_weight=1.0),
                 loss_dice=dict(type='DiceLoss', use_sigmoid=True, loss_weight=1.0),
                 loss_cldice=dict(type='SoftCLDiceLoss', foreground_index=0, loss_weight=0.3, iterations=10),
                 **kwargs):
        super().__init__(
            in_channels=in_channels,
            channels=channels,
            num_classes=1,
            **kwargs)

        self.conv1 = nn.Conv2d(self.in_channels, channels, 3, padding=1)
        self.conv2 = nn.Conv2d(channels, 1, 1)

        self.loss_bce = MODELS.build(loss_bce)
        self.loss_dice = MODELS.build(loss_dice)
        self.loss_cldice = MODELS.build(loss_cldice)

    def forward(self, inputs):
        x = self._transform_inputs(inputs)
        x = F.relu(self.conv1(x))
        return self.conv2(x)

    def loss(self, inputs, batch_data_samples, train_cfg=None):
        del train_cfg
        pred = self.forward(inputs)

        tgt_h, tgt_w = pred.shape[-2], pred.shape[-1]
        gts = []
        for ds in batch_data_samples:
            if not hasattr(ds, 'gt_skeleton'):
                raise AttributeError('data_samples missing gt_skeleton. Check PackSegInputsWithSkeleton.')
            gt = ds.gt_skeleton.data
            if gt.dim() == 3:
                gt = gt.squeeze(0)
            gt = gt.long()

            if gt.shape[-2] != tgt_h or gt.shape[-1] != tgt_w:
                gt = F.interpolate(
                    gt[None, None].float(),
                    size=(tgt_h, tgt_w),
                    mode='nearest')[0, 0].long()
            gts.append(gt)

        gt = torch.stack(gts, dim=0)

        loss_bce = self.loss_bce(pred, gt)
        loss_dice = self.loss_dice(pred, gt)
        loss_cldice = self.loss_cldice(pred, gt)

        return dict(
            loss_skel_bce=loss_bce,
            loss_skel_dice=loss_dice,
            loss_skel_cldice=loss_cldice)

    def predict(self, inputs, batch_img_metas, test_cfg):
        del batch_img_metas, test_cfg
        return self.forward(inputs)
