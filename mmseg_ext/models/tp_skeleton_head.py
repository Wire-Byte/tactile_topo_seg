import torch
import torch.nn as nn
import torch.nn.functional as F
from mmseg.registry import MODELS
from mmseg.models.decode_heads.decode_head import BaseDecodeHead

@MODELS.register_module()
class TPSkeletonHead(BaseDecodeHead):
    """Binary skeleton auxiliary head."""

    def __init__(self,
                 in_channels,
                 channels=64,
                 loss_bce=dict(type='CrossEntropyLoss', use_sigmoid=True, loss_weight=1.0),
                 loss_dice=dict(type='DiceLoss', use_sigmoid=True, loss_weight=1.0),
                 **kwargs):
        super().__init__(
            in_channels=in_channels,
            channels=channels,
            num_classes=1,
            **kwargs
        )
        self.conv1 = nn.Conv2d(self.in_channels, channels, 3, padding=1)
        self.conv2 = nn.Conv2d(channels, 1, 1)

        # 用 MODELS.build（你已经改对了）
        self.loss_bce = MODELS.build(loss_bce)
        self.loss_dice = MODELS.build(loss_dice)

    def forward(self, inputs):
        x = self._transform_inputs(inputs)  # -> Tensor [N,C,H,W]
        x = F.relu(self.conv1(x))
        return self.conv2(x)  # [N,1,H,W]

    def loss(self, inputs, batch_data_samples, train_cfg=None):
        # pred: [N,1,h,w]
        pred = self.forward(inputs)

        # 目标尺寸：用 pred 的空间尺寸做对齐基准
        tgt_h, tgt_w = pred.shape[-2], pred.shape[-1]

        gts = []
        for ds in batch_data_samples:
            if not hasattr(ds, 'gt_skeleton'):
                raise AttributeError("data_samples missing gt_skeleton. Check PackSegInputsWithSkeleton.")
            gt = ds.gt_skeleton.data  # 可能是 [1,H,W] 或 [H,W]
            if gt.dim() == 3:
                gt = gt.squeeze(0)     # -> [H,W]
            gt = gt.long()             # Dice(one_hot) 需要 LongTensor

            # 对齐到 pred 的空间尺寸（用 nearest，别用 bilinear）
            if gt.shape[-2] != tgt_h or gt.shape[-1] != tgt_w:
                gt = F.interpolate(
                    gt[None, None].float(),  # [1,1,H,W]
                    size=(tgt_h, tgt_w),
                    mode='nearest'
                )[0, 0].long()               # -> [h,w]

            gts.append(gt)

        gt = torch.stack(gts, dim=0)  # [N,h,w] long

        # BCE(use_sigmoid=True) 这类实现通常要求 label [N,h,w]
        loss_bce = self.loss_bce(pred, gt)
        loss_dice = self.loss_dice(pred, gt)

        return dict(loss_skel_bce=loss_bce, loss_skel_dice=loss_dice)


    def predict(self, inputs, batch_img_metas, test_cfg):
        return self.forward(inputs)
