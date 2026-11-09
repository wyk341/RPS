# Adapted from MTL-TabNet, TableMASTER, and OpenMMLab components.
# Changes: rps package imports and framework-independent integration.
# Upstream notices: THIRD_PARTY_NOTICES.md and licenses/.
from torch import nn

class DistLoss(nn.Module):

    def __init__(self, reduction='none'):
        super().__init__()
        assert isinstance(reduction, str)
        assert reduction in ['none', 'mean', 'sum']
        self.dist_loss = self.build_loss(reduction)

    def build_loss(self, reduction, **kwargs):
        raise NotImplementedError

    def format(self, outputs, targets_dict):
        raise NotImplementedError

    def forward(self, outputs, targets_dict, img_metas=None):
        outputs, targets = self.format(outputs, targets_dict)
        loss_dist = self.dist_loss(outputs, targets.to(outputs.device))
        losses = dict(loss_dist=loss_dist)
        return losses


class TableL1Loss(DistLoss):

    def __init__(self, reduction='sum', **kwargs):
        super().__init__(reduction)
        self.lambda_horizon = 1.0
        self.lambda_vertical = 1.0
        self.eps = 1e-09
        try:
            assert reduction == 'sum'
        except AssertionError:
            raise 'Table L1 loss in bbox branch should keep reduction is sum.'

    def build_loss(self, reduction):
        return nn.L1Loss(reduction=reduction)

    def format(self, outputs, targets_dict):
        bboxes = targets_dict['bbox'][:, 1:, :].to(outputs.device)
        bbox_masks = targets_dict['bbox_masks'][:, 1:].unsqueeze(-1).to(outputs
            .device)
        masked_outputs = outputs * bbox_masks
        masked_bboxes = bboxes * bbox_masks
        return masked_outputs, masked_bboxes, bbox_masks

    def forward(self, outputs, targets_dict, img_metas=None):
        outputs, targets, bbox_masks = self.format(outputs, targets_dict)
        horizon_sum_loss = self.dist_loss(outputs[:, :, 0::2].contiguous(),
            targets[:, :, 0::2].contiguous())
        horizon_loss = horizon_sum_loss / (bbox_masks.sum() + self.eps)
        vertical_sum_loss = self.dist_loss(outputs[:, :, 1::2].contiguous(),
            targets[:, :, 1::2].contiguous())
        vertical_loss = vertical_sum_loss / (bbox_masks.sum() + self.eps)
        losses = {'horizon_bbox_loss': horizon_loss, 'vertical_bbox_loss':
            vertical_loss}
        return losses
