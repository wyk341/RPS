# Adapted from MTL-TabNet, TableMASTER, and OpenMMLab components.
# Changes: rps package imports and framework-independent integration.
# Upstream notices: THIRD_PARTY_NOTICES.md and licenses/.
from torch import nn

class CELoss(nn.Module):

    def __init__(self, ignore_index=-1, reduction='none'):
        super().__init__()
        assert isinstance(ignore_index, int)
        assert isinstance(reduction, str)
        assert reduction in ['none', 'mean', 'sum']
        self.loss_ce = nn.CrossEntropyLoss(ignore_index=ignore_index,
            reduction=reduction)

    def format(self, outputs, targets_dict):
        targets = targets_dict['padded_targets']
        return outputs.permute(0, 2, 1).contiguous(), targets

    def forward(self, outputs, targets_dict, img_metas=None):
        outputs, targets = self.format(outputs, targets_dict)
        loss_ce = self.loss_ce(outputs, targets.to(outputs.device))
        losses = dict(loss_ce=loss_ce)
        return losses


class SARLoss(CELoss):

    def __init__(self, ignore_index=0, reduction='mean', **kwargs):
        super().__init__(ignore_index, reduction)

    def format(self, outputs, targets_dict):
        targets = targets_dict['padded_targets']
        targets = targets[:, 1:].contiguous()
        outputs = outputs[:, :-1, :].permute(0, 2, 1).contiguous()
        return outputs, targets


class TFLoss(CELoss):

    def __init__(self, ignore_index=-1, reduction='none', flatten=True, **
        kwargs):
        super().__init__(ignore_index, reduction)
        assert isinstance(flatten, bool)
        self.flatten = flatten

    def format(self, outputs, targets_dict):
        outputs = outputs[:, :-1, :].contiguous()
        targets = targets_dict['padded_targets']
        targets = targets[:, 1:].contiguous()
        if self.flatten:
            outputs = outputs.view(-1, outputs.size(-1))
            targets = targets.view(-1)
        else:
            outputs = outputs.permute(0, 2, 1).contiguous()
        return outputs, targets


class MASTERTFLoss(CELoss):

    def __init__(self, ignore_index=-1, reduction='none', flatten=True, **
        kwargs):
        super().__init__(ignore_index, reduction)
        assert isinstance(flatten, bool)
        self.flatten = flatten

    def format(self, outputs, targets_dict):
        targets = targets_dict['padded_targets']
        targets = targets[:, 1:].contiguous()
        if self.flatten:
            outputs = outputs.view(-1, outputs.size(-1))
            targets = targets.view(-1)
        else:
            outputs = outputs.permute(0, 2, 1).contiguous()
        return outputs, targets


class MASTERCELLLoss(CELoss):

    def __init__(self, ignore_index=-1, reduction='none', flatten=True, **
        kwargs):
        super().__init__(ignore_index, reduction)
        assert isinstance(flatten, bool)
        self.flatten = flatten

    def format(self, outputs, targets_dict):
        targets = targets_dict['padded_targets']
        targets = targets[:, 1:].contiguous()
        if self.flatten:
            outputs = outputs.view(-1, outputs.size(-1))
            targets = targets.view(-1)
        else:
            outputs = outputs.permute(0, 2, 1).contiguous()
        return outputs, targets

    def forward(self, outputs, targets_dict, img_metas=None):
        outputs, targets = self.format(outputs, targets_dict)
        loss_ce = self.loss_ce(outputs, targets.to(outputs.device))
        return loss_ce
