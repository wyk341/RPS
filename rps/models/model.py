import copy
import torch
from torch import nn
from rps.augment.features import FeatureJitter, adain_jitter
from .backbone import MY_TableResNetExtra
from .position import my_PositionalEncoding
from .decoder import my_momodecoder
from .convertor import TableMasterConvertor
from rps.runtime import Config


class RPSModel(nn.Module):
    """Table structure recognizer with autoregressive and teacher-forced paths."""

    def __init__(self, backbone, encoder, decoder, label_convertor, max_seq_len=400, **kwargs):
        super().__init__()
        convertor_cfg=copy.deepcopy(dict(label_convertor))
        convertor_cfg.pop('type',None)
        convertor_cfg['max_seq_len']=max_seq_len
        self.label_convertor=TableMasterConvertor(**convertor_cfg)
        b=copy.deepcopy(dict(backbone)); b.pop('type',None)
        self.backbone=MY_TableResNetExtra(**b)
        e=copy.deepcopy(dict(encoder)); e.pop('type',None)
        self.encoder=my_PositionalEncoding(**e)
        d=copy.deepcopy(dict(decoder)); d.pop('type',None)
        cv=self.label_convertor
        d.update(num_classes=cv.num_classes(),start_idx=cv.start_idx,end_idx=cv.end_idx,
                 padding_idx=cv.padding_idx,max_seq_len=max_seq_len,
                 num_classes_cell=cv.num_classes_cell(),start_idx_cell=cv.start_idx_cell,
                 end_idx_cell=cv.end_idx_cell,padding_idx_cell=cv.padding_idx_cell,
                 max_seq_len_cell=cv.max_seq_len_cell,idx_tag_cell=cv.idx_tag_cell())
        self.decoder=my_momodecoder(**Config(d))

    def forward(self, img, img_metas, **kwargs):
        return self.forward_train(img,img_metas,**kwargs)


    def extract_feat(self, img):
        feats = self.backbone(img)
        return feats

    @torch.no_grad()
    def _truncate_after_eos_pad(self, ids: torch.Tensor, eos_id: int,
        pad_id: int) ->torch.Tensor:
        if eos_id is None or eos_id < 0:
            return ids
        eos_mask = ids == eos_id
        after_or_at = eos_mask.cumsum(dim=1) >= 1
        after = after_or_at & ~eos_mask
        out = ids.clone()
        out[after] = pad_id
        return out

    def forward_train(self, img, img_metas, collect_va=False, strong_view=
        False, train_mode=True, targets_dict=None):
        feat_list = self.extract_feat(img)
        feat = feat_list[-1]
        if strong_view:
            feat = FeatureJitter(sigma=0.08, scale=0.1, dropblock_p=0.05,
                block_size=7)(feat)
            feat = adain_jitter(feat, m_scale=0.03, s_scale=0.03)
        out_enc = self.encoder(feat) if self.encoder is not None else None
        out_dec, out_bbox, out_cell = self.decoder(feat, out_enc,
            collect_va=collect_va, targets_dict=targets_dict, img_metas=
            img_metas, train_mode=train_mode)
        t_ids = None
        if targets_dict is None:
            if hasattr(self.decoder, '_last_t_ids'
                ) and self.decoder._last_t_ids is not None:
                t_ids = self.decoder._last_t_ids
            elif torch.is_tensor(out_dec) and out_dec.dim() == 3:
                t_ids = torch.argmax(out_dec, dim=-1)
            out_bbox = None
            out_cell = None
        return dict(out_dec=out_dec, out_bbox=out_bbox, out_cell=out_cell,
            t_ids=t_ids, targets=targets_dict, backbone_feat=feat.detach())
