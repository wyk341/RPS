import torch
import torch.nn.functional as F
from .cross_entropy import MASTERTFLoss, MASTERCELLLoss
from .distance import TableL1Loss

def pack_cell_to_master_format(out_cell_list, cell_targets_list,
    pad_cell_id, device):
    B = min(len(out_cell_list), len(cell_targets_list))
    Tc, Vc = None, None
    Nc_max = 0
    for b in range(B):
        if torch.is_tensor(out_cell_list[b]) and out_cell_list[b].numel() > 0:
            Nc_b, Tc_b, Vc_b = out_cell_list[b].shape
            Tc, Vc = Tc_b, Vc_b
            Nc_max = max(Nc_max, Nc_b)
        if isinstance(cell_targets_list[b], list):
            pass
    if Tc is None:
        return None, None
    outputs_pad = torch.zeros((B, Nc_max, Tc, Vc), device=device)
    targets_pad = torch.full((B, Nc_max, Tc + 1), pad_cell_id, device=
        device, dtype=torch.long)
    for b in range(B):
        logits_b = out_cell_list[b]
        tgt_list = cell_targets_list[b] if isinstance(cell_targets_list[b],
            list) else []
        Nc_logits = int(logits_b.shape[0]) if torch.is_tensor(logits_b
            ) and logits_b.numel() > 0 else 0
        Nc_tgt = len(tgt_list)
        Nc_use = min(Nc_logits, Nc_tgt, Nc_max)
        if Nc_use > 0:
            outputs_pad[(b), :Nc_use] = logits_b[:Nc_use].to(device)
            for i in range(Nc_use):
                t = tgt_list[i].to(device).long()
                L = min(t.numel(), Tc + 1)
                targets_pad[(b), (i), :L] = t[:L]
    outputs_flat = outputs_pad.view(B * Nc_max, Tc, Vc)
    targets_flat = targets_pad.view(B * Nc_max, Tc + 1)
    return outputs_flat, targets_flat


def cell_loss_from_targets(out_cell, cell_padded_targets, pad_cell_id,
    device=None):
    if device is None:
        for x in out_cell:
            if torch.is_tensor(x):
                device = x.device
                break
        if device is None:
            return torch.tensor(0.0)
    if not isinstance(out_cell, list) or len(out_cell) == 0:
        return torch.tensor(0.0, device=device)
    if not isinstance(cell_padded_targets, list) or len(cell_padded_targets
        ) == 0:
        return torch.tensor(0.0, device=device)
    total_loss = torch.zeros((), device=device)
    valid_terms = 0
    B = min(len(out_cell), len(cell_padded_targets))
    for b in range(B):
        logits_b = out_cell[b]
        tgt_list = cell_padded_targets[b]
        if not torch.is_tensor(logits_b) or logits_b.numel() == 0:
            continue
        if not isinstance(tgt_list, list) or len(tgt_list) == 0:
            continue
        Nc_b, Tc, Vc = logits_b.shape
        Nc = min(Nc_b, len(tgt_list))
        for i in range(Nc):
            logit_seq = logits_b[i]
            tgt_seq_all = tgt_list[i].to(device)
            if tgt_seq_all.numel() <= 1:
                continue
            tgt_seq = tgt_seq_all[1:1 + Tc]
            L = tgt_seq.size(0)
            if L == 0:
                continue
            logits_use = logit_seq[:L]
            target_use = tgt_seq
            mask = target_use != pad_cell_id
            if mask.sum() == 0:
                continue
            loss_i = F.cross_entropy(logits_use[mask], target_use[mask],
                reduction='mean')
            total_loss = total_loss + loss_i
            valid_terms += 1
    if valid_terms == 0:
        return torch.tensor(0.0, device=device)
    return total_loss / valid_terms


def loss_sups(sup_out, cfg, enable_bbox=None, enable_cell=None, w_cell=2.0):
    logits_dec = sup_out['out_dec']
    tgt_pack = sup_out['targets']
    device = logits_dec.device
    if enable_bbox is None:
        enable_bbox = bool(getattr(cfg, 'ENABLE_BBOX', True))
    if enable_cell is None:
        enable_cell = bool(getattr(cfg, 'ENABLE_CELL', True))
    struct_targets = {'padded_targets': tgt_pack['padded_targets'].to(
        device, non_blocking=True)}
    out_dec = logits_dec
    tgt_seq = struct_targets['padded_targets']
    Lp, Lt = out_dec.size(1), tgt_seq.size(1)
    if Lp > Lt - 1:
        out_dec = out_dec[:, :Lt - 1, :]
    elif Lp < Lt - 1:
        tgt_seq = tgt_seq[:, :Lp + 1]
    struct_targets['padded_targets'] = tgt_seq
    tf_loss_fn = MASTERTFLoss(ignore_index=cfg.PAD, reduction='mean',
        flatten=True)
    loss_tf_out = tf_loss_fn(out_dec, struct_targets)
    loss_tf = loss_tf_out['loss_ce'] if isinstance(loss_tf_out, dict
        ) else loss_tf_out
    bbox_pred = sup_out.get('out_bbox', None)
    if not enable_bbox or bbox_pred is None:
        loss_bbox = torch.zeros((), device=device)
    else:
        bbox_targets = {'bbox': tgt_pack['bbox'].to(device, non_blocking=
            True), 'bbox_masks': tgt_pack['bbox_masks'].to(device,
            non_blocking=True)}
        bbox_loss_fn = TableL1Loss(reduction='sum')
        bbox_losses = bbox_loss_fn(bbox_pred, bbox_targets)
        loss_bbox = bbox_losses['horizon_bbox_loss'] + bbox_losses[
            'vertical_bbox_loss']
    out_cell = sup_out.get('out_cell', None)
    cell_disabled = not enable_cell or out_cell is None
    if not cell_disabled and isinstance(out_cell, list):
        all_empty = True
        for x in out_cell:
            if torch.is_tensor(x) and x.numel() > 0:
                all_empty = False
                break
        if all_empty:
            cell_disabled = True
    if cell_disabled:
        loss_cell = torch.zeros((), device=device)
    else:
        outputs_flat, targets_flat = pack_cell_to_master_format(out_cell,
            tgt_pack['cell_padded_targets'], pad_cell_id=int(cfg.PAD_CELL),
            device=device)
        if outputs_flat is None:
            loss_cell = torch.zeros((), device=device)
        else:
            cell_loss_fn = MASTERCELLLoss(ignore_index=int(cfg.PAD_CELL),
                reduction='mean', flatten=True)
            loss_cell = cell_loss_fn(outputs_flat, {'padded_targets':
                targets_flat})
    loss_sup = loss_tf + loss_bbox + w_cell * loss_cell
    return loss_tf, loss_bbox, w_cell * loss_cell, loss_sup
