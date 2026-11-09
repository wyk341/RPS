import torch
import torch.nn.functional as F

def _safe_decode(ids_1d, id2tok, maxn=40):
    out = []
    for x in ids_1d[:maxn].tolist():
        if 0 <= x < len(id2tok):
            out.append(id2tok[x])
        else:
            out.append(f'<OOV:{x}>')
    return out


def loss_structure_unsup_tr_block(out_s, cfg, pseudo_targets, token_mask=
    None, token_weight=None, step=None):
    logits_dec = out_s['out_dec']
    device = logits_dec.device
    tgt_seq = pseudo_targets['padded_targets'].to(device, non_blocking=True)
    PAD = int(getattr(cfg, 'PAD', -1))
    EOS = int(getattr(cfg, 'EOS', -1))
    SOS = int(getattr(cfg, 'SOS', -1))
    B, L, V = logits_dec.shape
    Lt = tgt_seq.size(1)
    if getattr(cfg, 'debug_unsup_align', False):
        print(
            f'[ALIGN] logits_dec: B={B} L={L} V={V} | tgt_seq: Lt={Lt} | PAD={PAD} SOS={SOS} EOS={EOS}'
            )
    if Lt < L + 1:
        L = max(0, Lt - 1)
        logits_dec = logits_dec[:, :L, :]
    else:
        tgt_seq = tgt_seq[:, :L + 1]
    if L == 0:
        return logits_dec.new_zeros(B)
    tgt_use = tgt_seq[:, 1:1 + L]
    if EOS >= 0 and EOS != PAD:
        tgt_use = tgt_use.clone()
        tgt_use[tgt_use == EOS] = PAD
    loss_tok = F.cross_entropy(logits_dec.reshape(B * L, V), tgt_use.
        reshape(B * L), ignore_index=PAD, reduction='none').view(B, L)
    valid0 = tgt_use != PAD
    tm = None
    if token_mask is not None:
        tm = token_mask.to(device)
        if tm.size(1) < L:
            pad = torch.zeros((tm.size(0), L - tm.size(1)), dtype=torch.
                bool, device=device)
            tm = torch.cat([tm, pad], dim=1)
        elif tm.size(1) > L:
            tm = tm[:, :L]
        valid = valid0 & tm
    else:
        valid = valid0
    if token_weight is not None:
        w = token_weight.to(device).clamp(0.0, 1.0)
        weight = w * valid.float()
    else:
        weight = valid.float()
    if getattr(cfg, 'debug_unsup_align', False):
        with torch.no_grad():
            base_sum = int(valid0.sum().item())
            valid_sum = int(valid.sum().item())
            tm_sum = int(tm.sum().item()) if tm is not None else -1
            ratio = valid_sum / max(1, base_sum)
            print(
                f'[MASK] valid_sum={valid_sum} / base_sum={base_sum} (ratio={ratio:.3f}) tm_sum={tm_sum}'
                )
    if getattr(cfg, 'debug_unsup_align', False):
        with torch.no_grad():
            print(f'[MASK] per_sample_valid_tokens={valid.sum(dim=1).tolist()}'
                )
    spans = pseudo_targets.get('tr_spans', None)
    if getattr(cfg, 'debug_unsup_align', False):
        b0 = 0
        K = int(getattr(cfg, 'debug_align_tokens', 60))
        K = min(K, L)
        id2tok = None
        if hasattr(cfg, 'id2tok'):
            id2tok = cfg.id2tok
        elif 'id2tok' in pseudo_targets:
            id2tok = pseudo_targets['id2tok']
        with torch.no_grad():
            pred = logits_dec[(b0), :K].argmax(dim=-1)
        print(f'[ALIGN] ---- sample b={b0} head ----')
        print(
            f'[ALIGN] tgt_seq[0][: {K + 1}] ids = {tgt_seq[(b0), :K + 1].tolist()}'
            )
        print(f'[ALIGN] tgt_use[0][: {K}]   ids = {tgt_use[(b0), :K].tolist()}'
            )
        print(f'[ALIGN] pred   [0][: {K}]   ids = {pred.tolist()}')
        if id2tok is not None:
            print(
                f'[ALIGN] tgt_seq toks: {_safe_decode(tgt_seq[(b0), :K + 1], id2tok, maxn=K + 1)}'
                )
            print(
                f'[ALIGN] tgt_use toks: {_safe_decode(tgt_use[(b0), :K], id2tok, maxn=K)}'
                )
            print(f'[ALIGN] pred   toks: {_safe_decode(pred, id2tok, maxn=K)}')
        tm_line = tm[(b0), :K].int().tolist() if tm is not None else None
        print(f'[ALIGN] valid[:{K}]  = {valid[(b0), :K].int().tolist()}')
        if tm_line is not None:
            print(f'[ALIGN] tmask[:{K}]  = {tm_line}')
        print(f'[ALIGN] weight[:{K}] = {weight[(b0), :K].tolist()}')
        print(
            f'[ALIGN] ce[:{K}]     = {[round(x, 4) for x in loss_tok[(b0), :K].detach().cpu().tolist()]}'
            )
        max_off = int(getattr(cfg, 'debug_align_max_off', 12))
        best = -1, -1.0
        with torch.no_grad():
            pred_full = logits_dec[b0].argmax(dim=-1)
            for off in range(0, max_off + 1):
                if 1 + off + L > tgt_seq.size(1):
                    continue
                cand = tgt_seq[(b0), 1 + off:1 + off + L]
                m = cand != PAD
                denom = m.sum().item()
                if denom <= 0:
                    continue
                hit = ((pred_full == cand) & m).sum().item()
                rate = hit / float(denom)
                if rate > best[1]:
                    best = off, rate
        print(
            f'[ALIGN] best_offset_by_pred_match: off={best[0]} match_rate={best[1]:.3f}'
            )
        if best[0] not in (0, -1):
            print(
                f'[ALIGN] extra prefix length = {best[0]} (shift might need 1+{best[0]})'
                )
        if spans is not None and len(spans) > b0 and spans[b0]:
            show = spans[b0][:min(5, len(spans[b0]))]
            print(f'[ALIGN] spans(b0) first5: {show}')
        if spans is not None and len(spans) > b0 and spans[b0]:
            id2tok = getattr(cfg, 'id2tok', None)
            show = spans[b0][:min(5, len(spans[b0]))]
            print(
                f'[SPANCHK] show first {len(show)} spans (in loss index space, L={L})'
                )
            for i, (s, e, sec) in enumerate(show):
                s = max(0, min(int(s), L))
                e = max(0, min(int(e), L))
                if e <= s:
                    print(f'[SPANCHK] #{i} bad span (s,e)=({s},{e})')
                    continue
                wsum = float(weight[(b0), s:e].sum().item())
                msum = float(tm[(b0), s:e].float().sum().item() if tm is not
                    None else -1)
                vsum = float(valid[(b0), s:e].float().sum().item())
                ce_mean = float((loss_tok[(b0), s:e] * weight[(b0), s:e]).
                    sum().item() / max(wsum, 1e-06))
                print(
                    f'[SPANCHK] #{i} sec={sec} span=({s},{e}) len={e - s} valid_cnt={vsum:.0f} mask_cnt={msum:.0f} wsum={wsum:.3f} ce_wmean={ce_mean:.6f}'
                    )
                if vsum < 1 and id2tok is not None:
                    seg = tgt_use[(b0), s:e]
                    uniq = torch.unique(seg).detach().cpu().tolist()
                    toks = _safe_decode(seg, id2tok, maxn=80)
                    print(f'[SPANCHK]    (valid=0) uniq_ids={uniq}')
                    print(f"[SPANCHK]    (valid=0) toks={' '.join(toks[:40])}")
                if id2tok is not None:
                    toks = _safe_decode(tgt_use[(b0), s:e], id2tok, maxn=80)
                    print(f"[SPANCHK]    toks: {' '.join(toks[:40])}")
        tm0 = tm[b0].detach()
        first1 = int((tm0 > 0).float().argmax().item()) if tm0.any() else -1
        print(f'[TM] first_one_idx={first1} tm_sum={int(tm0.sum().item())}/{L}'
            )
        if id2tok is not None and first1 >= 0:
            s = max(0, first1 - 10)
            e = min(L, first1 + 20)
            toks = _safe_decode(tgt_use[(b0), s:e], id2tok, maxn=200)
            print(f'[TM] around first_one ({s}:{e}): ' + ' '.join(toks))
        print(f'[ALIGN] ---- end sample ----')
    if spans is None:
        denom = weight.sum(dim=1).clamp(min=1.0)
        return (loss_tok * weight).sum(dim=1) / denom
    loss_vec = logits_dec.new_zeros(B)
    for b in range(B):
        rows = spans[b]
        if rows is None or len(rows) == 0:
            denom = weight[b].sum().clamp(min=1.0)
            loss_vec[b] = (loss_tok[b] * weight[b]).sum() / denom
            continue
        row_losses = []
        for s, e, section_id in rows:
            s = max(0, min(int(s), L))
            e = max(0, min(int(e), L))
            if e <= s:
                continue
            w_be = weight[(b), s:e]
            if float(w_be.sum().item()) < 1e-06:
                continue
            row_loss = (loss_tok[(b), s:e] * w_be).sum() / w_be.sum().clamp(min
                =1e-06)
            if section_id == 0:
                row_loss = row_loss * float(getattr(cfg, 'thead_row_weight',
                    1.0))
            elif section_id == 1:
                row_loss = row_loss * float(getattr(cfg, 'tbody_row_weight',
                    1.0))
            row_losses.append(row_loss)
        if len(row_losses) == 0:
            denom = weight[b].sum().clamp(min=1.0)
            loss_vec[b] = (loss_tok[b] * weight[b]).sum() / denom
        else:
            loss_vec[b] = torch.stack(row_losses).mean()
    return loss_vec
