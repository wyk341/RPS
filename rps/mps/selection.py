import math
import torch

@torch.no_grad()
def build_row_gated_token_mask_v4(t_conf: torch.Tensor, p_on_t: torch.
    Tensor, p_t_on_t: torch.Tensor, t_ids: torch.Tensor, tr_spans,
    fill_mask: torch.Tensor, id2tok: list, PAD: int, EOS: int, cfg, step=
    None, epoch=None, return_weight: bool=True):
    device = t_ids.device
    B, L = t_ids.shape
    base_valid = (t_ids != PAD) & ~fill_mask
    if EOS is not None and EOS >= 0:
        eos_mask = t_ids == EOS
        after_or_at = eos_mask.cumsum(dim=1) >= 1
        after = after_or_at & ~eos_mask
        base_valid = base_valid & ~after
    use_confidence = bool(getattr(cfg, 'mps_use_confidence', True))
    use_stability = bool(getattr(cfg, 'mps_use_stability', True))
    use_agreement = bool(getattr(cfg, 'mps_use_agreement', True))
    q = float(getattr(cfg, 'row_conf_quantile', 0.35))
    min_row_tokens = int(getattr(cfg, 'min_row_tokens', 2))
    min_keep_rows = int(getattr(cfg, 'min_keep_rows', 3))
    keep_ratio = float(getattr(cfg, 'row_keep_ratio', 0.85))
    if not use_confidence:
        keep_ratio = 1.0
    min_keep_rows_thead = int(getattr(cfg, 'min_keep_rows_thead', 1))
    min_keep_rows_tbody = int(getattr(cfg, 'min_keep_rows_tbody', 2))
    fallback_row_weight = float(getattr(cfg, 'fallback_row_weight', 0.2))
    boundary_weight = float(getattr(cfg, 'boundary_token_weight', 0.3))
    boundary_set = {'<table>', '</table>', '<thead>', '</thead>', '<tbody>',
        '</tbody>'}
    stab_quantile = float(getattr(cfg, 'row_stab_quantile', q))
    agree_thr0 = float(getattr(cfg, 'row_agree_thr', 0.45))
    agree_thr1 = float(getattr(cfg, 'row_agree_thr_max', agree_thr0))
    agree_T = int(getattr(cfg, 'agree_thr_warmup_epochs', 10))
    stab_thr0 = float(getattr(cfg, 'row_stab_thr', 0.45))
    stab_thr1 = float(getattr(cfg, 'row_stab_thr_max', stab_thr0))
    stab_T = int(getattr(cfg, 'stab_thr_warmup_epochs', 10))
    if not use_agreement:
        agree_thr0 = agree_thr1 = 0.0
    if not use_stability:
        stab_thr0 = stab_thr1 = 0.0
    t_stageb = max(0, int(epoch or 0) - int(getattr(cfg, 'sup_only_epochs', 0))
        )
    agree_thr = agree_thr0 + (agree_thr1 - agree_thr0) * min(1.0, t_stageb /
        float(max(1, agree_T)))
    stab_thr = stab_thr0 + (stab_thr1 - stab_thr0) * min(1.0, t_stageb /
        float(max(1, stab_T)))
    boundary_mask = torch.zeros((B, L), dtype=torch.bool, device=device)
    for b in range(B):
        for i in range(L):
            if not base_valid[b, i]:
                continue
            tok = id2tok[int(t_ids[b, i].item())]
            if tok in boundary_set:
                boundary_mask[b, i] = True
    token_mask = torch.zeros((B, L), dtype=torch.bool, device=device)
    token_weight = torch.zeros((B, L), dtype=torch.float32, device=device)
    for b in range(B):
        rows = tr_spans[b] or []
        if len(rows) == 0:
            token_mask[b] |= boundary_mask[b]
            token_weight[b][boundary_mask[b]] = boundary_weight
            continue
        row_scores = []
        row_agrees = []
        row_stabs = []
        row_slices = []
        for s, e, sec in rows:
            s = max(0, min(int(s), L))
            e = max(0, min(int(e), L))
            if e <= s:
                continue
            m = base_valid[(b), s:e]
            valid_cnt = int(m.sum().item())
            if valid_cnt < min_row_tokens:
                continue
            conf_row = t_conf[(b), s:e][m]
            if conf_row.numel() == 0:
                continue
            if not use_confidence:
                conf_row = torch.ones_like(conf_row)
            score = torch.quantile(conf_row, q)
            pon = p_on_t[(b), s:e][m]
            agree = torch.quantile(pon, q) if pon.numel(
                ) > 0 else torch.tensor(0.0, device=device)
            pstab = p_t_on_t[(b), s:e][m]
            stab = torch.quantile(pstab, stab_quantile) if pstab.numel(
                ) > 0 else torch.tensor(0.0, device=device)
            row_scores.append(score)
            row_agrees.append(agree)
            row_stabs.append(stab)
            row_slices.append((s, e, sec))
        if len(row_scores) == 0:
            token_mask[b] |= boundary_mask[b]
            token_weight[b][boundary_mask[b]] = boundary_weight
            continue
        scores = torch.stack(row_scores)
        agrees = torch.stack(row_agrees)
        stabs = torch.stack(row_stabs)
        secs = torch.tensor([sec for _, _, sec in row_slices], device=
            device, dtype=torch.long)
        cand_row = torch.zeros_like(scores, dtype=torch.bool)

        def _mark_topk_global(min_keep: int):
            n = int(scores.numel())
            if n == 0:
                return
            k = int(max(min_keep, math.ceil(float(n) * keep_ratio)))
            k = min(k, n)
            topk_idx = torch.topk(scores, k=k, largest=True).indices
            cand_row[topk_idx] = True

        def _mark_topk_for_sec(sec_id: int, min_keep: int):
            idx = (secs == sec_id).nonzero(as_tuple=True)[0]
            if idx.numel() == 0:
                return
            sc = scores[idx]
            k = int(max(min_keep, math.ceil(float(sc.numel()) * keep_ratio)))
            k = min(k, int(sc.numel()))
            topk_local = torch.topk(sc, k=k, largest=True).indices
            cand_row[idx[topk_local]] = True
        has_valid_sec = bool(((secs == 0) | (secs == 1)).any().item())
        if has_valid_sec:
            _mark_topk_for_sec(0, min_keep_rows_thead)
            _mark_topk_for_sec(1, min_keep_rows_tbody)
            if int(cand_row.sum().item()) == 0:
                _mark_topk_global(min_keep_rows)
        else:
            _mark_topk_global(min_keep_rows)
        keep_row = cand_row & (agrees >= agree_thr) & (stabs >= stab_thr)
        fallback_row = torch.zeros_like(scores, dtype=torch.bool)
        if step == 1:
            kr = int(keep_row.sum().item())
            cr = int(cand_row.sum().item())
            sec_info = secs.tolist()
            print(
                f'[GATE] b={b} rows={len(row_slices)} cand={cr} keep={kr} score_mean={scores.mean().item():.3f} agree_mean={agrees.mean().item():.3f} stab_mean={stabs.mean().item():.3f} agree_thr={agree_thr:.3f} stab_thr={stab_thr:.3f} secs={sec_info}'
                )
        if int(keep_row.sum().item()) < min_keep_rows:
            need = min_keep_rows - int(keep_row.sum().item())
            stable_row = stabs >= stab_thr
            cand = (cand_row & stable_row & ~keep_row).nonzero(as_tuple=True)[0
                ]
            if cand.numel() > 0 and need > 0:
                sc = scores[cand]
                k = min(int(sc.numel()), int(need))
                add_local = torch.topk(sc, k=k, largest=True).indices
                add_idx = cand[add_local]
                fallback_row[add_idx] = True
            if int(fallback_row.sum().item()) < need:
                need2 = need - int(fallback_row.sum().item())
                cand2 = (~cand_row & stable_row & ~keep_row).nonzero(as_tuple
                    =True)[0]
                if cand2.numel() > 0 and need2 > 0:
                    sc2 = scores[cand2]
                    k2 = min(int(sc2.numel()), int(need2))
                    add_local2 = torch.topk(sc2, k=k2, largest=True).indices
                    add_idx2 = cand2[add_local2]
                    fallback_row[add_idx2] = True
        for flag, (s, e, _sec) in zip(keep_row.tolist(), row_slices):
            if not flag:
                continue
            m = base_valid[(b), s:e]
            if m.any():
                token_mask[(b), s:e] |= m
                token_weight[(b), s:e][m] = 1.0
        for flag, (s, e, _sec) in zip(fallback_row.tolist(), row_slices):
            if not flag:
                continue
            m = base_valid[(b), s:e]
            if m.any():
                token_mask[(b), s:e] |= m
                token_weight[(b), s:e][m] = torch.maximum(token_weight[(b),
                    s:e][m], torch.full_like(token_weight[(b), s:e][m],
                    fallback_row_weight))
        token_mask[b] |= boundary_mask[b]
        token_weight[b][boundary_mask[b]] = torch.maximum(token_weight[b][
            boundary_mask[b]], torch.full_like(token_weight[b][
            boundary_mask[b]], boundary_weight))
    if not return_weight:
        return token_mask
    token_weight = token_weight * token_mask.float()
    return token_mask, token_weight
