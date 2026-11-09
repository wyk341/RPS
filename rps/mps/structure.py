import math
import torch
import numpy as np
TAGS=['thead','tbody','tr','td']

def _classify_token(tok: str):
    for tag in TAGS:
        if f'<{tag}' in tok and f'</{tag}>' in tok:
            return [('balanced', tag)]
    if tok.startswith('<eb') and '</eb' in tok and tok.endswith('>'):
        return [('balanced', 'eb')]
    for tag in TAGS:
        if tok == f'</{tag}>':
            return [('close', tag)]
    for tag in TAGS:
        if tok == f'<{tag}>':
            return [('open', tag)]
        if tok.startswith(f'<{tag}') and not tok.startswith(f'</{tag}'):
            if tok.endswith('/>'):
                return [('balanced', tag)]
            return [('open', tag)]
    return []


def _get_eff_len(seq_1d: torch.Tensor, PAD: int, EOS: int):
    L = seq_1d.numel()
    eff = 0
    for i in range(L):
        tid = int(seq_1d[i].item())
        if tid == PAD or tid == EOS:
            break
        eff += 1
    return eff


def _insert_tokens_inplace(seq_1d: torch.Tensor, eff_len: int, insert_pos:
    int, ins_ids: list, PAD: int):
    L = seq_1d.numel()
    n = len(ins_ids)
    if n <= 0:
        return True, eff_len
    free = L - eff_len
    if free < n:
        return False, eff_len
    if eff_len > insert_pos:
        seq_1d[insert_pos + n:eff_len + n] = seq_1d[insert_pos:eff_len].clone()
    seq_1d[insert_pos:insert_pos + n] = torch.tensor(ins_ids, dtype=seq_1d.
        dtype, device=seq_1d.device)
    return True, eff_len + n


def _insert_logits_inplace(logits_row: torch.Tensor, eff: int, pos: int, n:
    int, pad_id: int, fill_to_pad: bool=True):
    L, V = logits_row.shape
    free = L - eff
    if n <= 0 or n > free:
        return False, eff
    if eff > pos:
        src = logits_row[pos:eff].clone()
        logits_row[pos + n:eff + n].copy_(src)
    if fill_to_pad:
        logits_row[pos:pos + n].fill_(-10000.0)
        logits_row[pos:pos + n, (pad_id)] = 10000.0
    else:
        logits_row[pos:pos + n].zero_()
    return True, eff + n


@torch.no_grad()
def repair_pubtabnet_thead_tbody_by_insertion(t_ids: torch.Tensor, t_logits:
    torch.Tensor, id2tok: list, tok2id: dict, PAD: int, EOS: int,
    max_close_tokens: int=64, fill_logits_to_pad: bool=True, return_ops:
    bool=False):
    B, L = t_ids.shape
    t_fix = t_ids.clone()
    t_logits_fix = t_logits.clone()
    fill_mask = torch.zeros((B, L), dtype=torch.bool, device=t_ids.device)
    debug_inserted = [[] for _ in range(B)]
    insert_ops = [[] for _ in range(B)]

    def tid(tok: str):
        v = tok2id.get(tok, None)
        return None if v is None else int(v)
    CLOSE = {'td': tid('</td>'), 'tr': tid('</tr>'), 'thead': tid(
        '</thead>'), 'tbody': tid('</tbody>')}
    for b in range(B):
        seq = t_fix[b]
        eff = _get_eff_len(seq, PAD, EOS)
        mode = None
        stack = []
        i = 0
        while i < eff:
            tok = id2tok[int(seq[i].item())]
            events = _classify_token(tok)
            for kind, tag in events:
                if kind == 'balanced':
                    continue
                if kind == 'open':
                    if tag == 'thead':
                        mode = 'thead'
                        stack.append('thead')
                    elif tag == 'tbody':
                        if mode == 'thead':
                            close_ids = []
                            tmp = stack.copy()
                            while tmp:
                                top = tmp.pop()
                                if top in ('td', 'tr', 'thead'):
                                    cid = CLOSE.get(top, None)
                                    if cid is not None:
                                        close_ids.append(cid)
                                if top == 'thead':
                                    break
                            close_ids = close_ids[:max_close_tokens]
                            if close_ids:
                                seq_backup = seq.clone()
                                logits_backup = t_logits_fix[b].clone()
                                ok, eff2 = _insert_tokens_inplace(seq, eff,
                                    i, close_ids, PAD)
                                if ok:
                                    ok_l, _ = _insert_logits_inplace(
                                        t_logits_fix[b], eff, i, len(
                                        close_ids), PAD, fill_to_pad=
                                        fill_logits_to_pad)
                                    if not ok_l:
                                        seq.copy_(seq_backup)
                                        t_logits_fix[b].copy_(logits_backup)
                                        ok = False
                                    else:
                                        fill_mask[(b), i:i + len(close_ids)] = True
                                        debug_inserted[b] += [id2tok[x] for x in
                                            close_ids]
                                        for k, cid in enumerate(close_ids):
                                            insert_ops[b].append((int(i) + k, int(cid))
                                                )
                                        eff = eff2
                                        i += len(close_ids)
                                        while stack:
                                            top = stack.pop()
                                            if top == 'thead':
                                                break
                        mode = 'tbody'
                        stack.append('tbody')
                    elif tag == 'tr':
                        if stack and stack[-1] == 'td' and CLOSE['td'
                            ] is not None:
                            ok, eff2 = _insert_tokens_inplace(seq, eff, i,
                                [CLOSE['td']], PAD)
                            if ok:
                                _insert_logits_inplace(t_logits_fix[b], eff,
                                    i, 1, PAD, fill_to_pad=fill_logits_to_pad)
                                fill_mask[(b), i:i + 1] = True
                                insert_ops[b].append((int(i), int(CLOSE['td']))
                                    )
                                debug_inserted[b].append(id2tok[CLOSE['td']])
                                eff = eff2
                                i += 1
                                if stack and stack[-1] == 'td':
                                    stack.pop()
                        if stack and stack[-1] == 'tr' and CLOSE['tr'
                            ] is not None:
                            ok, eff2 = _insert_tokens_inplace(seq, eff, i,
                                [CLOSE['tr']], PAD)
                            if ok:
                                _insert_logits_inplace(t_logits_fix[b], eff,
                                    i, 1, PAD, fill_to_pad=fill_logits_to_pad)
                                fill_mask[(b), i:i + 1] = True
                                insert_ops[b].append((int(i), int(CLOSE['tr']))
                                    )
                                debug_inserted[b].append(id2tok[CLOSE['tr']])
                                eff = eff2
                                i += 1
                                if stack and stack[-1] == 'tr':
                                    stack.pop()
                        stack.append('tr')
                    elif tag == 'td':
                        if stack and stack[-1] == 'td' and CLOSE['td'
                            ] is not None:
                            ok, eff2 = _insert_tokens_inplace(seq, eff, i,
                                [CLOSE['td']], PAD)
                            if ok:
                                _insert_logits_inplace(t_logits_fix[b], eff,
                                    i, 1, PAD, fill_to_pad=fill_logits_to_pad)
                                fill_mask[(b), i:i + 1] = True
                                insert_ops[b].append((int(i), int(CLOSE['td']))
                                    )
                                debug_inserted[b].append(id2tok[CLOSE['td']])
                                eff = eff2
                                i += 1
                                stack.pop()
                        stack.append('td')
                elif kind == 'close':
                    if tag in stack:
                        while stack and stack[-1] != tag:
                            stack.pop()
                        if stack and stack[-1] == tag:
                            stack.pop()
                            if tag in ('thead', 'tbody'):
                                mode = None
            i += 1
        if stack:
            close_ids = []
            while stack and len(close_ids) < max_close_tokens:
                top = stack.pop()
                cid = CLOSE.get(top, None)
                if cid is not None:
                    close_ids.append(cid)
            free = L - eff
            if close_ids and len(close_ids) <= free:
                n = len(close_ids)
                seq[eff:eff + n] = torch.tensor(close_ids, dtype=seq.dtype,
                    device=seq.device)
                fill_mask[(b), eff:eff + n] = True
                for k, cid in enumerate(close_ids):
                    insert_ops[b].append((int(eff) + k, int(cid)))
                debug_inserted[b] += [id2tok[x] for x in close_ids]
                if fill_logits_to_pad:
                    t_logits_fix[(b), eff:eff + n].fill_(-10000.0)
                    t_logits_fix[(b), eff:eff + n, (PAD)] = 10000.0
                else:
                    t_logits_fix[(b), eff:eff + n].zero_()
    if return_ops:
        return t_fix, t_logits_fix, fill_mask, debug_inserted, insert_ops
    return t_fix, t_logits_fix, fill_mask, debug_inserted


@torch.no_grad()
def extract_tr_spans_table_aware(t_ids: torch.Tensor, id2tok: list, tok2id:
    dict, PAD: int, EOS: int, max_rows: int=256, default_section_if_missing:
    int=1):
    B, L = t_ids.shape
    has_thead = '<thead>' in tok2id
    has_tbody = '<tbody>' in tok2id
    has_table = '<table>' in tok2id

    def _eff_len(seq_1d):
        eff = 0
        for i in range(seq_1d.numel()):
            x = int(seq_1d[i].item())
            if x == PAD or x == EOS:
                break
            eff += 1
        return eff
    spans_list = []
    for b in range(B):
        seq = t_ids[b]
        eff = _eff_len(seq)
        rows = []
        open_tr = None
        mode = -1
        saw_explicit_section = False
        inside_table = False
        i = 0
        while i < eff:
            tok = id2tok[int(seq[i].item())]
            if tok == '<table>' and has_table:
                inside_table = True
                if not has_thead and not has_tbody and mode == -1:
                    mode = default_section_if_missing
            elif tok == '</table>':
                if open_tr is not None:
                    rows.append((open_tr, i, mode if mode != -1 else
                        default_section_if_missing))
                    open_tr = None
                inside_table = False
                mode = -1
            elif tok == '<thead>' and has_thead:
                saw_explicit_section = True
                if open_tr is not None:
                    rows.append((open_tr, i, mode if mode != -1 else 0))
                    open_tr = None
                mode = 0
            elif tok == '</thead>':
                if open_tr is not None:
                    rows.append((open_tr, i, 0))
                    open_tr = None
                mode = -1 if has_tbody else default_section_if_missing
            elif tok == '<tbody>' and has_tbody:
                saw_explicit_section = True
                if open_tr is not None:
                    rows.append((open_tr, i, mode if mode != -1 else 1))
                    open_tr = None
                mode = 1
            elif tok == '</tbody>':
                if open_tr is not None:
                    rows.append((open_tr, i, 1))
                    open_tr = None
                mode = -1
            if tok == '<tr>':
                if mode == -1:
                    if inside_table:
                        cur_sec = default_section_if_missing
                    else:
                        cur_sec = default_section_if_missing
                else:
                    cur_sec = mode
                if open_tr is not None:
                    rows.append((open_tr, i, cur_sec))
                open_tr = i
                row_mode_at_open = cur_sec
            elif tok == '</tr>':
                if open_tr is not None:
                    rows.append((open_tr, i + 1, row_mode_at_open))
                    open_tr = None
            if len(rows) >= max_rows:
                break
            i += 1
        if open_tr is not None and len(rows) < max_rows:
            final_sec = row_mode_at_open if 'row_mode_at_open' in locals(
                ) else default_section_if_missing
            rows.append((open_tr, eff, final_sec))
        fixed_rows = []
        for s, e, sec in rows:
            if sec == -1:
                sec = default_section_if_missing
            fixed_rows.append((s, e, sec))
        spans_list.append(fixed_rows)
    return spans_list


@torch.no_grad()
def apply_insert_ops_to_seq_and_logits(t_ids_s: torch.Tensor, t_logits_s:
    torch.Tensor, insert_ops, PAD: int, fill_logits_to_pad: bool=True):
    device = t_ids_s.device
    B, Ls = t_ids_s.shape
    Bs, Llog, V = t_logits_s.shape
    assert Bs == B
    L0 = min(Ls, Llog)
    t_ids_s = t_ids_s[:, :L0]
    t_logits_s = t_logits_s[:, :L0, :]
    new_ids = []
    new_logits = []
    for b in range(B):
        ops = sorted(insert_ops[b], key=lambda x: int(x[0])
            ) if insert_ops is not None else []
        ids = t_ids_s[b]
        logits = t_logits_s[b]
        offset = 0
        for pos, tok_id in ops:
            pos = int(pos) + offset
            pos = max(0, min(pos, ids.numel()))
            tok_id = int(tok_id)
            ids = torch.cat([ids[:pos], torch.tensor([tok_id], device=
                device, dtype=ids.dtype), ids[pos:]], dim=0)
            if fill_logits_to_pad:
                row = torch.full((1, V), -10000.0, device=device, dtype=
                    logits.dtype)
                row[0, int(PAD)] = 10000.0
            else:
                row = torch.zeros((1, V), device=device, dtype=logits.dtype)
            logits = torch.cat([logits[:pos], row, logits[pos:]], dim=0)
            offset += 1
        new_ids.append(ids)
        new_logits.append(logits)
    Lnew = max(x.numel() for x in new_ids)
    out_ids = torch.full((B, Lnew), int(PAD), device=device, dtype=t_ids_s.
        dtype)
    out_logits = torch.full((B, Lnew, V), -10000.0, device=device, dtype=
        t_logits_s.dtype)
    out_logits[:, :, (int(PAD))] = 10000.0
    for b in range(B):
        lb = new_ids[b].numel()
        out_ids[(b), :lb] = new_ids[b]
        out_logits[(b), :lb, :] = new_logits[b]
    return out_ids, out_logits
