import os
import sys
import numpy as np
import torch
from tqdm import tqdm
from .teds import Analysis_TED
from rps.runtime import DataContainer

def cell_to_str(x):
    if isinstance(x, list):
        return ''.join(str(t) for t in x)
    return str(x)


def cxcywh_to_xyxy_np(boxes: np.ndarray) ->np.ndarray:
    if boxes.size == 0:
        return boxes.reshape(0, 4)
    cx, cy, w, h = boxes[:, (0)], boxes[:, (1)], boxes[:, (2)], boxes[:, (3)]
    x1 = cx - w / 2.0
    y1 = cy - h / 2.0
    x2 = cx + w / 2.0
    y2 = cy + h / 2.0
    return np.stack([x1, y1, x2, y2], axis=-1)


def iou_matrix_np(pred_xyxy: np.ndarray, gt_xyxy: np.ndarray) ->np.ndarray:
    if pred_xyxy.size == 0 or gt_xyxy.size == 0:
        return np.zeros((pred_xyxy.shape[0], gt_xyxy.shape[0]), dtype=float)
    pred = pred_xyxy[:, (None), :]
    gt = gt_xyxy[(None), :, :]
    x1 = np.maximum(pred[..., 0], gt[..., 0])
    y1 = np.maximum(pred[..., 1], gt[..., 1])
    x2 = np.minimum(pred[..., 2], gt[..., 2])
    y2 = np.minimum(pred[..., 3], gt[..., 3])
    inter_w = np.clip(x2 - x1, a_min=0.0, a_max=None)
    inter_h = np.clip(y2 - y1, a_min=0.0, a_max=None)
    inter_area = inter_w * inter_h
    area_pred = np.clip(pred[..., 2] - pred[..., 0], 0.0, None) * np.clip(
        pred[..., 3] - pred[..., 1], 0.0, None)
    area_gt = np.clip(gt[..., 2] - gt[..., 0], 0.0, None) * np.clip(gt[...,
        3] - gt[..., 1], 0.0, None)
    union = area_pred + area_gt - inter_area + 1e-06
    iou = inter_area / union
    return iou


def match_boxes_iou(gt_bbox_valid: np.ndarray, pred_bbox_valid: np.ndarray,
    iou_thr: float=0.5, iou_thr_debug: float=0.3):
    N_gt = gt_bbox_valid.shape[0]
    N_pred = pred_bbox_valid.shape[0]
    if N_gt == 0 and N_pred == 0:
        return 0, 0, 0, 0
    if N_gt == 0 and N_pred > 0:
        return 0, N_pred, 0, 0
    if N_gt > 0 and N_pred == 0:
        return 0, 0, N_gt, 0
    gt_xyxy = cxcywh_to_xyxy_np(gt_bbox_valid)
    pred_xyxy = cxcywh_to_xyxy_np(pred_bbox_valid)
    ious = iou_matrix_np(pred_xyxy, gt_xyxy)
    loose_tp = int((ious >= iou_thr_debug).sum())
    TP = 0
    used_pred = np.zeros(N_pred, dtype=bool)
    used_gt = np.zeros(N_gt, dtype=bool)
    while True:
        max_idx_flat = ious.argmax()
        max_iou = ious.flat[max_idx_flat]
        if max_iou < iou_thr:
            break
        pi, gi = divmod(max_idx_flat, N_gt)
        if used_pred[pi] or used_gt[gi]:
            ious[pi, gi] = -1.0
            continue
        TP += 1
        used_pred[pi] = True
        used_gt[gi] = True
        ious[(pi), :] = -1.0
        ious[:, (gi)] = -1.0
    FP = int((~used_pred).sum())
    FN = int((~used_gt).sum())
    return TP, FP, FN, loose_tp


def unwrap_img(x):
    if isinstance(x, DataContainer):
        x = x.data
    if isinstance(x, (list, tuple)):
        x = x[0]
    if isinstance(x, torch.Tensor) and x.dim() == 3:
        x = x.unsqueeze(0)
    return x


def unwrap_metas(x):
    if isinstance(x, DataContainer):
        x = x.data
    if isinstance(x, (list, tuple)):
        return x[0] if len(x) > 0 and isinstance(x[0], list) else list(x)
    if isinstance(x, dict):
        return [x]
    return x


def levenshtein(a: str, b: str) ->int:
    la, lb = len(a), len(b)
    if la == 0:
        return lb
    if lb == 0:
        return la
    prev = list(range(lb + 1))
    curr = [0] * (lb + 1)
    for i in range(1, la + 1):
        curr[0] = i
        ca = a[i - 1]
        for j in range(1, lb + 1):
            cb = b[j - 1]
            cost = 0 if ca == cb else 1
            curr[j] = min(prev[j] + 1, curr[j - 1] + 1, prev[j - 1] + cost)
        prev, curr = curr, prev
    return prev[lb]


def filter_zero_boxes(boxes, eps=1e-06):
    if isinstance(boxes, torch.Tensor):
        boxes = boxes.detach().cpu().numpy()
    b = np.asarray(boxes, dtype=float)
    if b.size == 0:
        return np.zeros((0, 4), dtype=float)
    mask = (np.abs(b).sum(axis=1) > eps) & (b[:, (2)] > eps) & (b[:, (3)] > eps
        )
    return b[mask]


@torch.no_grad()
def _evaluate_on_val(model, val_loader, convertor, max_batches=5,
    compute_bbox=True, compute_teds=True, compute_cer=True,
    structure_normalizer=None):
    device = next(model.parameters()).device
    model.eval()
    teds_scores = []
    n_images = n_scored = missing_gt = duplicate_ids = missing_ids = 0
    seen_ids = set()
    TP = FP = FN = 0
    total_edit = 0
    total_chars = 0
    IOU_THR = 0.5
    IOU_THR_DEBUG = 0.3
    TP_debug_loose = 0
    if hasattr(val_loader, '__len__'):
        total_iters = min(max_batches, len(val_loader))
    else:
        total_iters = max_batches
    disable_tqdm = not sys.__stderr__.isatty() or bool(os.environ.get(
        'DISABLE_TQDM', ''))
    pbar = tqdm(val_loader, total=total_iters, desc='Validating', leave=
        False, dynamic_ncols=True, file=sys.__stderr__, disable=disable_tqdm)
    for step, batch in enumerate(pbar, 1):
        if step > max_batches:
            break
        imgs = unwrap_img(batch['img']).to(device)
        metas = unwrap_metas(batch['img_metas'])
        out = model.forward_train(imgs, metas, collect_va=False,
            strong_view=False, train_mode=False, targets_dict=None)
        bbox_for_convertor = out.get('out_bbox', None)
        if bbox_for_convertor is None:
            B_tmp, L_tmp, _ = out['out_dec'].shape
            bbox_for_convertor = torch.zeros(B_tmp, L_tmp, 4, device=out[
                'out_dec'].device)
        out_cell = out.get('out_cell', None)
        if out_cell is None:
            B_tmp = out['out_dec'].shape[0]
            out_cell = [torch.zeros(0, 1, 1, device=out['out_dec'].device) for
                _ in range(B_tmp)]
        strings, scores, pred_bboxes_pixel, cell_strings, cell_scores = (
            convertor.output_format(out['out_dec'], bbox_for_convertor,
            out_cell, metas))
        B = len(metas)
        if compute_bbox:
            pred_bbox_norm = out.get('out_bbox', None)
            if pred_bbox_norm is None:
                pred_bbox_norm = bbox_for_convertor
            _, L, _ = pred_bbox_norm.shape
        for bi in range(B):
            meta = metas[bi]
            pred_html = strings[bi]
            n_images += 1
            stable_id = meta.get('ori_filename') or meta.get('orifilename'
                ) or meta.get('filename')
            if stable_id is None or stable_id == '':
                missing_ids += 1
            else:
                stable_id = os.path.normpath(os.fspath(stable_id))
                duplicate_ids += int(stable_id in seen_ids)
                seen_ids.add(stable_id)
            if meta.get('text', None) is None:
                missing_gt += 1
            if compute_teds:
                gt_html = meta.get('text', None)
                if gt_html is not None:
                    score_gt, score_pred = gt_html, pred_html
                    if structure_normalizer is not None:
                        score_gt = structure_normalizer(score_gt)
                        score_pred = structure_normalizer(score_pred)
                    t = Analysis_TED(score_gt, score_pred)
                    teds_scores.append(float(t))
                    n_scored += 1
            if compute_bbox:
                gt_bbox_pack = convertor.str_bbox_format([meta])
                gt_bbox_norm = gt_bbox_pack['bbox'][0].detach().cpu().numpy()
                gt_masks = gt_bbox_pack['bbox_masks'][0].detach().cpu().numpy()
                min_len_gt = min(gt_bbox_norm.shape[0], gt_masks.shape[0])
                gt_bbox_norm = gt_bbox_norm[:min_len_gt]
                gt_masks = gt_masks[:min_len_gt]
                valid_gt_mask = gt_masks > 0.5
                gt_bbox_valid = gt_bbox_norm[valid_gt_mask]
                gt_bbox_valid = filter_zero_boxes(gt_bbox_valid)
                N_gt = gt_bbox_valid.shape[0]
                pred_norm_raw = pred_bbox_norm[bi].detach().cpu().numpy()
                pred_str = pred_html
                pred_mask_arr = convertor._get_pred_bbox_mask([pred_str])[0]
                pred_mask_arr = np.asarray(pred_mask_arr, dtype=np.int64)
                Lp = min(pred_norm_raw.shape[0], pred_mask_arr.shape[0])
                pred_norm_tok = pred_norm_raw[:Lp]
                pred_mask_bool = pred_mask_arr[:Lp].astype(bool)
                pred_bbox_candidate = pred_norm_tok[pred_mask_bool]
                if pred_bbox_candidate.size > 0:
                    area_mask = (np.abs(pred_bbox_candidate).sum(axis=1) > 
                        1e-06) & (pred_bbox_candidate[:, (2)] > 0.001) & (
                        pred_bbox_candidate[:, (3)] > 0.001)
                    pred_bbox_valid = pred_bbox_candidate[area_mask]
                else:
                    pred_bbox_valid = np.zeros((0, 4), dtype=float)
                N_pred = pred_bbox_valid.shape[0]
                if N_gt == 0 and N_pred == 0:
                    pass
                else:
                    tp_b, fp_b, fn_b, loose_tp_b = match_boxes_iou(
                        gt_bbox_valid, pred_bbox_valid, iou_thr=IOU_THR,
                        iou_thr_debug=IOU_THR_DEBUG)
                    TP += tp_b
                    FP += fp_b
                    FN += fn_b
                    TP_debug_loose += loose_tp_b
                    if step == 1 and bi == 0 and N_gt > 0 and N_pred > 0:
                        gt_xyxy_dbg = cxcywh_to_xyxy_np(gt_bbox_valid)
                        pred_xyxy_dbg = cxcywh_to_xyxy_np(pred_bbox_valid)
                        ious_dbg = iou_matrix_np(pred_xyxy_dbg, gt_xyxy_dbg)
                        first_best = float(ious_dbg[:, (0)].max())
                        print('[VAL DEBUG] Best IoU for gt[0] =', first_best)
            if compute_cer:
                gt_cells = meta['cell_content']
                pred_cells = cell_strings[bi]
                num_cells = min(len(gt_cells), len(pred_cells))
                for k in range(num_cells):
                    gt_raw = gt_cells[k]
                    pred_raw = pred_cells[k]
                    gt = cell_to_str(gt_raw)
                    pred = cell_to_str(pred_raw)
                    ed = levenshtein(pred, gt)
                    total_edit += ed
                    denom = max(len(gt), len(pred), 1)
                    total_chars += denom
    mean_teds = np.mean(teds_scores) if len(teds_scores) else 0.0
    if compute_bbox:
        precision = TP / (TP + FP + 1e-06)
        recall = TP / (TP + FN + 1e-06)
        f1 = 2 * precision * recall / (precision + recall + 1e-06)
    else:
        precision = recall = f1 = 0.0
    cer = total_edit / (total_chars + 1e-06
        ) if compute_cer and total_chars > 0 else 0.0
    print('\n[VAL DEBUG] ---- bbox matching summary ----')
    if compute_bbox:
        print(f'TP (IoU >= {IOU_THR})              : {TP}')
        print(f'FP                                  : {FP}')
        print(f'FN                                  : {FN}')
        print(f'Loose TP (IoU >= {IOU_THR_DEBUG})   : {TP_debug_loose}')
    else:
        print('BBox metrics disabled (compute_bbox=False)')
    print('============================================\n')
    return {'TEDS': mean_teds, 'precision': precision, 'recall': recall,
        'F1': f1, 'CER': cer, 'score': mean_teds + f1 + (1 - cer),
        'n_images': n_images, 'n_scored': n_scored, 'n_unique': len(
        seen_ids), 'missing_gt': missing_gt, 'duplicate_ids': duplicate_ids,
        'missing_ids': missing_ids, 'structure_normalizer_enabled': 
        structure_normalizer is not None}

from .normalization import normalize_structure_for_teds


def evaluate_on_val(*args, **kwargs):
    kwargs.setdefault('structure_normalizer',normalize_structure_for_teds)
    return _evaluate_on_val(*args,**kwargs)
