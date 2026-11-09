import os
import json
import random
from typing import List, Optional, Tuple
import numpy as np
import torch
import cv2

class OCRBoxCache:

    def __init__(self, jsonl_path: str):
        self.map = {}
        self.jsonl_path = jsonl_path
        if not jsonl_path or not os.path.isfile(jsonl_path):
            print(f'[OCRBoxCache] WARN: jsonl not found: {jsonl_path}')
            return
        cnt = 0
        with open(jsonl_path, 'r', encoding='utf-8') as f:
            for line in f:
                try:
                    rec = json.loads(line)
                except Exception:
                    continue
                p = rec.get('img_path', None)
                b = rec.get('boxes_xyxy_480', None)
                if p and isinstance(b, list) and len(b) > 0:
                    self.map[p] = b
                    self.map[os.path.basename(p)] = b
                    cnt += 1
        print(f'[OCRBoxCache] loaded {cnt} records from {jsonl_path}')

    def get_boxes(self, key: str):
        if key is None:
            return None
        b = self.map.get(key, None)
        if b is None:
            b = self.map.get(os.path.basename(str(key)), None)
        return b


def _draw_boxes_bgr(img_bgr: np.ndarray, boxes_xyxy, color=(0, 255, 0), th=1):
    H, W = img_bgr.shape[:2]
    out = img_bgr.copy()
    if boxes_xyxy is None:
        return out
    for b in boxes_xyxy:
        bb = _clamp_box_xyxy(b, W, H)
        if bb is None:
            continue
        x1, y1, x2, y2 = bb
        cv2.rectangle(out, (x1, y1), (x2, y2), color, th)
    return out


def _hstack_resize_bgr(a: np.ndarray, b: np.ndarray, height=900):

    def resize_to_h(img, h):
        H, W = img.shape[:2]
        if H == h:
            return img
        scale = h / float(H)
        return cv2.resize(img, (int(W * scale), h), interpolation=cv2.
            INTER_AREA)
    a2 = resize_to_h(a, height)
    b2 = resize_to_h(b, height)
    return np.hstack([a2, b2])


def _ensure_dir(d):
    if d:
        os.makedirs(d, exist_ok=True)


def _parse_norm_cfg(cfg):
    norm = getattr(cfg, 'img_norm_cfg', None)
    if norm is None and hasattr(cfg, 'get'):
        norm = cfg.get('img_norm_cfg', None)
    if norm is None:
        return None, None
    mean = norm.get('mean', None)
    std = norm.get('std', None)
    if mean is None or std is None:
        return None, None
    mean = np.array(mean, dtype=np.float32)
    std = np.array(std, dtype=np.float32)
    return mean, std


def _is_probably_normalized(imgs: torch.Tensor) ->bool:
    mn = float(imgs.detach().min().cpu())
    mx = float(imgs.detach().max().cpu())
    return mn < -0.2 or mx > 1.5


def _denorm_to_01(imgs: torch.Tensor, mean: Optional[np.ndarray], std:
    Optional[np.ndarray]):
    x = imgs
    if mean is None or std is None:
        return x.clamp(0, 1)
    mean_t = torch.tensor(mean, device=x.device, dtype=x.dtype).view(1, -1,
        1, 1)
    std_t = torch.tensor(std, device=x.device, dtype=x.dtype).view(1, -1, 1, 1)
    y = x * std_t + mean_t
    if float(mean.max()) > 10.0:
        y = y / 255.0
    return y.clamp(0, 1)


def _renorm_from_01(imgs01: torch.Tensor, mean: Optional[np.ndarray], std:
    Optional[np.ndarray]):
    if mean is None or std is None:
        return imgs01
    mean_t = torch.tensor(mean, device=imgs01.device, dtype=imgs01.dtype).view(
        1, -1, 1, 1)
    std_t = torch.tensor(std, device=imgs01.device, dtype=imgs01.dtype).view(
        1, -1, 1, 1)
    y = imgs01
    if float(mean.max()) > 10.0:
        y = y * 255.0
    y = (y - mean_t) / (std_t + 1e-06)
    return y


def _tensor_to_u8_bgr(img_chw: torch.Tensor) ->np.ndarray:
    x = img_chw.detach().float().cpu().clamp(0, 1)
    x = (x.permute(1, 2, 0).numpy() * 255.0).astype(np.uint8)
    return cv2.cvtColor(x, cv2.COLOR_RGB2BGR)


def _u8_bgr_to_tensor(img_bgr: np.ndarray, device, dtype=torch.float32
    ) ->torch.Tensor:
    x = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB).astype(np.float32) / 255.0
    return torch.from_numpy(x).permute(2, 0, 1).to(device=device, dtype=dtype)


def _clamp_box_xyxy(box, W, H):
    x1, y1, x2, y2 = box
    x1 = int(max(0, min(round(x1), W - 1)))
    y1 = int(max(0, min(round(y1), H - 1)))
    x2 = int(max(0, min(round(x2), W - 1)))
    y2 = int(max(0, min(round(y2), H - 1)))
    if x2 <= x1 + 1 or y2 <= y1 + 1:
        return None
    return x1, y1, x2, y2


def _iou_xyxy(a, b):
    ax1, ay1, ax2, ay2 = a
    bx1, by1, bx2, by2 = b
    ix1 = max(ax1, bx1)
    iy1 = max(ay1, by1)
    ix2 = min(ax2, bx2)
    iy2 = min(ay2, by2)
    iw = max(0, ix2 - ix1)
    ih = max(0, iy2 - iy1)
    inter = iw * ih
    area_a = max(0, ax2 - ax1) * max(0, ay2 - ay1)
    area_b = max(0, bx2 - bx1) * max(0, by2 - by1)
    return inter / (area_a + area_b - inter + 1e-06)


def _box_wh_area(box):
    x1, y1, x2, y2 = box
    w = max(1.0, float(x2 - x1))
    h = max(1.0, float(y2 - y1))
    a = w * h
    return w, h, a


def _size_similarity_score(box1, box2):
    w1, h1, a1 = _box_wh_area(box1)
    w2, h2, a2 = _box_wh_area(box2)
    wr = max(w1, w2) / max(1e-06, min(w1, w2))
    hr = max(h1, h2) / max(1e-06, min(h1, h2))
    ar = max(a1, a2) / max(1e-06, min(a1, a2))
    score = abs(np.log(wr)) + abs(np.log(hr)) + 0.5 * abs(np.log(ar))
    return float(score)


def _is_similar_size(box1, box2, max_width_ratio=1.35, max_height_ratio=
    1.35, max_area_ratio=1.8):
    w1, h1, a1 = _box_wh_area(box1)
    w2, h2, a2 = _box_wh_area(box2)
    wr = max(w1, w2) / max(1e-06, min(w1, w2))
    hr = max(h1, h2) / max(1e-06, min(h1, h2))
    ar = max(a1, a2) / max(1e-06, min(a1, a2))
    return (wr <= max_width_ratio and hr <= max_height_ratio and ar <=
        max_area_ratio)


def _build_non_repeated_similar_pairs(boxes_xyxy: List[Tuple[int, int, int,
    int]], num_pairs: int, rng: random.Random, avoid_iou: float=0.2,
    max_width_ratio: float=1.35, max_height_ratio: float=1.35,
    max_area_ratio: float=1.8, random_topk: int=8):
    n = len(boxes_xyxy)
    if n < 2:
        return []
    candidates = []
    for i in range(n):
        for j in range(i + 1, n):
            bi = boxes_xyxy[i]
            bj = boxes_xyxy[j]
            if _iou_xyxy(bi, bj) > avoid_iou:
                continue
            if not _is_similar_size(bi, bj, max_width_ratio=max_width_ratio,
                max_height_ratio=max_height_ratio, max_area_ratio=
                max_area_ratio):
                continue
            score = _size_similarity_score(bi, bj)
            candidates.append((i, j, score))
    if len(candidates) == 0:
        return []
    candidates.sort(key=lambda x: x[2])
    pairs = []
    used = set()
    pool = candidates.copy()
    while len(pairs) < num_pairs and len(pool) > 0:
        topk = min(random_topk, len(pool))
        chosen_idx = rng.randrange(topk)
        i, j, score = pool[chosen_idx]
        if i not in used and j not in used:
            pairs.append((i, j))
            used.add(i)
            used.add(j)
        new_pool = []
        for a, b, s in pool:
            if a == i or a == j or b == i or b == j:
                continue
            new_pool.append((a, b, s))
        pool = new_pool
    return pairs


def _swap_box_contents_bgr(img_bgr: np.ndarray, boxes_xyxy: List, num_pairs:
    int, rng: random.Random, avoid_iou: float=0.2, max_width_ratio: float=
    1.35, max_height_ratio: float=1.35, max_area_ratio: float=1.8,
    random_topk: int=8) ->np.ndarray:
    H, W = img_bgr.shape[:2]
    boxes = []
    for b in boxes_xyxy:
        bb = _clamp_box_xyxy(b, W, H)
        if bb is not None:
            boxes.append(bb)
    if len(boxes) < 2:
        return img_bgr
    pairs = _build_non_repeated_similar_pairs(boxes_xyxy=boxes, num_pairs=
        num_pairs, rng=rng, avoid_iou=avoid_iou, max_width_ratio=
        max_width_ratio, max_height_ratio=max_height_ratio, max_area_ratio=
        max_area_ratio, random_topk=random_topk)
    if len(pairs) == 0:
        return img_bgr
    out = img_bgr.copy()
    for i, j in pairs:
        x1, y1, x2, y2 = boxes[i]
        xa, ya, xb, yb = boxes[j]
        patch_i = out[y1:y2, x1:x2].copy()
        patch_j = out[ya:yb, xa:xb].copy()
        patch_j_to_i = cv2.resize(patch_j, (x2 - x1, y2 - y1),
            interpolation=cv2.INTER_CUBIC)
        patch_i_to_j = cv2.resize(patch_i, (xb - xa, yb - ya),
            interpolation=cv2.INTER_CUBIC)
        out[y1:y2, x1:x2] = patch_j_to_i
        out[ya:yb, xa:xb] = patch_i_to_j
    return out


def strong_box_swap_tensor1(imgs: torch.Tensor, metas: List[dict],
    box_cache: OCRBoxCache, cfg=None, mean: Optional[np.ndarray]=None, std:
    Optional[np.ndarray]=None, p: float=1.0, num_pairs: int=1, avoid_iou:
    float=0.2, seed: int=0, boxes_key: str='boxes_xyxy_480',
    max_width_ratio: float=1.35, max_height_ratio: float=1.35,
    max_area_ratio: float=1.8, random_topk: int=8, debug_dir: Optional[str]
    =None, debug_max: int=8, debug_tag: str='step0', debug_height: int=900
    ) ->torch.Tensor:
    if mean is None or std is None:
        if cfg is not None:
            mean2, std2 = _parse_norm_cfg(cfg)
            mean = mean2 if mean is None else mean
            std = std2 if std is None else std
    device = imgs.device
    dtype = imgs.dtype
    normalized = _is_probably_normalized(imgs)
    if normalized:
        imgs01 = _denorm_to_01(imgs, mean, std)
    else:
        imgs01 = imgs.clamp(0, 1)
    out01 = imgs01.clone()
    _ensure_dir(debug_dir)
    saved_dbg = 0
    rng_global = random.Random(seed)
    B = out01.size(0)
    for i in range(B):
        if p < 1.0 and rng_global.random() > p:
            continue
        m = metas[i] if i < len(metas) else {}
        boxes = m.get(boxes_key, None)
        if not boxes and box_cache is not None:
            key = m.get('filename', None) or m.get('img_path', None) or m.get(
                'ori_filename', None)
            boxes = box_cache.get_boxes(key) if key else None
        if not boxes or len(boxes) < 2:
            continue
        img_bgr_orig = _tensor_to_u8_bgr(out01[i])
        rng = random.Random(seed + i * 9973)
        img_bgr2 = _swap_box_contents_bgr(img_bgr=img_bgr_orig, boxes_xyxy=
            boxes, num_pairs=num_pairs, rng=rng, avoid_iou=avoid_iou,
            max_width_ratio=max_width_ratio, max_height_ratio=
            max_height_ratio, max_area_ratio=max_area_ratio, random_topk=
            random_topk)
        out01[i] = _u8_bgr_to_tensor(img_bgr2, device=device, dtype=dtype)
        if debug_dir is not None and saved_dbg < int(debug_max):
            saved_dbg += 1
            name = m.get('filename', None) or m.get('img_path', None
                ) or f'img{i:02d}'
            base = os.path.basename(str(name))
            stem = os.path.splitext(base)[0]
            pref = f'{debug_tag}_b{i:02d}_{stem}'
            orig_boxes = _draw_boxes_bgr(img_bgr_orig, boxes, color=(0, 255,
                0), th=1)
            strong_boxes = _draw_boxes_bgr(img_bgr2, boxes, color=(0, 255, 
                0), th=1)
            comp = _hstack_resize_bgr(img_bgr_orig, img_bgr2, height=int(
                debug_height))
            cv2.imwrite(os.path.join(debug_dir, f'{pref}_orig.png'),
                img_bgr_orig)
            cv2.imwrite(os.path.join(debug_dir, f'{pref}_strong.png'), img_bgr2
                )
            cv2.imwrite(os.path.join(debug_dir, f'{pref}_compare.png'), comp)
            cv2.imwrite(os.path.join(debug_dir, f'{pref}_orig_boxes.png'),
                orig_boxes)
            cv2.imwrite(os.path.join(debug_dir, f'{pref}_strong_boxes.png'),
                strong_boxes)
    if normalized:
        out = _renorm_from_01(out01, mean, std)
    else:
        out = out01
    return out
