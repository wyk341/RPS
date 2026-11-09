import math
import os
import sys
import time
import csv
import gc
import copy
import json
from datetime import datetime
import psutil
import torch
import torch.nn.functional as F
from torch.cuda.amp import GradScaler
from rps.models import build_convertor
from rps.runtime import load_checkpoint
from rps.augment import OCRBoxCache, strong_box_swap_tensor1
from rps.losses.reduction import weighted_mean
from rps.losses.supervised import loss_sups
from rps.metrics.validation import evaluate_on_val
from rps.mps import (apply_insert_ops_to_seq_and_logits, build_row_gated_token_mask_v4, extract_tr_spans_table_aware, repair_pubtabnet_thead_tbody_by_insertion)
from rps.engine.state import (_save_ckpt_full, unwrap_img, unwrap_metas, capture_training_rng, restore_training_rng, capture_loader_states, restore_loader_states)
from rps.engine.logging import log_cfg_snapshot


PROC = psutil.Process(os.getpid())


def _amp_gradient_status(model):
    gradients = [(name, p.grad) for name, p in model.named_parameters() if 
        p.grad is not None]
    if not gradients:
        return dict(parameters=0, nan=0, inf=0, bad_parameters=[])
    finite = torch.stack([torch.isfinite(g).all() for _, g in gradients])
    bad_indices = (~finite).nonzero(as_tuple=False).flatten().tolist()
    bad = []
    for index in bad_indices:
        name, grad = gradients[index]
        bad.append(dict(name=name, module=name.rsplit('.', 1)[0], nan=int(
            torch.isnan(grad).sum()), inf=int(torch.isinf(grad).sum()),
            dtype=str(grad.dtype)))
    return dict(parameters=len(gradients), nan=sum(p['nan'] for p in bad),
        inf=sum(p['inf'] for p in bad), bad_parameters=bad)


def _managed_amp_backward(loss, model, optimizer, scaler, observer, phase,
    state):
    if not bool(torch.isfinite(loss)):
        raise RuntimeError('Nonfinite %s loss before AMP backward' % phase)
    rng = capture_training_rng()
    retries = 0
    while True:
        scale_before = scaler.get_scale()
        scaler.scale(loss).backward(retain_graph=True)
        scaled = _amp_gradient_status(model)
        scaler.unscale_(optimizer)
        unscaled = _amp_gradient_status(model)
        if not unscaled['parameters']:
            raise RuntimeError(
                'No parameter gradients after %s backward: epoch=%s step=%s' %
                (phase, state['epoch'], state['step']))
        if not unscaled['bad_parameters']:
            state['amp_retries'] = retries
            state['amp_scale_before'] = scale_before
            if observer is not None:
                observer(phase + '_backward', state)
            return scale_before
        found_inf = sum(float(value) for value in scaler.
            _per_optimizer_states[id(optimizer)]['found_inf_per_device'].
            values())
        diagnosis = dict(phase=phase, epoch=int(state['epoch']), step=int(
            state['step']), global_iter=int(state['global_iter']), loss=
            float(loss.detach()), scale_before=scale_before, attempt=
            retries + 1, pre_unscale=scaled, post_unscale=unscaled,
            found_inf=found_inf)
        if not scaler.is_enabled() or found_inf <= 0:
            raise RuntimeError('Nonfinite gradients without AMP overflow: ' +
                json.dumps(diagnosis))
        step_before = getattr(optimizer, '_step_count', None)
        scaler.step(optimizer)
        scaler.update()
        diagnosis['scale_after'] = scaler.get_scale()
        if getattr(optimizer, '_step_count', None
            ) != step_before or not 0 < diagnosis['scale_after'
            ] < scale_before:
            raise RuntimeError('AMP failed to skip/back off safely: ' +
                json.dumps(diagnosis))
        state['amp_overflow'] = diagnosis
        if observer is not None:
            observer('amp_overflow', state)
        print('[AMP-RETRY] ' + json.dumps(diagnosis))
        retries += 1
        if retries > 16:
            raise RuntimeError(
                'Persistent nonfinite gradients after 16 AMP retries: ' +
                json.dumps(diagnosis))
        optimizer.zero_grad(set_to_none=True)
        restore_training_rng(rng)


class DualLogger(object):

    def __init__(self, file_path):
        self.terminal = sys.stdout
        self.log = open(file_path, 'a', encoding='utf-8')
        self.log.write('\n\n' + '=' * 80 +
            f'\nSession started: {datetime.now()}\n' + '=' * 80 + '\n')

    def write(self, message):
        self.terminal.write(message)
        self.log.write(message)
        self.log.flush()

    def flush(self):
        self.terminal.flush()
        self.log.flush()


def top_percent_keep(w: torch.Tensor, keep_top_p: float=0.5):
    B = w.numel()
    if B == 0:
        return torch.zeros_like(w, dtype=torch.bool)
    keep_top_p = float(keep_top_p)
    keep_top_p = min(max(keep_top_p, 0.0), 1.0)
    if keep_top_p <= 0:
        return torch.zeros_like(w, dtype=torch.bool)
    if keep_top_p >= 1.0:
        return torch.ones_like(w, dtype=torch.bool)
    k = max(1, int(round(B * keep_top_p)))
    thr = torch.topk(w, k, largest=True).values.min()
    return w >= thr


def build_pseudo_targets_from_tids(t_ids, PAD, SOS):
    B, L = t_ids.shape
    sos = torch.full((B, 1), SOS, dtype=torch.long, device=t_ids.device)
    padded_targets = torch.cat([sos, t_ids], dim=1)
    bbox_masks = torch.zeros((B, L), dtype=torch.bool, device=t_ids.device)
    return {'padded_targets': padded_targets, 'bbox_masks': bbox_masks,
        'cell_padded_targets': [[] for _ in range(B)]}


@torch.no_grad()
def ema_update(teacher, student, momentum=0.999):
    for (n_t, p_t), (n_s, p_s) in zip(teacher.named_parameters(), student.
        named_parameters()):
        p_t.data.mul_(momentum).add_(p_s.data, alpha=1.0 - momentum)
    sd_s = student.state_dict()
    for n_t, b_t in teacher.named_buffers():
        if n_t in sd_s:
            b_t.data.copy_(sd_s[n_t])


@torch.no_grad()
def copy_model(teacher, student):
    teacher.load_state_dict(student.state_dict(), strict=True)


def set_requires_grad(model, flag: bool):
    for p in model.parameters():
        p.requires_grad_(flag)


def add_gaussian_noise_like(x: torch.Tensor, sigma=0.02, p=0.5, per_channel
    =True):
    if sigma <= 0:
        return x
    if p < 1.0 and torch.rand(1, device=x.device).item() > p:
        return x
    xf = x.float()
    mn = float(xf.detach().min().cpu())
    mx = float(xf.detach().max().cpu())
    if mn >= -1.5 and mx <= 1.5 and mn < -0.1:
        low, high = -1.0, 1.0
        scale = sigma
    elif mx <= 1.5 and mn >= 0.0:
        low, high = 0.0, 1.0
        scale = sigma
    else:
        low, high = 0.0, 255.0
        scale = sigma * 255.0
    if per_channel and xf.dim() == 4:
        noise = torch.randn((xf.size(0), xf.size(1), 1, 1), device=xf.
            device, dtype=xf.dtype) * scale
    else:
        noise = torch.randn_like(xf) * scale
    y = (xf + noise).clamp_(low, high)
    if not x.dtype.is_floating_point:
        return y.to(dtype=x.dtype)
    return y


def build_unsup_strong_view(img_unlabeled, metas_unlabeled, box_cache, cfg,
    global_iter, boxswap_p, boxswap_pairs, boxswap_avoid_iou):
    use_spp = bool(getattr(cfg, 'use_spp', True))
    if not use_spp:
        img_strong = add_gaussian_noise_like(img_unlabeled, sigma=0.03, p=
            1.0, per_channel=False)
        return img_strong
    base_seed = getattr(cfg, 'seed', None)
    if base_seed is None:
        base_seed = 0
    seed = int(base_seed) * 1000000 + int(global_iter)
    img_strong = strong_box_swap_tensor1(imgs=img_unlabeled, metas=
        metas_unlabeled, box_cache=box_cache, cfg=cfg, p=boxswap_p,
        num_pairs=boxswap_pairs, avoid_iou=boxswap_avoid_iou, seed=seed,
        max_width_ratio=1.5, max_height_ratio=1.5, max_area_ratio=2.2,
        random_topk=20)
    return img_strong


def maybe_apply_prm_repair(t_ids, t_logits, t_ids_s, t_logits_s, id2tok,
    tok2id, PAD, EOS, cfg):
    use_prm = bool(getattr(cfg, 'use_prm', True))
    if not use_prm:
        fill_mask = torch.zeros_like(t_ids, dtype=torch.bool)
        debug_inserted = None
        insert_ops = None
        return (t_ids, t_logits, fill_mask, debug_inserted, insert_ops,
            t_ids_s, t_logits_s)
    t_ids, t_logits, fill_mask, debug_inserted, insert_ops = (
        repair_pubtabnet_thead_tbody_by_insertion(t_ids=t_ids, t_logits=
        t_logits, id2tok=id2tok, tok2id=tok2id, PAD=PAD, EOS=EOS,
        fill_logits_to_pad=True, return_ops=True))
    if (t_logits_s is not None and t_ids_s is not None and insert_ops is not
        None):
        t_ids_s, t_logits_s = apply_insert_ops_to_seq_and_logits(t_ids_s,
            t_logits_s, insert_ops, PAD=PAD, fill_logits_to_pad=True)
    return (t_ids, t_logits, fill_mask, debug_inserted, insert_ops, t_ids_s,
        t_logits_s)


def train(model_student, data_loaders_labeled,
    data_loaders_unlabeled, data_loaders_val, optimizer, cfg):
    torch.backends.cudnn.benchmark = bool(getattr(cfg, 'cudnn_benchmark', True)
        )
    torch.backends.cudnn.deterministic = bool(getattr(cfg,
        'cudnn_deterministic', getattr(cfg, 'deterministic', False)))
    training_observer = getattr(cfg, 'training_observer', None)
    save_checkpoints = bool(getattr(cfg, 'save_checkpoints', True))
    save_final_checkpoint = bool(getattr(cfg, 'save_final_checkpoint', False))
    strict_checkpoint_loading = bool(getattr(cfg,
        'strict_checkpoint_loading', False))
    stop_after_epochs = int(getattr(cfg, 'stop_after_epochs', 0))
    best_metric_name = str(getattr(cfg, 'best_metric_name', 'score'))
    managed_training = bool(getattr(cfg, 'managed_training', False))
    stop_after_epoch = int(getattr(cfg, 'stop_after_epoch', 0))
    training_identity = getattr(cfg, 'training_identity', None)
    if managed_training and not isinstance(training_identity, dict):
        raise ValueError(
            'Managed training requires a frozen training_identity dictionary')
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    model_student = model_student.to(device)
    model_student.train()
    model_teacher = copy.deepcopy(model_student).to(device)
    model_teacher.eval()
    set_requires_grad(model_teacher, False)
    base_dir = getattr(cfg, 'work_dir', './work_dir')
    timestamp = datetime.now().strftime('%Y_%m_%d_%H%M')
    work_dir = base_dir if bool(getattr(cfg, 'work_dir_exact', False)
        ) else os.path.join(base_dir, timestamp)
    os.makedirs(work_dir, exist_ok=True)
    cfg.work_dir = work_dir
    log_path = os.path.join(work_dir, str(getattr(cfg, 'train_log_name',
        'train_log.txt')))
    sys.stdout = DualLogger(log_path)
    sys.stderr = sys.stdout
    print(f'✔ work_dir: {work_dir}')
    print(f'✔ log_path: {log_path}')
    csv_path = os.path.join(work_dir, str(getattr(cfg, 'train_stats_name',
        'train_stats.csv')))
    if not os.path.exists(csv_path):
        with open(csv_path, 'w', newline='') as f:
            writer = csv.writer(f)
            writer.writerow(['epoch', 'avg_loss_total', 'avg_loss_sup',
                'avg_loss_unsup', 'avg_st_sup', 'avg_st_unsup', 'lr',
                'val_TEDS', 'val_precision', 'val_recall', 'val_F1',
                'val_CER', 'val_score'])
    resume_from = getattr(cfg, 'resume_from', None)
    load_from = getattr(cfg, 'load_from', None)
    pretrained = getattr(cfg, 'pretrained', None)
    start_epoch = int(getattr(cfg, 'start_epoch', 1))
    best_metric = float('-inf')
    best_epoch = None
    global_iter = 0
    resume_state = None
    ckpt_obj = None
    update_counts = dict(supervised=0, unsupervised=0, interleave=0)
    cumulative_optimizer_steps = 0
    base_lrs = [float(group['lr']) for group in optimizer.param_groups]

    def rigorous_load(model, path, device):
        print(f'[CKPT] Loading from {path} ...')
        checkpoint = load_checkpoint(model, path, map_location=device,
            strict=strict_checkpoint_loading)
        state_dict = checkpoint['state_dict'
            ] if 'state_dict' in checkpoint else checkpoint
        model_keys = set(model.state_dict().keys())
        ckpt_keys = set(state_dict.keys())
        cls_weight = 'decoder.cls_fc.weight'
        if cls_weight in model_keys:
            found = False
            for k in ckpt_keys:
                if k.endswith(cls_weight):
                    found = True
                    break
            if found:
                print(f"[CKPT] ✅ Classifier head '{cls_weight}' found matches."
                    )
            else:
                print(
                    f"[CKPT] ❌ WARNING: Classifier head '{cls_weight}' NOT found in checkpoint! Model head is RANDOM."
                    )
        return checkpoint
    resume_path = None
    if managed_training and resume_from:
        if not os.path.isfile(resume_from):
            raise FileNotFoundError(resume_from)
        print(f'[CKPT] Strict managed resume from: {resume_from}')
        ckpt_obj = torch.load(resume_from, map_location='cpu')
        required = ('state_dict', 'teacher_state_dict', 'optimizer',
            'scaler', 'scheduler', 'global_iter', 'meta', 'training_state')
        if not isinstance(ckpt_obj, dict) or any(key not in ckpt_obj for
            key in required):
            raise ValueError(
                'Managed resume requires a complete managed FULL checkpoint')
        resume_state = ckpt_obj['training_state']
        state_keys = ('schema_version', 'identity', 'completed_epoch',
            'total_epochs', 'cumulative_optimizer_steps', 'update_counts',
            'best_metric', 'best_epoch', 'base_lrs', 'global_iter', 'rng',
            'loaders', 'scheduler_step_complete', 'optimizer_extra',
            'optimizer_state_ids')
        if not isinstance(resume_state, dict) or any(key not in
            resume_state for key in state_keys) or resume_state[
            'schema_version'] != 2:
            raise ValueError('Invalid managed training_state'
                )
        if resume_state['identity'] != training_identity:
            raise ValueError(
                'Checkpoint training identity differs from the requested protocol/variant/data'
                )
        if resume_state['scheduler_step_complete'] is not True:
            raise ValueError(
                'Checkpoint was not committed after the epoch scheduler boundary'
                )
        for state_key in ('state_dict', 'teacher_state_dict'):
            state_dict = ckpt_obj[state_key]
            if not isinstance(state_dict, dict):
                raise ValueError(f'Checkpoint {state_key} is invalid')
            for name, value in state_dict.items():
                if not torch.is_tensor(value) or value.is_floating_point(
                    ) and not torch.isfinite(value).all():
                    raise ValueError(
                        f'Checkpoint {state_key} parameter {name} is invalid/non-finite'
                        )
        model_student.load_state_dict(ckpt_obj['state_dict'], strict=True)
        model_teacher.load_state_dict(ckpt_obj['teacher_state_dict'],
            strict=True)
        resume_path = resume_from
    elif resume_from and os.path.isfile(resume_from):
        print(f'[CKPT] Resume from: {resume_from}')
        rigorous_load(model_student, resume_from, device)
        copy_model(model_teacher, model_student)
        resume_path = resume_from
    elif load_from and os.path.isfile(load_from):
        print(f'[CKPT] Load checkpoint from: {load_from}')
        rigorous_load(model_student, load_from, device)
        copy_model(model_teacher, model_student)
    elif pretrained and os.path.isfile(pretrained):
        print(f'[CKPT] Load pretrained from: {pretrained}')
        rigorous_load(model_student, pretrained, device)
        copy_model(model_teacher, model_student)
    else:
        print(
            '[CKPT] ⚠ Train from scratch! (Check if you forgot to set pretrained path)'
            )
        copy_model(model_teacher, model_student)
    scaler = GradScaler(enabled=True)
    convertor = build_convertor(cfg.get('label_convertor'))
    ocr_jsonl = cfg.ocr_jsonl_trainpad
    box_cache = OCRBoxCache(ocr_jsonl)
    boxswap_p = float(cfg.boxswap_p)
    boxswap_pairs = int(cfg.boxswap_pairs)
    boxswap_avoid_iou = float(cfg.boxswap_avoid_iou)
    cfg.strong_noise_sigma = float(getattr(cfg, 'strong_noise_sigma', 0.03))
    cfg.grad_clip = float(getattr(cfg, 'grad_clip', 5.0))
    cfg.ema_in_sup = bool(getattr(cfg, 'ema_in_sup', False))
    cfg.unsup_steps_per_epoch = int(getattr(cfg, 'unsup_steps_per_epoch', -1))
    cfg.lambda_u = float(getattr(cfg, 'lambda_u', 0.2))
    cfg.unsup_warmup_epochs = int(getattr(cfg, 'unsup_warmup_epochs', 10))
    cfg.unsup_max_len = int(getattr(cfg, 'unsup_max_len', 0))
    if hasattr(convertor, 'padding_idx'):
        cfg.PAD = convertor.padding_idx
    if hasattr(convertor, 'start_idx'):
        cfg.SOS = convertor.start_idx
    if hasattr(convertor, 'end_idx'):
        cfg.EOS = convertor.end_idx
    if hasattr(convertor, 'padding_idx_cell'):
        cfg.PAD_CELL = convertor.padding_idx_cell
    if hasattr(convertor, 'start_idx_cell'):
        cfg.SOS_CELL = convertor.start_idx_cell
    if hasattr(convertor, 'end_idx_cell'):
        cfg.EOS_CELL = convertor.end_idx_cell
    cfg.use_spp = bool(getattr(cfg, 'use_spp', True))
    cfg.use_prm = bool(getattr(cfg, 'use_prm', False))
    sup_loader = data_loaders_labeled[0]
    unsup_loader = data_loaders_unlabeled[0]
    val_loader = data_loaders_val[0] if data_loaders_val else None
    mode = str(getattr(cfg, 'train_mode', 'supervised')).lower()
    if mode == 'supervised':
        total_epochs = int(getattr(cfg, 'sup_epochs', 200))
    else:
        total_epochs = int(getattr(cfg, 'semi_epochs', 300))
    val_interval = int(getattr(cfg, 'val_interval', 1))
    if val_interval <= 0:
        raise ValueError('val_interval must be a positive epoch interval')

    def expected_update_counts(completed_epoch):
        sup_per_epoch = int(getattr(cfg, 'sup_steps_per_epoch', 1000))
        stage_b_epochs = max(0, completed_epoch - int(getattr(cfg,
            'sup_only_epochs', 0)))
        target = int(getattr(cfg, 'unsup_target_updates', 300))
        if target <= 0:
            raise ValueError(
                'Managed training requires an explicit positive unsup_target_updates'
                )
        interleave = int(getattr(cfg, 'interleave_sup_every', 30))
        return dict(supervised=completed_epoch * sup_per_epoch,
            unsupervised=stage_b_epochs * target, interleave=stage_b_epochs *
            (target // interleave if interleave > 0 else 0))
    if managed_training:
        if stop_after_epochs:
            raise ValueError(
                'Managed training uses absolute stop_after_epoch, not stop_after_epochs'
                )
        if not 1 <= (stop_after_epoch or total_epochs) <= total_epochs:
            raise ValueError(
                'stop_after_epoch must lie within the unchanged total epoch budget'
                )
        if resume_state is not None:
            completed_epoch = resume_state['completed_epoch']
            if (type(completed_epoch) is not int or not 1 <=
                completed_epoch <= total_epochs or resume_state[
                'total_epochs'] != total_epochs or ckpt_obj['meta'].get(
                'epoch') != completed_epoch):
                raise ValueError(
                    'Checkpoint completed_epoch/meta/total_epochs are inconsistent'
                    )
            expected_counts = expected_update_counts(completed_epoch)
            if not isinstance(resume_state['update_counts'], dict) or any(
                type(value) is not int for value in resume_state[
                'update_counts'].values()) or resume_state['update_counts'
                ] != expected_counts:
                raise ValueError(
                    'Checkpoint optimizer update counts do not match completed epochs'
                    )
            update_counts = dict(resume_state['update_counts'])
            cumulative_optimizer_steps = sum(update_counts.values())
            if type(ckpt_obj['global_iter']) is not int or type(resume_state
                ['global_iter']) is not int or type(resume_state[
                'cumulative_optimizer_steps']) is not int:
                raise ValueError('Checkpoint update counters must be integers')
            global_iter = ckpt_obj['global_iter']
            if resume_state['cumulative_optimizer_steps'
                ] != cumulative_optimizer_steps or resume_state['global_iter'
                ] != global_iter or global_iter != cumulative_optimizer_steps:
                raise ValueError(
                    'Checkpoint optimizer/global update counters are inconsistent'
                    )
            start_epoch = completed_epoch + 1
            if (stop_after_epoch or total_epochs) < start_epoch and not (
                completed_epoch == total_epochs and (stop_after_epoch or
                total_epochs) == total_epochs):
                raise ValueError(
                    'Requested stop epoch is already completed in the checkpoint'
                    )
            best_metric = float(resume_state['best_metric'])
            best_epoch = resume_state['best_epoch']
            if best_epoch is not None and (type(best_epoch) is not int or 
                not 1 <= best_epoch <= completed_epoch or not math.isfinite
                (best_metric)):
                raise ValueError('Checkpoint best metric/epoch is invalid')
            if best_epoch is None and best_metric != float('-inf'):
                raise ValueError(
                    'Checkpoint best metric has no corresponding best epoch')
            restored_base_lrs = [float(value) for value in resume_state[
                'base_lrs']]
            if restored_base_lrs != base_lrs:
                raise ValueError(
                    'Checkpoint warmup base learning rates differ from the configured optimizer'
                    )
            base_lrs = restored_base_lrs
        elif start_epoch != 1:
            raise ValueError('New managed training must start at epoch 1')
        capture_loader_states(dict(sup=sup_loader, unsup=unsup_loader, val=
            val_loader))
    print(f'[CFG] total_epochs={total_epochs}')
    print(
        f'[CFG] sup_loader  : {len(sup_loader.dataset)} samples, steps/epoch={len(sup_loader)}'
        )
    print(
        f'[CFG] unsup_loader: {len(unsup_loader.dataset)} samples, steps/epoch={len(unsup_loader)}'
        )
    log_cfg_snapshot(cfg)
    print(f'[ABLATION] use_spp={cfg.use_spp} | use_prm={cfg.use_prm}')
    if val_loader is not None:
        print(
            f'[CFG] val_loader  : {len(val_loader.dataset)} samples, steps/val={len(val_loader)}'
            )
    mode = str(getattr(cfg, 'train_mode', 'semi')).lower()
    if mode in ('semi', 'ssl', 'semisupervised', 'semi_supervised'):
        lr_cfg = getattr(cfg, 'lr_config_semi', getattr(cfg, 'lr_config', None)
            )
    else:
        lr_cfg = getattr(cfg, 'lr_config_sup', getattr(cfg, 'lr_config', None))
    warmup_policy = None
    warmup_iters = 0
    warmup_ratio = 1.0
    scheduler = None
    if lr_cfg is not None:
        warmup_policy = lr_cfg.get('warmup', None)
        warmup_iters = int(lr_cfg.get('warmup_iters', 0))
        warmup_ratio = float(lr_cfg.get('warmup_ratio', 1.0))
        policy = str(lr_cfg.get('policy', 'cosineannealing')).lower()
        if policy == 'cosineannealing':
            min_lr = float(lr_cfg.get('min_lr', 0.0))
            scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer,
                T_max=total_epochs, eta_min=min_lr)
        elif policy == 'step':
            milestones = lr_cfg.get('step', [])
            gamma = float(lr_cfg.get('gamma', 0.1))
            scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer,
                milestones=milestones, gamma=gamma)
        else:
            print(f"⚠ Unknown lr policy '{policy}', use constant lr")

    def _is_full_ckpt(obj):
        return isinstance(obj, dict) and ('optimizer' in obj or 'scaler' in
            obj or 'scheduler' in obj or 'global_iter' in obj)
    if managed_training and resume_state is not None:
        if ckpt_obj['optimizer'] is None or ckpt_obj['scaler'] is None:
            raise ValueError(
                'Managed checkpoint is missing optimizer or GradScaler state')
        optimizer_state = ckpt_obj['optimizer']
        if not isinstance(optimizer_state, dict) or not optimizer_state.get(
            'state') or not isinstance(optimizer_state.get('param_groups'),
            list) or len(optimizer_state['param_groups']) != len(optimizer.
            param_groups):
            raise ValueError(
                'Trained checkpoint optimizer state is invalid')
        if not isinstance(resume_state['optimizer_state_ids'], list) or sorted(
            optimizer_state['state']) != resume_state['optimizer_state_ids']:
            raise ValueError(
                'Checkpoint optimizer parameter state IDs differ from the committed signature'
                )
        for current_group, saved_group in zip(optimizer.param_groups,
            optimizer_state['param_groups']):
            ignored = {'params', 'lr', 'initial_lr'}
            if {key: value for key, value in current_group.items() if key
                 not in ignored} != {key: value for key, value in
                saved_group.items() if key not in ignored} or len(current_group
                ['params']) != len(saved_group.get('params', [])):
                raise ValueError(
                    'Checkpoint optimizer parameter groups/hyperparameters differ'
                    )
            if not isinstance(saved_group.get('lr'), (int, float)
                ) or not math.isfinite(saved_group['lr']) or saved_group['lr'
                ] <= 0:
                raise ValueError(
                    'Checkpoint optimizer learning rate is invalid')
            if hasattr(optimizer, 'radam_buffer'):
                for parameter, state_id in zip(current_group['params'],
                    saved_group['params']):
                    parameter_state = optimizer_state['state'].get(state_id)
                    if parameter_state is None:
                        continue
                    step_count = parameter_state.get('step')
                    if (type(step_count) is not int or not 0 < step_count <=
                        cumulative_optimizer_steps):
                        raise ValueError(
                            'Ranger parameter update counter is invalid')
                    for name in ('exp_avg', 'exp_avg_sq', 'slow_buffer'):
                        value = parameter_state.get(name)
                        expected_dtype = (parameter.dtype if name ==
                            'slow_buffer' else torch.float32)
                        if (not torch.is_tensor(value) or value.shape !=
                            parameter.shape or value.dtype !=
                            expected_dtype or not torch.isfinite(value).all()):
                            raise ValueError(
                                f'Ranger parameter state {name} is missing/invalid'
                                )
        optimizer.load_state_dict(ckpt_obj['optimizer'])
        extra = resume_state['optimizer_extra']
        expected_extra_keys = ('alpha', 'k', 'N_sma_threshhold', 'gc_loc',
            'use_gc', 'gc_conv_only')
        if not isinstance(extra, dict):
            raise ValueError('Managed checkpoint optimizer_extra is invalid')
        for name in expected_extra_keys:
            if hasattr(optimizer, name):
                if name not in extra or extra[name] != getattr(optimizer, name
                    ):
                    raise ValueError(
                        f'Optimizer runtime attribute {name} differs from checkpoint'
                        )
        if hasattr(optimizer, '_step_count'):
            if type(extra.get('_step_count')) is not int or extra['_step_count'
                ] != cumulative_optimizer_steps:
                raise ValueError(
                    'Optimizer scheduler-wrapper step counter is inconsistent')
            optimizer._step_count = extra['_step_count']
        if hasattr(optimizer, 'radam_buffer'):
            if 'radam_buffer' not in extra:
                raise ValueError('Checkpoint is missing Ranger radam_buffer')
            buffer = extra['radam_buffer']
            if not isinstance(buffer, list) or len(buffer) != 10:
                raise ValueError('Ranger radam_buffer length is invalid')
            for entry in buffer:
                if not isinstance(entry, list) or len(entry) != 3:
                    raise ValueError('Ranger radam_buffer entry is invalid')
                if entry == [None, None, None]:
                    continue
                if type(entry[0]) is not int or not 0 < entry[0
                    ] <= cumulative_optimizer_steps or any(type(value) not in
                    (float, int) or not math.isfinite(value) for value in
                    entry[1:]) or entry[2] <= 0:
                    raise ValueError('Ranger radam_buffer values are invalid')
            optimizer.radam_buffer = copy.deepcopy(extra['radam_buffer'])
        scaler_state = ckpt_obj['scaler']
        if scaler.is_enabled():
            scaler_keys = {'scale', 'growth_factor', 'backoff_factor',
                'growth_interval', '_growth_tracker'}
            if not isinstance(scaler_state, dict) or set(scaler_state
                ) != scaler_keys or not math.isfinite(scaler_state['scale']
                ) or scaler_state['scale'] <= 0 or not math.isfinite(
                scaler_state['growth_factor']) or scaler_state['growth_factor'
                ] <= 1 or not 0 < scaler_state['backoff_factor'] < 1 or type(
                scaler_state['growth_interval']) is not int or scaler_state[
                'growth_interval'] <= 0 or type(scaler_state['_growth_tracker']
                ) is not int or scaler_state['_growth_tracker'] < 0:
                raise ValueError('Checkpoint GradScaler state is invalid')
        scaler.load_state_dict(ckpt_obj['scaler'])
        if scheduler is not None:
            if ckpt_obj['scheduler'] is None:
                raise ValueError(
                    'Managed checkpoint is missing scheduler state')
            saved_scheduler = ckpt_obj['scheduler']
            if isinstance(scheduler, torch.optim.lr_scheduler.CosineAnnealingLR
                ) and (saved_scheduler.get('T_max') != total_epochs or 
                saved_scheduler.get('eta_min') != scheduler.eta_min):
                raise ValueError(
                    'Checkpoint cosine scheduler no longer matches the full training plan'
                    )
            expected_scheduler_steps = sum(warmup_iters == 0 or sum(
                expected_update_counts(ep).values()) >= warmup_iters for ep in
                range(1, resume_state['completed_epoch'] + 1))
            if saved_scheduler.get('last_epoch'
                ) != expected_scheduler_steps or saved_scheduler.get(
                '_step_count') != expected_scheduler_steps + 1:
                raise ValueError(
                    'Checkpoint scheduler step counters disagree with completed epochs/warmup'
                    )
            saved_lrs = [float(group['lr']) for group in optimizer.param_groups
                ]
            if expected_scheduler_steps > 0:
                if saved_scheduler.get('_last_lr') != saved_lrs:
                    raise ValueError(
                        'Checkpoint optimizer LR differs from the committed scheduler LR'
                        )
            elif warmup_policy is not None and warmup_iters > 0 and global_iter <= warmup_iters:
                factor = warmup_ratio + global_iter / float(max(1,
                    warmup_iters)) * (1.0 - warmup_ratio)
                if saved_lrs != [(initial_lr * factor) for initial_lr in
                    base_lrs]:
                    raise ValueError(
                        'Checkpoint optimizer LR differs from the completed warmup position'
                        )
            scheduler.load_state_dict(ckpt_obj['scheduler'])
        elif ckpt_obj['scheduler'] is not None:
            raise ValueError(
                'Checkpoint scheduler differs from configured scheduler')
        print(
            f"[CKPT] Managed resume verified: start_epoch={start_epoch}, updates={cumulative_optimizer_steps}, best={best_metric} @ {best_epoch}, lr={optimizer.param_groups[0]['lr']:.6e}"
            )
    elif resume_path is not None:
        ckpt_obj = torch.load(resume_path, map_location=device)
        if _is_full_ckpt(ckpt_obj):
            print(
                '[CKPT] Detected FULL checkpoint, restoring optimizer/scaler/scheduler/global_iter...'
                )
            sd = ckpt_obj.get('state_dict', ckpt_obj)
            model_student.load_state_dict(sd, strict=strict_checkpoint_loading)
            tsd = ckpt_obj.get('teacher_state_dict', None)
            if tsd is not None:
                model_teacher.load_state_dict(tsd, strict=
                    strict_checkpoint_loading)
            else:
                copy_model(model_teacher, model_student)
            if optimizer is not None and ckpt_obj.get('optimizer', None
                ) is not None:
                optimizer.load_state_dict(ckpt_obj['optimizer'])
            if ckpt_obj.get('scaler', None) is not None:
                scaler.load_state_dict(ckpt_obj['scaler'])
            if scheduler is not None and ckpt_obj.get('scheduler', None
                ) is not None:
                scheduler.load_state_dict(ckpt_obj['scheduler'])
            global_iter = int(ckpt_obj.get('global_iter', 0))
            if isinstance(ckpt_obj.get('meta', None), dict
                ) and 'epoch' in ckpt_obj['meta']:
                start_epoch = int(ckpt_obj['meta']['epoch']) + 1
            print(
                f"[CKPT] ✅ full resume ok: start_epoch={start_epoch}, global_iter={global_iter}, lr={optimizer.param_groups[0]['lr']:.6e}"
                )
        else:
            print(
                '[CKPT] ⚠ resume_from is NOT a full checkpoint (only weights loaded).'
                )
    base_lr = float(optimizer.param_groups[0]['lr'])
    if not managed_training:
        base_lrs = [base_lr for _ in optimizer.param_groups]

    def apply_warmup(step_idx: int):
        if warmup_policy is None or warmup_iters <= 0:
            return
        if step_idx > warmup_iters:
            return
        alpha = step_idx / float(max(1, warmup_iters))
        factor = warmup_ratio + alpha * (1.0 - warmup_ratio)
        for g, initial_lr in zip(optimizer.param_groups, base_lrs):
            g['lr'] = initial_lr * factor
    ckpt_subdir = os.path.join(cfg.work_dir, 'ckpt')
    if save_checkpoints or save_final_checkpoint:
        os.makedirs(ckpt_subdir, exist_ok=True)
    if (managed_training and resume_state is not None and best_epoch is not
        None):
        best_path = os.path.join(ckpt_subdir, 'best_full.pth')
        best_error = None
        try:
            saved_best = torch.load(best_path, map_location='cpu')
            best_state = saved_best['training_state']
            best_meta = saved_best['meta']
            metric_key = ('val_TEDS' if best_metric_name == 'TEDS' else
                'val_score')
            if best_state.get('schema_version') != 2 or best_state.get(
                'identity') != training_identity or best_state.get(
                'completed_epoch') != best_epoch or best_state.get('best_epoch'
                ) != best_epoch or best_state.get('best_metric'
                ) != best_metric or best_meta.get('epoch'
                ) != best_epoch or best_meta.get(metric_key
                ) != best_metric or any(key not in saved_best for key in (
                'state_dict', 'teacher_state_dict', 'optimizer',
                'scheduler', 'scaler', 'global_iter')):
                raise ValueError(
                    'Best checkpoint identity/epoch/metric is inconsistent')
            if best_epoch == resume_state['completed_epoch']:
                for state_key in ('state_dict', 'teacher_state_dict'):
                    if set(saved_best[state_key]) != set(ckpt_obj[state_key]
                        ) or any(not torch.equal(value, ckpt_obj[state_key]
                        [name]) for name, value in saved_best[state_key].
                        items()):
                        raise ValueError(
                            'Best model tensors differ from the committed best epoch'
                            )
            del saved_best
        except Exception as error:
            best_error = str(error)
            if 'saved_best' in locals():
                del saved_best
        if best_error is not None:
            if best_epoch != resume_state['completed_epoch']:
                raise ValueError(
                    f'Historical best checkpoint is missing/corrupt ({best_path}): {best_error}; latest cannot reconstruct an earlier best epoch'
                    )
            _save_ckpt_full(model_student, model_teacher, optimizer, scaler,
                scheduler, best_path, meta=ckpt_obj['meta'], global_iter=
                global_iter, training_state=resume_state)
            print(
                f'[CKPT] Atomically recovered best from committed latest: {best_path}'
                )
    if managed_training and resume_state is not None and resume_state[
        'completed_epoch'] == total_epochs and save_final_checkpoint:
        final_path = os.path.join(ckpt_subdir, 'final_full.pth')
        final_valid = False
        try:
            saved_final = torch.load(final_path, map_location='cpu')
            final_state = saved_final['training_state']
            final_valid = final_state.get('schema_version'
                ) == 2 and final_state.get('identity'
                ) == training_identity and final_state.get('completed_epoch'
                ) == total_epochs and saved_final['meta'].get('epoch'
                ) == total_epochs and final_state.get(
                'cumulative_optimizer_steps'
                ) == cumulative_optimizer_steps and saved_final.get(
                'global_iter') == global_iter
            if final_valid:
                for state_key in ('state_dict', 'teacher_state_dict'):
                    final_valid = final_valid and (set(saved_final[
                        state_key]) == set(ckpt_obj[state_key]) and all(
                        torch.equal(value, ckpt_obj[state_key][name]) for 
                        name, value in saved_final[state_key].items()))
            del saved_final
        except Exception:
            if 'saved_final' in locals():
                del saved_final
        if not final_valid:
            _save_ckpt_full(model_student, model_teacher, optimizer, scaler,
                scheduler, final_path, meta=ckpt_obj['meta'], global_iter=
                global_iter, training_state=resume_state)
            print(
                f'[CKPT] Atomically recovered final from completed latest: {final_path}'
                )
        print(
            '[CKPT] The complete training plan is already committed; no training updates remain.'
            )
    LOG_EVERY = int(getattr(cfg, 'log_every', 50))
    checkpoint_interval = int(getattr(cfg, 'checkpoint_interval', 2))
    ema_m_sup = float(getattr(cfg, 'ema_momentum_sup', 0.999))
    print(f'[CFG] EMA momentum sup: {ema_m_sup:.4f}')
    noise_sigma = cfg.strong_noise_sigma
    grad_clip = cfg.grad_clip
    ema_in_sup = cfg.ema_in_sup
    unsup_steps_limit = cfg.unsup_steps_per_epoch

    def _save_ckpt_light_fp16(m, save_path, meta=None):
        state = {}
        for k, v in m.state_dict().items():
            t = v.detach().cpu()
            if t.is_floating_point():
                t = t.half()
            state[k] = t
        torch.save({'state_dict': state, 'meta': meta or {}}, save_path,
            _use_new_zipfile_serialization=False)
        del state
    torch.backends.cuda.matmul.allow_tf32 = bool(getattr(cfg, 'allow_tf32',
        True))
    torch.backends.cudnn.allow_tf32 = bool(getattr(cfg, 'allow_tf32', True))
    if training_observer is not None:
        training_observer('initialized', locals())
    if managed_training and resume_state is not None:
        restore_loader_states(dict(sup=sup_loader, unsup=unsup_loader, val=
            val_loader), resume_state['loaders'])
        restore_training_rng(resume_state['rng'])
        del ckpt_obj

    def record_optimizer_update(kind, scale_before):
        nonlocal cumulative_optimizer_steps
        if managed_training and scaler.get_scale() >= scale_before:
            update_counts[kind] += 1
            cumulative_optimizer_steps += 1
    if val_loader is not None and bool(getattr(cfg,
        'validation_before_train', True)) and not (managed_training and 
        resume_state is not None):
        print('\n' + '=' * 80)
        print(
            f'[Baseline Validation] Checking pretrained model performance... (use_spp={cfg.use_spp}, use_prm={cfg.use_prm})'
            )
        print('=' * 80)
        with torch.no_grad():
            _ = evaluate_on_val(model_teacher, val_loader, convertor,
                max_batches=len(val_loader), compute_bbox=False,
                compute_teds=True, compute_cer=False)
        print('=' * 80 + '\n')
    epochs_completed = 0
    for epoch in range(start_epoch, total_epochs + 1):
        t0 = time.time()
        model_student.train()
        model_student.apply(lambda m: m.eval() if isinstance(m, (torch.nn.
            BatchNorm2d, torch.nn.BatchNorm1d)) else None)
        model_teacher.eval()
        sup_loss_sum = sup_steps = 0
        unsup_loss_sum = unsup_steps = 0
        sup_st_sum = 0.0
        unsup_st_sum = 0.0
        print('\n' + '=' * 80)
        print(
            f"[{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}] Epoch {epoch}/{total_epochs} (Decoupled Semi, Structure-only)"
            )
        print('=' * 80)
        print('\n' + '-' * 80)
        print('[Stage A] Supervised')
        print('-' * 80)
        prev_iter_end = time.time()
        max_sup_iters = int(getattr(cfg, 'sup_steps_per_epoch', 1000))
        sup_iter_start = 1
        sup_iter = iter(sup_loader)
        for step in range(sup_iter_start, max_sup_iters + 1):
            iter_start = time.time()
            global_iter += 1
            apply_warmup(global_iter)
            try:
                batch_sup = next(sup_iter)
            except StopIteration:
                sup_iter = iter(sup_loader)
                batch_sup = next(sup_iter)
            optimizer.zero_grad(set_to_none=True)
            img_sup = unwrap_img(batch_sup['img']).to(device, non_blocking=True
                )
            metas_sup = unwrap_metas(batch_sup['img_metas'])
            targets_sup = convertor.str_bbox_format(metas_sup)
            with torch.cuda.amp.autocast(enabled=True):
                out_sup = model_student.forward_train(img_sup, metas_sup,
                    collect_va=False, strong_view=False, train_mode=True,
                    targets_dict=targets_sup)
                loss_tf_sup, _, _, loss_main_sup = loss_sups(out_sup, cfg,
                    enable_bbox=False, enable_cell=False, w_cell=0.0)
                sup_loss = loss_main_sup
            if not torch.isfinite(sup_loss):
                print(
                    f'[FATAL] sup_loss is not finite at ep={epoch} iter={step}: {sup_loss.item()}'
                    )
                break
            if managed_training:
                old_scale = _managed_amp_backward(sup_loss, model_student,
                    optimizer, scaler, training_observer, 'supervised',
                    locals())
            else:
                scaler.scale(sup_loss).backward()
                scaler.unscale_(optimizer)
                old_scale = scaler.get_scale()
                if training_observer is not None:
                    training_observer('supervised_backward', locals())
            if step == sup_iter_start:
                with torch.no_grad():
                    bad = []
                    total = 0.0
                    for name, p in model_student.named_parameters():
                        if p.grad is None:
                            continue
                        g = p.grad.data
                        if not torch.isfinite(g).all():
                            bad.append(name)
                            if len(bad) >= 5:
                                break
                        else:
                            total += g.norm(2).item() ** 2
            torch.nn.utils.clip_grad_norm_(model_student.parameters(),
                max_norm=grad_clip)
            scaler.step(optimizer)
            scaler.update()
            record_optimizer_update('supervised', old_scale)
            if ema_in_sup:
                ema_update(model_teacher, model_student, momentum=ema_m_sup)
            sup_steps += 1
            sup_loss_sum += float(sup_loss.detach().cpu())
            sup_st_sum += float(loss_tf_sup.detach().cpu())
            if training_observer is not None:
                training_observer('supervised_updated', locals())
            if step % LOG_EVERY == 0 or step == max_sup_iters:
                lr_now = optimizer.param_groups[0]['lr']
                print(
                    f'[SUP] ep={epoch} iter={step}/{max_sup_iters} lr={lr_now:.6e} loss={sup_loss_sum / max(1, sup_steps):.4f} st={sup_st_sum / max(1, sup_steps):.4f} | iter={time.time() - iter_start:.3f}s load={iter_start - prev_iter_end:.3f}s'
                    )
                if torch.cuda.is_available():
                    mem = torch.cuda.max_memory_allocated() / 1024 ** 2
                    print(f'[CUDA] max_alloc={mem:.1f} MB')
            prev_iter_end = time.time()
            del img_sup, metas_sup, targets_sup, out_sup, sup_loss, loss_tf_sup
            if managed_training:
                del loss_main_sup
                locals()
        sup_only_epochs = int(getattr(cfg, 'sup_only_epochs', 0))
        skip_stage_b = epoch <= sup_only_epochs
        if skip_stage_b:
            print(f'\n[Stage B] SKIPPED (sup_only_epochs={sup_only_epochs})\n')
            avg_unsup = 0.0
            avg_st_unsup = 0.0
        else:
            debug_unsup = bool(getattr(cfg, 'debug_unsup', True))
            debug_every = int(getattr(cfg, 'debug_unsup_every', LOG_EVERY))
            print('\n' + '-' * 80)
            print(
                f'[Stage B] Unsupervised (teacher weak -> pseudo ; student strong -> CE) | use_spp={cfg.use_spp} use_prm={cfg.use_prm}'
                )
            print('-' * 80)
            model_student.train()
            model_student.apply(lambda m: m.eval() if isinstance(m, (torch.
                nn.BatchNorm2d, torch.nn.BatchNorm1d)) else None)
            model_teacher.eval()
            set_requires_grad(model_teacher, False)
            PAD = cfg.PAD
            EOS = cfg.EOS
            SOS = cfg.SOS
            id2tok = convertor.idx2char
            tok2id = convertor.char2idx
            lambda_u = cfg.lambda_u
            warmup_u = cfg.unsup_warmup_epochs
            max_len_u = cfg.unsup_max_len
            pseudo_max_len = None if max_len_u <= 0 else int(max_len_u)
            student_strong_view = bool(getattr(cfg,
                'unsup_student_strong_view', True))
            total_steps_unsup = len(unsup_loader)
            max_steps_unsup = (total_steps_unsup if unsup_steps_limit < 0 else
                min(total_steps_unsup, unsup_steps_limit))
            prev_iter_end = time.time()
            unsup_seen = 0
            fail_none = 0
            fail_shape = 0
            fail_mismatch = 0
            fail_sos = 0
            fail_nan_logits = 0
            target_unsup_updates = int(getattr(cfg, 'unsup_target_updates',
                300))
            interleave_sup_every = int(getattr(cfg, 'interleave_sup_every', 30)
                )
            interleave_sup_weight = float(getattr(cfg,
                'interleave_sup_weight', 0.7))
            sup_iter = iter(sup_loader)
            for step, batch_unsup in enumerate(unsup_loader, 1):
                iter_start = time.time()
                unsup_seen += 1
                if (target_unsup_updates > 0 and unsup_steps >=
                    target_unsup_updates):
                    print(
                        f'[UNSUP] reach target_unsup_updates={target_unsup_updates}, break Stage B.'
                        )
                    break
                global_iter += 1
                apply_warmup(global_iter)
                optimizer.zero_grad(set_to_none=True)
                if step > max_steps_unsup:
                    break
                img_unlabeled = unwrap_img(batch_unsup['img']).to(device,
                    non_blocking=True)
                metas_unlabeled = unwrap_metas(batch_unsup['img_metas'])
                img_weak = img_unlabeled
                img_strong = build_unsup_strong_view(img_unlabeled=
                    img_unlabeled, metas_unlabeled=metas_unlabeled,
                    box_cache=box_cache, cfg=cfg, global_iter=global_iter,
                    boxswap_p=boxswap_p, boxswap_pairs=boxswap_pairs,
                    boxswap_avoid_iou=boxswap_avoid_iou)
                if debug_unsup and (step % debug_every == 0 or step == 1):
                    with torch.no_grad():
                        diff = (img_strong - img_unlabeled).abs().mean().item()
                        print(f'[AUG] mean|strong-weak|={diff:.6f}')
                with torch.inference_mode(), torch.cuda.amp.autocast(enabled
                    =True):
                    out_t = model_teacher.forward_train(img_weak,
                        metas_unlabeled, collect_va=False, strong_view=
                        False, train_mode=True, targets_dict=None)
                t_logits = out_t.get('out_dec', None)
                t_ids = out_t.get('t_ids', None)
                if t_logits is None or t_ids is None:
                    fail_none += 1
                    prev_iter_end = time.time()
                    continue
                if not torch.is_tensor(t_logits) or t_logits.dim() != 3:
                    fail_shape += 1
                    prev_iter_end = time.time()
                    continue
                if not torch.is_tensor(t_ids) or t_ids.dim() != 2:
                    fail_shape += 1
                    prev_iter_end = time.time()
                    continue
                with torch.inference_mode(), torch.cuda.amp.autocast(enabled
                    =True):
                    out_t_strong = model_teacher.forward_train(img_strong,
                        metas_unlabeled, collect_va=False, strong_view=
                        False, train_mode=True, targets_dict=None)
                t_logits_s = out_t_strong.get('out_dec', None)
                t_ids_s = out_t_strong.get('t_ids', None)
                V = int(t_logits.size(-1))
                PAD_i = int(PAD)
                if not 0 <= PAD_i < V:
                    raise ValueError(
                        f'[BAD CFG] PAD token id={PAD_i} but vocab size={V}. PAD must come from convertor.padding_idx, not max_seq_len.'
                        )
                with torch.no_grad():
                    if t_ids.size(0) != t_logits.size(0) or t_ids.size(1
                        ) != t_logits.size(1):
                        fail_mismatch += 1
                        print(
                            f'[WARN] teacher outputs mismatch: t_ids={tuple(t_ids.shape)} vs t_logits={tuple(t_logits.shape)}'
                            )
                        prev_iter_end = time.time()
                        continue
                    if not torch.isfinite(t_logits).all():
                        fail_nan_logits += 1
                        prev_iter_end = time.time()
                        continue
                    sos_ratio = (t_ids[:, (0)] == SOS).float().mean().item()
                    if sos_ratio >= 0.01:
                        fail_sos += 1
                        prev_iter_end = time.time()
                        continue
                    if (EOS is not None and EOS >= 0 and PAD is not None and
                        PAD >= 0):
                        eos_mask = t_ids == EOS
                        after_or_at = eos_mask.cumsum(dim=1) >= 1
                        after = after_or_at & ~eos_mask
                        t_ids = t_ids.clone()
                        t_ids[after] = PAD
                    pad_pos = t_ids == PAD
                    if pad_pos.any():
                        with torch.cuda.amp.autocast(enabled=False):
                            t_logits_fp32 = t_logits.float().clone()
                            t_logits_fp32 = t_logits_fp32.masked_fill(pad_pos
                                .unsqueeze(-1), -10000.0)
                            b_idx, l_idx = pad_pos.nonzero(as_tuple=True)
                            V = t_logits_fp32.size(-1)
                            assert 0 <= int(PAD) < int(V)
                            t_logits_fp32[b_idx, l_idx, PAD] = 10000.0
                        t_logits = t_logits_fp32.to(dtype=t_logits.dtype)
                if pseudo_max_len is not None:
                    Lkeep = min(int(pseudo_max_len), int(t_ids.size(1)))
                    t_ids = t_ids[:, :Lkeep]
                    t_logits = t_logits[:, :Lkeep, :]
                t_ids_before_repair = t_ids.clone()
                (t_ids, t_logits, fill_mask, debug_inserted, insert_ops,
                    t_ids_s, t_logits_s) = (maybe_apply_prm_repair(t_ids=
                    t_ids, t_logits=t_logits, t_ids_s=t_ids_s, t_logits_s=
                    t_logits_s, id2tok=id2tok, tok2id=tok2id, PAD=PAD, EOS=
                    EOS, cfg=cfg))
                with torch.no_grad():
                    T_w = float(getattr(cfg, 'row_conf_T', 2.0))
                    with torch.cuda.amp.autocast(enabled=False):
                        t_logits_fp32 = t_logits.float()
                        t_prob = F.softmax(t_logits_fp32 / T_w, dim=-1)
                        ent = -(t_prob * t_prob.clamp_min(1e-12).log()).sum(dim
                            =-1)
                        ent_norm = ent / (math.log(t_prob.size(-1)) + 1e-12)
                        t_conf = 1.0 - ent_norm
                    base_valid = (t_ids != PAD) & ~fill_mask
                    if EOS is not None and EOS >= 0:
                        eos_mask = t_ids == EOS
                        after_or_at = eos_mask.cumsum(dim=1) >= 1
                        after = after_or_at & ~eos_mask
                        base_valid = base_valid & ~after
                with torch.no_grad():
                    p_t_on_t = torch.ones_like(t_conf)
                    if t_logits_s is not None and torch.is_tensor(t_logits_s
                        ) and t_logits_s.dim(
                        ) == 3 and t_ids_s is not None and torch.is_tensor(
                        t_ids_s) and t_ids_s.dim() == 2:
                        if (EOS is not None and EOS >= 0 and PAD is not
                            None and PAD >= 0):
                            eos_mask_s = t_ids_s == EOS
                            after_or_at_s = eos_mask_s.cumsum(dim=1) >= 1
                            after_s = after_or_at_s & ~eos_mask_s
                            t_ids_s = t_ids_s.clone()
                            t_ids_s[after_s] = PAD
                        if pseudo_max_len is not None:
                            Lkeep_s = min(int(pseudo_max_len), int(t_ids_s.
                                size(1)), int(t_logits_s.size(1)))
                            t_ids_s = t_ids_s[:, :Lkeep_s]
                            t_logits_s = t_logits_s[:, :Lkeep_s, :]
                        pad_pos_s = t_ids_s[:, :t_logits_s.size(1)] == PAD
                        if pad_pos_s.any():
                            t_logits_s_fp32 = t_logits_s.float().clone()
                            t_logits_s_fp32 = t_logits_s_fp32.masked_fill(
                                pad_pos_s.unsqueeze(-1), -10000.0)
                            b_idx, l_idx = pad_pos_s.nonzero(as_tuple=True)
                            V = t_logits_s_fp32.size(-1)
                            t_logits_s_fp32[b_idx, l_idx, PAD] = 10000.0
                            t_logits_s = t_logits_s_fp32.to(dtype=
                                t_logits_s.dtype)
                        Bt, Lt = t_ids.shape
                        Bs, Ls, Vs = t_logits_s.shape
                        Lmin = min(Lt, Ls)
                        prob_s = torch.softmax(t_logits_s[:, :Lmin, :].
                            float(), dim=-1)
                        p = prob_s.gather(-1, t_ids[:, :Lmin].unsqueeze(-1)
                            ).squeeze(-1)
                        same = (t_ids_s[:, :Lmin] == t_ids[:, :Lmin]).float()
                        p = torch.maximum(p, same)
                        if Lmin < Lt:
                            pad = torch.zeros((Bt, Lt - Lmin), device=t_ids
                                .device, dtype=p.dtype)
                            if debug_unsup and step == 1:
                                print(
                                    f'[ALIGN] after repair: Lt={t_ids.size(1)}  strong_L={t_logits_s.size(1)}  inserted={fill_mask.sum().item()}'
                                    )
                            p_t_on_t = torch.cat([p, pad], dim=1)
                        else:
                            p_t_on_t = p
                        p_t_on_t = torch.where(base_valid, p_t_on_t, torch.
                            ones_like(p_t_on_t))
                targets_pseudo = build_pseudo_targets_from_tids(t_ids, PAD=
                    PAD, SOS=SOS)
                targets_pseudo['fill_mask_struct'] = fill_mask
                tr_spans = extract_tr_spans_table_aware(t_ids=t_ids, id2tok
                    =id2tok, tok2id=tok2id, PAD=PAD, EOS=EOS, max_rows=256,
                    default_section_if_missing=1)
                targets_pseudo['tr_spans'] = tr_spans
                if debug_unsup and (step % debug_every == 0 or step == 1):
                    print(
                        f'[DIAG] shapes: t_ids={tuple(t_ids.shape)} t_logits={tuple(t_logits.shape)} fill_mask={tuple(fill_mask.shape)}'
                        )
                with torch.cuda.amp.autocast(enabled=True):
                    out_s = model_student.forward_train(img_strong,
                        metas_unlabeled, collect_va=False, strong_view=
                        student_strong_view, train_mode=True, targets_dict=
                        targets_pseudo)
                with torch.no_grad():
                    s_logits = out_s.get('out_dec', None)
                    if s_logits is None or not torch.is_tensor(s_logits
                        ) or s_logits.dim() != 3:
                        p_on_t = torch.ones_like(t_conf)
                    else:
                        Bs, Ls, Vs = s_logits.shape
                        Bt, Lt = t_ids.shape
                        Lmin = min(Ls, Lt)
                        s_prob = torch.softmax(s_logits[:, :Lmin, :].float(
                            ), dim=-1)
                        p = s_prob.gather(-1, t_ids[:, :Lmin].unsqueeze(-1)
                            ).squeeze(-1)
                        if Lmin < Lt:
                            pad = torch.zeros((Bt, Lt - Lmin), device=t_ids
                                .device, dtype=p.dtype)
                            p_on_t = torch.cat([p, pad], dim=1)
                        else:
                            p_on_t = p
                        if debug_unsup and (step % debug_every == 0 or step ==
                            1):
                            print(
                                f'[ALIGN] s_logits Ls={Ls} t_ids Lt={Lt} Lmin={Lmin} tail={Lt - Lmin}'
                                )
                            if Lmin < Lt:
                                print(
                                    f'[ALIGN] tail p_on_t mean={p_on_t[:, Lmin:].mean().item():.3f} min={p_on_t[:, Lmin:].min().item():.3f} max={p_on_t[:, Lmin:].max().item():.3f}'
                                    )
                token_mask, token_weight = build_row_gated_token_mask_v4(t_conf
                    =t_conf, p_on_t=p_on_t, p_t_on_t=p_t_on_t, t_ids=t_ids,
                    tr_spans=tr_spans, fill_mask=fill_mask, id2tok=id2tok,
                    PAD=PAD, EOS=EOS, cfg=cfg, step=step, epoch=epoch,
                    return_weight=True)
                if debug_unsup and (step % debug_every == 0 or step == 1):
                    sel = token_mask & base_valid
                    ts = float(p_t_on_t[sel].mean().item()) if sel.any(
                        ) else 0.0
                    print(f'[STAB] p_t_on_t_mean_on_kept={ts:.3f}')
                if debug_unsup and (step % debug_every == 0 or step == 1):
                    with torch.no_grad():
                        bv = base_valid
                        kept = (token_mask & bv).sum().item()
                        valid = bv.sum().item()
                        print(
                            f'[MASK] kept={kept} valid={valid} kept_ratio={kept / max(1, valid):.3f} w_mean={token_weight[token_mask].mean().item() if token_mask.any() else 0.0:.3f}'
                            )
                if debug_unsup and (step % debug_every == 0 or step == 1):
                    with torch.no_grad():
                        sel = token_mask & base_valid
                        a = float(p_on_t[sel].mean().item()) if sel.any(
                            ) else 0.0
                        print(f'[AGREE] p_on_t_mean_on_kept={a:.3f}')
                token_mask = token_mask & base_valid
                token_weight = token_weight * token_mask.float()
                if debug_unsup and (step % debug_every == 0 or step == 1):
                    bad = (fill_mask & token_mask).sum().item()
                    if bad != 0:
                        raise RuntimeError(
                            f'[BUG] fill_mask tokens leaked into token_mask: bad={bad}'
                            )
                if debug_unsup and (step % debug_every == 0 or step == 1):
                    with torch.no_grad():
                        B = t_ids.size(0)
                        L = t_ids.size(1)
                        ins_cnt = fill_mask.sum(dim=1)
                        changed = (t_ids != t_ids_before_repair).sum(dim=1)
                        row_num = torch.tensor([(len(x) if x is not None else
                            0) for x in tr_spans], device=t_ids.device)
                        cover_tok = []
                        for b in range(B):
                            c = 0
                            for s, e, _ in (tr_spans[b] or []):
                                s = max(0, min(int(s), L))
                                e = max(0, min(int(e), L))
                                c += max(0, e - s)
                            cover_tok.append(c)
                        cover_tok = torch.tensor(cover_tok, device=t_ids.device
                            )
                        tm = token_mask
                        base_valid = (t_ids != PAD) & ~fill_mask
                        if EOS is not None and EOS >= 0:
                            eos_mask = t_ids == EOS
                            after_or_at = eos_mask.cumsum(dim=1) >= 1
                            after = after_or_at & ~eos_mask
                            base_valid = base_valid & ~after
                        kept_tok = (tm & base_valid).sum(dim=1).float()
                        valid_tok = base_valid.sum(dim=1).float().clamp(min=1.0
                            )
                        kept_ratio = kept_tok / valid_tok
                        fill_in_valid = (fill_mask & base_valid).sum(dim=1
                            ).float()
                        fill_kept = (fill_mask & tm & base_valid).sum(dim=1
                            ).float()
                        fill_drop_ratio = (1.0 - fill_kept / fill_in_valid.
                            clamp(min=1.0))
                        dropped_rows = []
                        total_rows = []
                        for b in range(B):
                            rows = tr_spans[b] or []
                            tot = len(rows)
                            drop = 0
                            for s, e, _ in rows:
                                s = max(0, min(int(s), L))
                                e = max(0, min(int(e), L))
                                if e <= s:
                                    continue
                                m = tm[(b), s:e] & base_valid[(b), s:e]
                                if m.sum().item() == 0:
                                    drop += 1
                            dropped_rows.append(drop)
                            total_rows.append(tot)
                        dropped_rows = torch.tensor(dropped_rows, device=
                            t_ids.device)
                        total_rows = torch.tensor(total_rows, device=t_ids.
                            device).clamp(min=1)
                        kept_rows = (total_rows - dropped_rows).float()
                        print(
                            f'[DIAG] ep={epoch} step={step} ins(mean)={ins_cnt.float().mean().item():.2f} changed(mean)={changed.float().mean().item():.2f} rows(mean)={row_num.float().mean().item():.1f} kept_rows(mean)={kept_rows.mean().item():.1f} span_cover(mean)={cover_tok.float().mean().item() / max(1, L):.2f} kept_tok_ratio(mean)={kept_ratio.mean().item():.2f} fill_drop_ratio(mean)={fill_drop_ratio.mean().item():.2f} drop_rows_ratio(mean)={(dropped_rows.float() / total_rows.float()).mean().item():.2f}'
                            )
                from rps.losses.unsupervised import loss_structure_unsup_tr_block
                t_stageb = max(0, epoch - sup_only_epochs)
                ramp = min(1.0, t_stageb / float(max(1, warmup_u)))
                loss_vec = loss_structure_unsup_tr_block(out_s, cfg,
                    targets_pseudo, token_mask=token_mask, token_weight=
                    token_weight)
                per_cnt = token_mask.sum(dim=1).float().clamp(min=1.0)
                tot_tok = int(token_mask.sum().item())
                min_batch_tok = int(getattr(cfg, 'unsup_min_batch_tokens', 64))
                if debug_unsup and (step % debug_every == 0 or step == 1):
                    print(
                        f'[TOK] tot_tok={tot_tok} min_batch_tok={min_batch_tok} skip={tot_tok < min_batch_tok} ramp={ramp:.2f} lambda_u={lambda_u}'
                        )
                if tot_tok < min_batch_tok:
                    if debug_unsup and (step % debug_every == 0 or step == 1):
                        print(
                            f'[UNSUP] skip step={step} because too few kept tokens: {tot_tok} < {min_batch_tok}'
                            )
                    prev_iter_end = time.time()
                    if training_observer is not None:
                        training_observer('unsupervised_skipped', locals())
                    continue
                loss_u = (loss_vec * per_cnt).sum() / per_cnt.sum()
                unsup_loss = loss_u * (lambda_u * ramp)
                if not unsup_loss.requires_grad:
                    raise RuntimeError(
                        '[BUG] unsup_loss does not require grad (graph broken).'
                        )
                if training_observer is not None:
                    training_observer('unsupervised_before_backward', locals())
                if managed_training:
                    _managed_amp_backward(unsup_loss, model_student,
                        optimizer, scaler, training_observer,
                        'unsupervised', locals())
                else:
                    scaler.scale(unsup_loss).backward()
                    scaler.unscale_(optimizer)
                    if training_observer is not None:
                        training_observer('unsupervised_backward', locals())
                torch.nn.utils.clip_grad_norm_(model_student.parameters(),
                    max_norm=grad_clip)
                old_unsup_scale = scaler.get_scale()
                scaler.step(optimizer)
                scaler.update()
                record_optimizer_update('unsupervised', old_unsup_scale)
                m0 = float(getattr(cfg, 'ema_momentum_start', 0.99))
                m1 = float(getattr(cfg, 'ema_momentum_end', 0.999))
                t = int(epoch - sup_only_epochs)
                T = int(getattr(cfg, 'ema_warmup_epochs', 10))
                ema_m_unsup = m0 + (m1 - m0) * min(1.0, t / float(max(1, T)))
                ema_update(model_teacher, model_student, momentum=ema_m_unsup)
                unsup_steps += 1
                if training_observer is not None:
                    training_observer('unsupervised_updated', locals())
                if managed_training:
                    unsup_loss = unsup_loss.detach()
                    loss_u = loss_u.detach()
                    del out_s, loss_vec, s_logits
                    locals()
                if (interleave_sup_every > 0 and unsup_steps %
                    interleave_sup_every == 0):
                    global_iter += 1
                    apply_warmup(global_iter)
                    try:
                        batch_sup = next(sup_iter)
                    except StopIteration:
                        sup_iter = iter(sup_loader)
                        batch_sup = next(sup_iter)
                    img_sup = unwrap_img(batch_sup['img']).to(device,
                        non_blocking=True)
                    metas_sup = unwrap_metas(batch_sup['img_metas'])
                    targets_sup = convertor.str_bbox_format(metas_sup)
                    optimizer.zero_grad(set_to_none=True)
                    with torch.cuda.amp.autocast(enabled=True):
                        out_sup = model_student.forward_train(img_sup,
                            metas_sup, collect_va=False, strong_view=False,
                            train_mode=True, targets_dict=targets_sup)
                        loss_tf_sup2, _, _, loss_main_sup2 = loss_sups(out_sup,
                            cfg, enable_bbox=False, enable_cell=False,
                            w_cell=0.0)
                        loss_sup = loss_main_sup2 * interleave_sup_weight
                    if managed_training:
                        _managed_amp_backward(loss_sup, model_student,
                            optimizer, scaler, training_observer,
                            'interleave', locals())
                    else:
                        scaler.scale(loss_sup).backward()
                        scaler.unscale_(optimizer)
                        if training_observer is not None:
                            training_observer('interleave_backward', locals())
                    torch.nn.utils.clip_grad_norm_(model_student.parameters
                        (), max_norm=grad_clip)
                    old_interleave_scale = scaler.get_scale()
                    scaler.step(optimizer)
                    scaler.update()
                    record_optimizer_update('interleave', old_interleave_scale)
                    ema_update(model_teacher, model_student, momentum=
                        ema_m_unsup)
                    if training_observer is not None:
                        training_observer('interleave_updated', locals())
                    if managed_training:
                        del out_sup, loss_tf_sup2, loss_main_sup2, loss_sup
                        del img_sup, metas_sup, targets_sup
                        locals()
                unsup_loss_sum += float(unsup_loss.item())
                unsup_st_sum += float(loss_u.detach().cpu())
                if step % LOG_EVERY == 0 or step == 1:
                    lr_now = optimizer.param_groups[0]['lr']
                    t_data = iter_start - prev_iter_end
                    t_iter = time.time() - iter_start
                    with torch.no_grad():
                        B, L = t_ids.shape
                        tm = token_mask
                        base_valid2 = (t_ids != PAD) & ~fill_mask
                        if EOS is not None and EOS >= 0:
                            eos_mask = t_ids == EOS
                            after_or_at = eos_mask.cumsum(dim=1) >= 1
                            after = after_or_at & ~eos_mask
                            base_valid2 = base_valid2 & ~after
                        kept_tok = (tm & base_valid2).sum(dim=1).float()
                        valid_tok = base_valid2.sum(dim=1).float().clamp(min
                            =1.0)
                        kept_ratio_mean = (kept_tok / valid_tok).mean().item()
                        kept_tok_mean = kept_tok.mean().item()
                    print(
                        f'[UNSUP-CE] ep={epoch} step={step}/{max_steps_unsup} lr={lr_now:.6e} loss={unsup_loss.item():.4f} ce={loss_u.item():.4f} | kept_ratio={kept_ratio_mean:.2f} kept_tok={kept_tok_mean:.1f} | ramp={ramp:.2f} | iter={t_iter:.3f}s load={t_data:.3f}s'
                        )
                prev_iter_end = time.time()
                del out_t, t_logits, t_ids, img_unlabeled, img_weak, img_strong
                if managed_training:
                    del unsup_loss, loss_u
                    locals()
                else:
                    del out_s, loss_vec
            print(
                f'[UNSUP-DIAG] seen={unsup_seen} updates={unsup_steps} fail_none={fail_none} fail_shape={fail_shape} fail_mismatch={fail_mismatch} fail_sos={fail_sos} fail_nan_logits={fail_nan_logits}'
                )
        if managed_training:
            expected_counts = expected_update_counts(epoch)
            if (update_counts != expected_counts or 
                cumulative_optimizer_steps != sum(expected_counts.values()) or
                global_iter != cumulative_optimizer_steps):
                raise RuntimeError(
                    f'Epoch {epoch} update count mismatch: actual optimizer counts={update_counts}, expected={expected_counts}, global_iter={global_iter}; latest checkpoint was not advanced'
                    )
        avg_sup = sup_loss_sum / max(1, sup_steps)
        avg_unsup = unsup_loss_sum / max(1, unsup_steps)
        avg_total = avg_sup + avg_unsup
        avg_st_sup = sup_st_sum / max(1, sup_steps)
        avg_st_unsup = unsup_st_sum / max(1, unsup_steps)
        lr_now = optimizer.param_groups[0]['lr']
        print('\n' + '=' * 80)
        print(f'[EpochDone] ep={epoch} time={time.time() - t0:.1f}s')
        print(f'  avg_sup   = {avg_sup:.4f}   (steps={sup_steps})')
        print(
            f'  avg_unsup = {avg_unsup:.4f} (steps={unsup_steps}, lambda_u included)'
            )
        print(f'  avg_total = {avg_total:.4f}')
        print('=' * 80)
        (val_TEDS) = (val_precision) = (val_recall) = (val_F1) = (val_CER) = (
            val_score) = None if managed_training else 0.0
        val_metrics = None
        do_val = val_loader is not None and (epoch % val_interval == 0 or 
            epoch == total_epochs)
        if do_val:
            with torch.no_grad():
                val_metrics = evaluate_on_val(model_teacher, val_loader,
                    convertor, max_batches=len(val_loader), compute_bbox=
                    False, compute_teds=True, compute_cer=False)
            val_TEDS = float(val_metrics['TEDS'])
            val_precision = float(val_metrics['precision'])
            val_recall = float(val_metrics['recall'])
            val_F1 = float(val_metrics['F1'])
            val_CER = float(val_metrics['CER'])
            val_score = float(val_metrics['score'])
            print('\n[VAL] ====================')
            print(f'Epoch     : {epoch}')
            print(f'TEDS      : {val_TEDS:.4f}')
            print(f'Precision : {val_precision:.4f}')
            print(f'Recall    : {val_recall:.4f}')
            print(f'F1        : {val_F1:.4f}')
            print(f'CER       : {val_CER:.4f}')
            print(f'SCORE     : {val_score:.4f}')
            print('[VAL] ====================\n')
        else:
            print(f'[VAL] skip epoch {epoch} (val_interval={val_interval})')
        with open(csv_path, 'a', newline='') as f:
            writer = csv.writer(f)
            writer.writerow([epoch, avg_total, avg_sup, avg_unsup,
                avg_st_sup, avg_st_unsup, lr_now, val_TEDS if do_val else
                '', val_precision if do_val else '', val_recall if do_val else
                '', val_F1 if do_val else '', val_CER if do_val else '', 
                val_score if do_val else ''])
        current_metric = float(val_metrics[best_metric_name]
            ) if do_val else None
        if (managed_training and current_metric is not None and not math.
            isfinite(current_metric)):
            raise RuntimeError(
                'Non-finite validation metric; refusing to commit this epoch')
        is_best = False
        if current_metric is not None and current_metric > best_metric:
            is_best = True
            best_metric = current_metric
            best_epoch = epoch
        if managed_training:
            if scheduler is not None and (warmup_iters == 0 or global_iter >=
                warmup_iters):
                scheduler.step()
            epochs_completed += 1
            if training_observer is not None:
                training_observer('epoch_end', locals())
            optimizer_extra = {name: copy.deepcopy(getattr(optimizer, name)
                ) for name in ('radam_buffer', 'alpha', 'k',
                'N_sma_threshhold', 'gc_loc', 'use_gc', 'gc_conv_only',
                '_step_count') if hasattr(optimizer, name)}
            managed_state = dict(schema_version=2, identity=copy.deepcopy(
                training_identity), completed_epoch=epoch, total_epochs=
                total_epochs, cumulative_optimizer_steps=
                cumulative_optimizer_steps, update_counts=dict(
                update_counts), global_iter=global_iter, best_metric=
                best_metric, best_epoch=best_epoch, base_lrs=list(base_lrs),
                scheduler_step_complete=True, optimizer_extra=
                optimizer_extra, optimizer_state_ids=sorted(optimizer.
                state_dict()['state']), observer_state=training_observer.
                state_dict() if training_observer is not None and hasattr(
                training_observer, 'state_dict') else None, rng=
                capture_training_rng(), loaders=capture_loader_states(dict(
                sup=sup_loader, unsup=unsup_loader, val=val_loader)))
            meta = dict(epoch=epoch, avg_total=avg_total, avg_sup=avg_sup,
                avg_unsup=avg_unsup, lr=lr_now, next_lr=[group['lr'] for
                group in optimizer.param_groups], val_TEDS=val_TEDS if
                do_val else None, val_score=val_score if do_val else None,
                best_metric=best_metric, best_epoch=best_epoch)
            if save_checkpoints:
                latest_path = os.path.join(ckpt_subdir, 'latest_full.pth')
                _save_ckpt_full(model_student, model_teacher, optimizer,
                    scaler, scheduler, latest_path, meta=meta, global_iter=
                    global_iter, training_state=managed_state)
                print(f'✔ Latest FULL atomically committed: {latest_path}')
                if is_best:
                    best_path = os.path.join(ckpt_subdir, 'best_full.pth')
                    _save_ckpt_full(model_student, model_teacher, optimizer,
                        scaler, scheduler, best_path, meta=meta,
                        global_iter=global_iter, training_state=managed_state)
                    print(f'✔ Best FULL atomically saved: {best_path}')
            if save_final_checkpoint and epoch == total_epochs:
                final_path = os.path.join(ckpt_subdir, 'final_full.pth')
                _save_ckpt_full(model_student, model_teacher, optimizer,
                    scaler, scheduler, final_path, meta=meta, global_iter=
                    global_iter, training_state=managed_state)
                print(
                    f'✔ Final FULL saved after the complete plan: {final_path}'
                    )
            gc.collect()
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
            if stop_after_epoch > 0 and epoch >= stop_after_epoch:
                break
            continue
        if save_checkpoints and is_best:
            meta = dict(epoch=epoch, avg_total=avg_total, avg_sup=avg_sup,
                avg_unsup=avg_unsup, lr=lr_now, val_score=val_score,
                val_TEDS=val_TEDS, val_F1=val_F1)
            best_path = os.path.join(ckpt_subdir, 'best_full.pth')
            _save_ckpt_full(model_student, model_teacher, optimizer, scaler,
                scheduler, best_path, meta=meta, global_iter=global_iter)
            print(f'✔ Best FULL saved: {best_path} (metric={best_metric:.4f})')
            best_teacher_path = os.path.join(ckpt_subdir,
                'best_teacher_fp16.pth')
            _save_ckpt_light_fp16(model_teacher, best_teacher_path, meta=meta)
            print(f'✔ Best Teacher (fp16) saved: {best_teacher_path}')
        elif save_checkpoints and epoch % checkpoint_interval == 0:
            ckpt_path = os.path.join(ckpt_subdir, f'epoch_{epoch}_full.pth')
            _save_ckpt_full(model_student, model_teacher, optimizer, scaler,
                scheduler, ckpt_path, meta={'epoch': epoch}, global_iter=
                global_iter)
            print(f'✔ CKPT FULL saved: {ckpt_path}')
            teacher_path = os.path.join(ckpt_subdir,
                f'epoch_{epoch}_teacher.pth')
            _save_ckpt_light_fp16(model_teacher, teacher_path, meta={
                'epoch': epoch})
            print(f'✔ CKPT Teacher saved: {teacher_path}')
        if scheduler is not None and (warmup_iters == 0 or global_iter >=
            warmup_iters):
            scheduler.step()
        epochs_completed += 1
        if training_observer is not None:
            training_observer('epoch_end', locals())
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        if stop_after_epochs > 0 and epochs_completed >= stop_after_epochs:
            break
    if not managed_training and save_final_checkpoint and epochs_completed > 0:
        final_path = os.path.join(ckpt_subdir, 'final_full.pth')
        _save_ckpt_full(model_student, model_teacher, optimizer, scaler,
            scheduler, final_path, meta=dict(epoch=epoch, avg_total=
            avg_total, avg_sup=avg_sup, avg_unsup=avg_unsup, val_TEDS=
            val_TEDS if do_val else None, val_score=val_score if do_val else
            None), global_iter=global_iter)
        print(f'✔ Final FULL saved: {final_path}')
    print('\n' + '=' * 80)
