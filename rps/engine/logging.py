def _cfg_get(cfg, key, default=None):
    if cfg is None:
        return default
    if isinstance(cfg, dict):
        return cfg.get(key, default)
    if hasattr(cfg, key):
        return getattr(cfg, key)
    try:
        return cfg.get(key, default)
    except Exception:
        return default


def _safe_len_file(path):
    try:
        with open(path, 'r', encoding='utf-8') as f:
            return sum(1 for _ in f)
    except Exception:
        return None


def log_cfg_snapshot(cfg):
    snap = {}
    snap['log_every'] = int(_cfg_get(cfg, 'log_every', -1))
    snap['checkpoint_interval'] = int(_cfg_get(cfg, 'checkpoint_interval', -1))
    snap['sup_bs'] = int(_cfg_get(cfg, 'sup_bs', -1))
    snap['unsup_bs'] = int(_cfg_get(cfg, 'unsup_bs', -1))
    snap['val_bs'] = int(_cfg_get(cfg, 'val_bs', -1))
    snap['PAD'] = int(_cfg_get(cfg, 'PAD', -1))
    snap['SOS'] = int(_cfg_get(cfg, 'SOS', -1))
    snap['EOS'] = int(_cfg_get(cfg, 'EOS', -1))
    snap['max_seq_len'] = int(_cfg_get(cfg, 'max_seq_len', -1))
    snap['lambda_u'] = float(_cfg_get(cfg, 'lambda_u', 0.0))
    snap['ema_momentum'] = float(_cfg_get(cfg, 'ema_momentum', 0.0))
    snap['strong_noise_sigma'] = float(_cfg_get(cfg, 'strong_noise_sigma', 0.0)
        )
    snap['grad_clip'] = float(_cfg_get(cfg, 'grad_clip', 0.0))
    snap['unsup_steps_per_epoch'] = int(_cfg_get(cfg,
        'unsup_steps_per_epoch', -1))
    snap['unsup_warmup_epochs'] = int(_cfg_get(cfg, 'unsup_warmup_epochs', 0))
    snap['unsup_keep_top_p'] = float(_cfg_get(cfg, 'unsup_keep_top_p', -1))
    snap['unsup_keep_thr'] = float(_cfg_get(cfg, 'unsup_keep_thr', -1))
    snap['unsup_max_len'] = int(_cfg_get(cfg, 'unsup_max_len', -1))
    snap['ema_in_sup'] = bool(_cfg_get(cfg, 'ema_in_sup', False))
    opt_cfg = _cfg_get(cfg, 'optimizer', None)
    lr_cfg = _cfg_get(cfg, 'lr_config', None)
    snap['optimizer'] = opt_cfg if isinstance(opt_cfg, dict) else str(opt_cfg)
    snap['lr_config'] = lr_cfg if isinstance(lr_cfg, dict) else str(lr_cfg)
    snap['pretrained'] = _cfg_get(cfg, 'pretrained', None)
    snap['resume_from'] = _cfg_get(cfg, 'resume_from', None)
    print('\n' + '=' * 80)
    print('[CFG SNAPSHOT] (important settings)')
    for k in sorted(snap.keys()):
        print(f'{k}: {snap[k]}')
    print('=' * 80 + '\n')
