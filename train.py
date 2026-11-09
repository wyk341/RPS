#!/usr/bin/env python3
"""Train RPS with user-provided image manifests and cached OCR boxes."""
import argparse
import copy
import hashlib
import json
import os
from pathlib import Path
import random

os.environ.setdefault('CUBLAS_WORKSPACE_CONFIG', ':4096:8')


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--config', default=str(Path(__file__).parent/'configs/rps.py'))
    parser.add_argument('--labeled', help='JSONL manifest for labeled training images')
    parser.add_argument('--unlabeled', help='JSONL manifest for unlabeled training images')
    parser.add_argument('--val', help='JSONL manifest for validation images')
    parser.add_argument('--image-root', default='.')
    parser.add_argument('--ocr-cache', help='Offline boxes_xyxy_480 JSONL cache')
    parser.add_argument('--work-dir', default='runs/rps')
    parser.add_argument('--pretrained', help='Supervised model initialization checkpoint')
    parser.add_argument('--resume', help='Full RPS training checkpoint')
    parser.add_argument('--seed', type=int, help='User-selected random seed')
    parser.add_argument('--epochs', type=int, help='Total optimization schedule in epochs')
    parser.add_argument('--stop-after-epoch', type=int, default=0)
    parser.add_argument('--mode', choices=['semi','supervised'], default='semi')
    parser.add_argument('--check-config', action='store_true', help='Validate configuration without loading data')
    return parser.parse_args()


def sha256(path):
    digest = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda:stream.read(1024*1024), b''):
            digest.update(block)
    return digest.hexdigest()


def main():
    args = parse_args()
    from rps.runtime import Config
    cfg = Config.fromfile(args.config)
    cfg.train_mode = args.mode
    if args.epochs is not None:
        if args.epochs < 1:
            raise ValueError('--epochs must be positive.')
        cfg.semi_epochs = cfg.sup_epochs = args.epochs
    if args.check_config:
        print(json.dumps(dict(config=str(Path(args.config).resolve()), decoder_layers=cfg.model.decoder.N,
                              max_sequence_length=cfg.model.max_seq_len, image_size=cfg.image_size,
                              epochs=cfg.semi_epochs, spp_pairs=cfg.boxswap_pairs), indent=2))
        return
    required = ['labeled','val'] + (['unlabeled','ocr_cache'] if args.mode=='semi' else [])
    if any(getattr(args, name) is None for name in required):
        raise ValueError('Required arguments: '+', '.join('--'+name.replace('_','-') for name in required))
    if args.resume and args.pretrained:
        raise ValueError('Use either --resume or --pretrained.')

    import numpy as np
    import torch
    from rps.models import build_model
    from rps.data import TableDataset, make_loader, check_disjoint
    from rps.optim import Ranger
    from rps.augment import OCRBoxCache
    from rps.engine.trainer import train

    saved_identity = None
    if args.resume:
        checkpoint = torch.load(args.resume, map_location='cpu')
        saved_identity = checkpoint['training_state']['identity']
        seed = saved_identity['seed']
        if args.seed is not None and args.seed != seed:
            raise ValueError('The requested seed differs from the resumed run.')
        del checkpoint
    else:
        seed = args.seed if args.seed is not None else int.from_bytes(os.urandom(4), 'little')
    if not 0 <= seed < 2**32:
        raise ValueError('Seed must be an unsigned 32-bit integer.')
    random.seed(seed); np.random.seed(seed); torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    cfg.seed = seed
    cfg.work_dir = str(Path(args.work_dir).resolve())
    cfg.pretrained = str(Path(args.pretrained).resolve()) if args.pretrained else None
    cfg.resume_from = str(Path(args.resume).resolve()) if args.resume else None
    cfg.stop_after_epoch = args.stop_after_epoch
    cfg.ocr_jsonl_trainpad = str(Path(args.ocr_cache).resolve()) if args.ocr_cache else None

    labeled = TableDataset(args.labeled, args.image_root, cfg.image_size)
    validation = TableDataset(args.val, args.image_root, cfg.image_size)
    unlabeled = TableDataset(args.unlabeled, args.image_root, cfg.image_size, labeled=False) if args.unlabeled else labeled
    if args.mode=='semi':
        check_disjoint(labeled,unlabeled,validation)
        cache=OCRBoxCache(cfg.ocr_jsonl_trainpad)
        coverage=sum(bool(cache.get_boxes(str(unlabeled.path(r)))) for r in unlabeled.records)
        if not coverage:
            raise ValueError('The OCR cache has no boxes for the unlabeled manifest.')
        print('OCR box coverage: {}/{}'.format(coverage,len(unlabeled)))
        del cache
    else:
        names={str(labeled.path(r).resolve()) for r in labeled.records}
        if names & {str(validation.path(r).resolve()) for r in validation.records}:
            raise ValueError('Training and validation image lists must be disjoint.')
        cfg.use_spp=False
        cfg.sup_only_epochs=cfg.sup_epochs
    sup_loader=make_loader(labeled,cfg.sup_bs,seed)
    unsup_loader=make_loader(unlabeled,cfg.unsup_bs,seed)
    val_loader=make_loader(validation,cfg.val_bs,seed,shuffle=False)
    cfg.sup_steps_per_epoch=len(sup_loader)
    cfg.unsup_target_updates=int(cfg.unsup_steps_per_epoch)
    if args.mode=='semi' and len(unsup_loader)<cfg.unsup_target_updates:
        raise ValueError('The unlabeled manifest supplies fewer batches than unsup_steps_per_epoch; configure the per-epoch budget for this manifest.')
    config_identity=cfg.to_dict()
    for name in ['seed','work_dir','pretrained','resume_from','stop_after_epoch']:
        config_identity.pop(name,None)
    config_identity['model']['label_convertor']['dict_file']=sha256(cfg.model.label_convertor.dict_file)
    config_identity['model']['label_convertor']['cell_dict_file']=sha256(cfg.model.label_convertor.cell_dict_file)
    config_identity['label_convertor']['dict_file']=sha256(cfg.label_convertor.dict_file)
    config_identity['label_convertor']['cell_dict_file']=sha256(cfg.label_convertor.cell_dict_file)
    identity=dict(seed=seed,run_directory=cfg.work_dir,mode=args.mode,
                  config_sha256=hashlib.sha256(json.dumps(config_identity,sort_keys=True).encode()).hexdigest(),
                  labeled_sha256=sha256(args.labeled),validation_sha256=sha256(args.val),
                  unlabeled_sha256=sha256(args.unlabeled) if args.unlabeled else None,
                  ocr_sha256=sha256(args.ocr_cache) if args.ocr_cache else None)
    if saved_identity is not None and identity!=saved_identity:
        raise ValueError('Configuration, data manifests, OCR cache, or output directory differs from the resumed run.')
    cfg.training_identity=identity
    model=build_model(cfg.model)
    optimizer_cfg=dict(cfg.optimizer); optimizer_cfg.pop('type')
    optimizer=Ranger(model.parameters(),**optimizer_cfg)
    train(model,[sup_loader],[unsup_loader],[val_loader],optimizer,cfg)


if __name__=='__main__':
    main()
