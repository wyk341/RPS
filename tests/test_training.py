"""CPU control-flow and resume tests using a tiny synthetic recognizer."""
import contextlib
import copy
import os
from pathlib import Path
import sys
import tempfile
import unittest
from unittest import mock
import warnings

import numpy as np
import torch
from torch import nn
from torch.utils.data import Dataset

from rps.runtime import Config
from rps.models import build_convertor
from rps.data import make_loader
from rps.engine.state import capture_training_rng, restore_training_rng
from rps.engine.trainer import train
from rps.optim import Ranger

ROOT=Path(__file__).resolve().parents[1]


class TinyDataset(Dataset):
    flag=np.zeros(1,dtype=np.uint8)

    def __len__(self):
        return 1

    def __getitem__(self,index):
        tokens=['<tbody>','<tr>','<td></td>','</tr>','</tbody>']
        return dict(img=torch.zeros(3,8,8),img_metas=dict(
            filename='table.png',text=','.join(tokens),bbox=np.zeros((len(tokens),4),dtype=np.float32),
            bbox_masks=np.zeros(len(tokens),dtype=np.int64),cell_content=[]))


class TinyRecognizer(nn.Module):
    def __init__(self,convertor):
        super().__init__()
        sequence=[convertor.char2idx[t] for t in ['<tbody>','<tr>','<td></td>','</tr>','</tbody>']]
        sequence+=[convertor.end_idx]
        sequence+=[convertor.padding_idx]*(12-len(sequence))
        self.register_buffer('tokens',torch.tensor(sequence,dtype=torch.long))
        logits=torch.zeros(12,convertor.num_classes())
        logits.scatter_(1,self.tokens[:,None],5.)
        self.logits=nn.Parameter(logits)

    def forward_train(self,imgs,metas,targets_dict=None,**kwargs):
        length=targets_dict['padded_targets'].size(1)-1 if targets_dict is not None else self.tokens.numel()
        return dict(out_dec=self.logits[None,:length].expand(imgs.size(0),-1,-1),
                    out_bbox=None,out_cell=None,
                    t_ids=self.tokens[None,:length].expand(imgs.size(0),-1),targets=targets_dict)


class TrainingTests(unittest.TestCase):
    def test_segmented_resume_matches_continuous_updates(self):
        with tempfile.TemporaryDirectory() as directory, \
                mock.patch('torch.cuda.is_available',return_value=False), \
                warnings.catch_warnings():
            warnings.simplefilter('ignore',UserWarning)
            root=Path(directory)
            template=Config.fromfile(ROOT/'configs/rps.py')
            template.label_convertor.max_seq_len=12
            template.update(semi_epochs=2,sup_only_epochs=0,sup_steps_per_epoch=1,
                unsup_steps_per_epoch=1,unsup_target_updates=1,interleave_sup_every=1,
                unsup_min_batch_tokens=1,unsup_max_len=12,use_spp=False,
                ocr_jsonl_trainpad=None,log_every=1,pretrained=None)
            template.lr_config.warmup_iters=0
            stream_seed=int.from_bytes(os.urandom(4),'little')
            initial_rng=capture_training_rng()

            def run(destination,stop=0,resume=None):
                cfg=copy.deepcopy(template)
                cfg.update(seed=stream_seed,work_dir=str(destination),stop_after_epoch=stop,
                           resume_from=str(resume) if resume else None,
                           training_identity=dict(case='resume-control-flow'))
                convertor=build_convertor(cfg.label_convertor)
                model=TinyRecognizer(convertor)
                optimizer=Ranger(model.parameters(),lr=2e-4)
                labeled=make_loader(TinyDataset(),1,stream_seed)
                unlabeled=make_loader(TinyDataset(),1,stream_seed)
                stdout,stderr=sys.stdout,sys.stderr
                log=None
                try:
                    train(model,[labeled],[unlabeled],[],optimizer,cfg)
                    log=getattr(sys.stdout,'log',None)
                finally:
                    if log is None:log=getattr(sys.stdout,'log',None)
                    sys.stdout,sys.stderr=stdout,stderr
                    if log is not None:log.close()
                return torch.load(destination/'ckpt/latest_full.pth',map_location='cpu')

            with open(os.devnull,'w') as stream,contextlib.redirect_stdout(stream),contextlib.redirect_stderr(stream):
                restore_training_rng(initial_rng)
                continuous=run(root/'continuous')
                restore_training_rng(initial_rng)
                first=run(root/'segmented',stop=1)
                self.assertEqual(first['global_iter'],3)
                resumed=run(root/'segmented',resume=root/'segmented/ckpt/latest_full.pth')
            self.assertEqual(continuous['global_iter'],6)
            self.assertEqual(resumed['global_iter'],6)
            for key in ['state_dict','teacher_state_dict']:
                self.assertEqual(set(continuous[key]),set(resumed[key]))
                for name in continuous[key]:
                    torch.testing.assert_close(continuous[key][name],resumed[key][name],rtol=0,atol=0)
            self.assertEqual(continuous['training_state']['update_counts'],
                             dict(supervised=2,unsupervised=2,interleave=2))
            self.assertEqual(continuous['scheduler'],resumed['scheduler'])
            torch.testing.assert_close(continuous['training_state']['rng']['torch_cpu'],
                                       resumed['training_state']['rng']['torch_cpu'],rtol=0,atol=0)


if __name__=='__main__':
    unittest.main()
