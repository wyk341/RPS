"""Manifest validation and command-line wiring without model optimization."""
import argparse
import contextlib
import copy
import importlib.util
import io
import json
from pathlib import Path
import runpy
import tempfile
import unittest
from unittest import mock

import cv2
import numpy as np
from torch import nn

ROOT=Path(__file__).resolve().parents[1]
spec=importlib.util.spec_from_file_location('rps_train_entry',ROOT/'train.py')
entry=importlib.util.module_from_spec(spec)
spec.loader.exec_module(entry)


class CommandLineTests(unittest.TestCase):
    def inputs(self,root):
        config=runpy.run_path(str(ROOT/'configs/rps.py'))['config']
        config.update(sup_bs=1,unsup_bs=1,val_bs=1,unsup_steps_per_epoch=1)
        path=root/'config.py';path.write_text('config='+repr(config)+'\n')
        for name in ['labeled','unlabeled','val']:
            cv2.imwrite(str(root/(name+'.png')),np.zeros((20,40,3),dtype=np.uint8))
            item=dict(image=name+'.png',width=40,height=20)
            if name!='unlabeled':item['structure']=['<tbody>','<tr>','<td></td>','</tr>','</tbody>']
            (root/(name+'.jsonl')).write_text(json.dumps(item)+'\n')
        boxes=root/'ocr.jsonl'
        boxes.write_text(json.dumps(dict(img_path='unlabeled.png',boxes_xyxy_480=[[10,10,30,30],[50,10,70,30]]))+'\n')
        return argparse.Namespace(config=str(path),labeled=str(root/'labeled.jsonl'),
            unlabeled=str(root/'unlabeled.jsonl'),val=str(root/'val.jsonl'),image_root=str(root),
            ocr_cache=str(boxes),work_dir=str(root/'run'),pretrained=None,resume=None,
            seed=None,epochs=None,stop_after_epoch=0,mode='semi',check_config=False)

    def test_semi_entry_passes_manifests_and_identity_to_trainer(self):
        with tempfile.TemporaryDirectory() as folder:
            args=self.inputs(Path(folder))
            with mock.patch.object(entry,'parse_args',return_value=args), \
                    mock.patch('rps.models.build_model',return_value=nn.Linear(1,1)), \
                    mock.patch('rps.engine.trainer.train') as trainer, \
                    contextlib.redirect_stdout(io.StringIO()):
                entry.main()
            self.assertEqual(trainer.call_count,1)
            cfg=trainer.call_args[0][-1]
            self.assertEqual(cfg.sup_steps_per_epoch,1)
            self.assertEqual(cfg.unsup_target_updates,1)
            self.assertTrue(cfg.use_spp)
            self.assertEqual(cfg.training_identity['labeled_sha256'],entry.sha256(args.labeled))
            self.assertEqual(cfg.training_identity['ocr_sha256'],entry.sha256(args.ocr_cache))

    def test_zero_ocr_coverage_is_rejected_before_building_model(self):
        with tempfile.TemporaryDirectory() as folder:
            root=Path(folder);args=self.inputs(root)
            Path(args.ocr_cache).write_text('')
            with mock.patch.object(entry,'parse_args',return_value=args), \
                    mock.patch('rps.models.build_model') as build, \
                    contextlib.redirect_stdout(io.StringIO()):
                with self.assertRaisesRegex(ValueError,'no boxes'):
                    entry.main()
            build.assert_not_called()

    def test_supervised_entry_requires_no_ocr_cache(self):
        with tempfile.TemporaryDirectory() as folder:
            args=self.inputs(Path(folder));args.mode='supervised';args.unlabeled=None;args.ocr_cache=None
            with mock.patch.object(entry,'parse_args',return_value=args), \
                    mock.patch('rps.models.build_model',return_value=nn.Linear(1,1)), \
                    mock.patch('rps.engine.trainer.train') as trainer, \
                    contextlib.redirect_stdout(io.StringIO()):
                entry.main()
            cfg=trainer.call_args[0][-1]
            self.assertFalse(cfg.use_spp)
            self.assertEqual(cfg.sup_only_epochs,cfg.sup_epochs)


if __name__=='__main__':
    unittest.main()
