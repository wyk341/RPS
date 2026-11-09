import copy
import json
import os
from pathlib import Path
import tempfile
import unittest

import cv2
import numpy as np
import torch
import torch.nn.functional as F

from rps.runtime import Config
from rps.mps import (build_row_gated_token_mask_v4, confidence_from_logits,
                     stability_from_logits, agreement_from_logits)
from rps.losses.unsupervised import loss_structure_unsup_tr_block
from rps.augment import OCRBoxCache, strong_box_swap_tensor1
from rps.data import TableDataset, make_loader, check_disjoint
from rps.engine.state import capture_loader_states, restore_loader_states
from rps.models import build_model, build_convertor
from rps.metrics.normalization import normalize_structure_for_teds
from rps.metrics.teds import Analysis_TED
from rps.utils import list_from_file

torch.set_num_threads(1)
ROOT=Path(__file__).resolve().parents[1]


class ReliabilityTests(unittest.TestCase):
    def test_confidence_uniform_and_concentrated(self):
        logits=torch.tensor([[[0.,0.,0.],[40.,-40.,-40.]]])
        scores=confidence_from_logits(logits)
        self.assertLess(abs(float(scores[0,0])),1e-6)
        self.assertGreater(float(scores[0,1]),.99)

    def test_stability_same_token_override_and_invalid_positions(self):
        weak=torch.tensor([[1,1,1]])
        strong=torch.tensor([[1,0,0]])
        logits=torch.tensor([[[0.,.1,0.],[2.,0.,0.],[0.,0.,0.]]])
        valid=torch.tensor([[True,True,False]])
        scores=stability_from_logits(weak,strong,logits,valid)
        self.assertEqual(float(scores[0,0]),1.)
        self.assertAlmostEqual(float(scores[0,1]),float(logits[0,1].softmax(-1)[1]),places=7)
        self.assertEqual(float(scores[0,2]),1.)

    def test_agreement_is_probability_on_teacher_target(self):
        logits=torch.tensor([[[3.,1.],[1.,3.]]])
        values=agreement_from_logits(logits,torch.tensor([[1,1]]))
        torch.testing.assert_close(values,torch.tensor([[.11920292,.88079703]]))


class SelectionAndLossTests(unittest.TestCase):
    def setUp(self):
        self.cfg=Config(PAD=0,EOS=2,SOS=3,row_keep_ratio=.5,row_conf_quantile=.5,
            row_stab_quantile=.5,row_agree_thr=.5,row_stab_thr=.5,min_keep_rows=2,
            min_keep_rows_thead=0,min_keep_rows_tbody=0,fallback_row_weight=.15,
            boundary_token_weight=.3,min_row_tokens=2)
        self.names=['PAD','x','EOS','SOS','<tbody>','<tr>','<td></td>','</tr>','</tbody>']
        self.ids=torch.tensor([[4,5,6,7,5,6,7,8,2,0]])
        self.spans=[[(1,4,1),(4,7,1)]]
        self.conf=torch.tensor([[.9]*4+[.5]*3+[.9]*3])
        self.agree=torch.full_like(self.conf,.1)
        self.stab=torch.tensor([[1.]*4+[0.]*3+[1.]*3])

    def gate(self,cfg=None,filled=None):
        return build_row_gated_token_mask_v4(self.conf,self.agree,self.stab,self.ids,
            self.spans,torch.zeros_like(self.ids,dtype=torch.bool) if filled is None else filled,
            self.names,0,2,cfg or self.cfg,epoch=20,return_weight=True)

    def test_fallback_still_requires_stability(self):
        mask,weight=self.gate()
        self.assertTrue(mask[0,1:4].all())
        self.assertFalse(mask[0,4:7].any())
        torch.testing.assert_close(weight[0,1:4],torch.full((3,),.15))

    def test_disable_v_also_changes_fallback_admission(self):
        cfg=copy.deepcopy(self.cfg);cfg.mps_use_stability=False
        mask,_=self.gate(cfg)
        self.assertTrue(mask[0,1:7].all())

    def test_inserted_tokens_are_excluded(self):
        filled=torch.zeros_like(self.ids,dtype=torch.bool);filled[0,2]=True
        mask,_=self.gate(filled=filled)
        self.assertFalse(bool(mask[0,2]))

    def test_uniform_fallback_weight_cancels_within_row(self):
        mask,weight=self.gate()
        target=dict(padded_targets=torch.cat([torch.tensor([[3]]),self.ids],1),tr_spans=self.spans)
        logits=torch.arange(90,dtype=torch.float32).reshape(1,10,9)/13
        discounted=loss_structure_unsup_tr_block(dict(out_dec=logits),self.cfg,target,mask,weight)
        full=weight.clone();full[0,1:4]=1.
        ordinary=loss_structure_unsup_tr_block(dict(out_dec=logits),self.cfg,target,mask,full)
        torch.testing.assert_close(discounted,ordinary)
        manual=F.cross_entropy(logits[0,1:4],self.ids[0,1:4])
        torch.testing.assert_close(discounted[0],manual)

    def test_section_boundaries_have_no_loss_in_active_row_path(self):
        mask,weight=self.gate()
        target=dict(padded_targets=torch.cat([torch.tensor([[3]]),self.ids],1),tr_spans=self.spans)
        logits=torch.zeros(1,10,9,requires_grad=True)
        loss_structure_unsup_tr_block(dict(out_dec=logits),self.cfg,target,mask,weight).sum().backward()
        self.assertEqual(float(logits.grad[0,0].abs().sum()),0.)
        self.assertGreater(float(logits.grad[0,1:4].abs().sum()),0.)


class DataAndSPPTests(unittest.TestCase):
    def make_data(self,root,name,labeled=True):
        image=np.zeros((15,31,3),dtype=np.uint8);image[:,:,0]=255
        cv2.imwrite(str(root/(name+'.png')),image)
        record=dict(image=name+'.png')
        if labeled:record['structure']=['<tbody>','<tr>','<td></td>','</tr>','</tbody>']
        path=root/(name+'.jsonl');path.write_text(json.dumps(record)+'\n')
        return TableDataset(path,root,labeled=labeled)

    def test_preprocessing_geometry_and_channel_order(self):
        with tempfile.TemporaryDirectory() as folder:
            ds=self.make_data(Path(folder),'table')
            item=ds[0]
            self.assertEqual(tuple(item['img'].shape),(3,480,480))
            self.assertEqual(item['img_metas']['img_shape'],(232,480,3))
            self.assertEqual(float(item['img'][0,0,0]),1.)
            self.assertEqual(float(item['img'][2,0,0]),-1.)
            self.assertTrue((item['img'][:,300]==-1).all())

    def test_overlap_is_rejected(self):
        with tempfile.TemporaryDirectory() as folder:
            ds=self.make_data(Path(folder),'table')
            with self.assertRaises(ValueError):check_disjoint(ds,ds,ds)

    def test_sampler_and_loader_state_restore(self):
        with tempfile.TemporaryDirectory() as folder:
            ds=self.make_data(Path(folder),'table')
            chosen=int.from_bytes(os.urandom(4),'little')
            loader=make_loader(ds,1,chosen)
            state=capture_loader_states(dict(train=loader))
            first=list(iter(loader.sampler))
            restore_loader_states(dict(train=loader),state)
            self.assertEqual(first,list(iter(loader.sampler)))

    def test_spp_exchanges_regions_using_cached_boxes(self):
        boxes=[[10,10,30,30],[50,10,70,30]]
        with tempfile.TemporaryDirectory() as folder:
            path=Path(folder)/'boxes.jsonl'
            path.write_text(json.dumps(dict(img_path='table.png',boxes_xyxy_480=boxes))+'\n')
            cache=OCRBoxCache(str(path));self.assertEqual(cache.get_boxes('/images/table.png'),boxes)
            image=torch.zeros(1,3,480,480)
            image[:,:,10:30,10:30]=-.5;image[:,:,10:30,50:70]=.5
            cfg=Config(img_norm_cfg=dict(mean=[.5]*3,std=[.5]*3))
            result=strong_box_swap_tensor1(image,[dict(filename='table.png')],cache,cfg,
                                          p=1.,num_pairs=1,avoid_iou=.39)
            self.assertGreater(float((result-image).abs().sum()),1.)


class ModelAndMetricTests(unittest.TestCase):
    def test_vocabulary_preserves_unicode_separator_tokens(self):
        with tempfile.TemporaryDirectory() as folder:
            path=Path(folder)/'vocab.txt';path.write_text('a\n\u2028\nb\n',encoding='utf-8')
            self.assertEqual(list_from_file(path),['a','\u2028','b'])

    def test_model_teacher_forcing_and_greedy(self):
        cfg=Config.fromfile(ROOT/'configs/rps.py')
        cfg.model.max_seq_len=12
        cfg.model.label_convertor.max_seq_len=12
        model=build_model(cfg.model).eval()
        target=model.label_convertor.str2tensor([['<tbody>','<tr>','<td></td>','</tr>','</tbody>']])
        with torch.no_grad():
            output=model.forward_train(torch.zeros(1,3,32,32),[{}],targets_dict=target)
            greedy=model.forward_train(torch.zeros(1,3,32,32),[{}],targets_dict=None)
        self.assertEqual(tuple(output['out_dec'].shape),(1,11,43))
        self.assertEqual(tuple(greedy['t_ids'].shape),(1,12))
        self.assertTrue(torch.isfinite(output['out_dec']).all())

    def test_teds_ignores_cell_text(self):
        left='<tbody><tr><td>left</td></tr></tbody>'
        right='<tbody><tr><td>right</td></tr></tbody>'
        self.assertEqual(Analysis_TED(left,right),1.)

    def test_empty_cell_span_normalization_is_symmetric(self):
        left='<tbody>,<tr>,<eb></eb>, colspan="2",>,</td>,</tr>,</tbody>'
        right='<tbody><tr><td colspan="2"></td></tr></tbody>'
        self.assertEqual(Analysis_TED(normalize_structure_for_teds(left),
                                     normalize_structure_for_teds(right)),1.)


if __name__=='__main__':
    unittest.main()
