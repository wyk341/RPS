# Adapted from MTL-TabNet, TableMASTER, and OpenMMLab components.
# Changes: rps package imports and framework-independent integration.
# Upstream notices: THIRD_PARTY_NOTICES.md and licenses/.
import torch
import numpy as np
from rps import utils
from .convertor_base import BaseConvertor

class MasterConvertor(BaseConvertor):

    def __init__(self, dict_type='DICT90', dict_file=None, dict_list=None,
        with_unknown=True, max_seq_len=40, lower=False, start_end_same=True,
        **kwargs):
        super().__init__(dict_type, dict_file, dict_list)
        assert isinstance(with_unknown, bool)
        assert isinstance(max_seq_len, int)
        assert isinstance(lower, bool)
        self.with_unknown = with_unknown
        self.max_seq_len = max_seq_len
        self.lower = lower
        self.start_end_same = start_end_same
        self.update_dict()

    def update_dict(self):
        start_token = '<SOS>'
        end_token = '<EOS>'
        unknown_token = '<UKN>'
        padding_token = '<PAD>'
        self.unknown_idx = None
        if self.with_unknown:
            self.idx2char.append(unknown_token)
            self.unknown_idx = len(self.idx2char) - 1
        self.idx2char.append(start_token)
        self.start_idx = len(self.idx2char) - 1
        if not self.start_end_same:
            self.idx2char.append(end_token)
        self.end_idx = len(self.idx2char) - 1
        self.idx2char.append(padding_token)
        self.padding_idx = len(self.idx2char) - 1
        self.char2idx = {}
        for idx, char in enumerate(self.idx2char):
            self.char2idx[char] = idx

    def str2tensor(self, strings):
        assert utils.is_type_list(strings, str) or utils.is_type_list(strings,
            list)
        tensors, padded_targets = [], []
        indexes = self.str2idx(strings)
        for index in indexes:
            tensor = torch.LongTensor(index)
            tensors.append(tensor)
            src_target = torch.LongTensor(tensor.size(0) + 2).fill_(0)
            src_target[-1] = self.end_idx
            src_target[0] = self.start_idx
            src_target[1:-1] = tensor
            padded_target = (torch.ones(self.max_seq_len) * self.padding_idx
                ).long()
            char_num = src_target.size(0)
            if char_num > self.max_seq_len:
                padded_target = src_target[:self.max_seq_len]
            else:
                padded_target[:char_num] = src_target
            padded_targets.append(padded_target)
        padded_targets = torch.stack(padded_targets, 0).long()
        return {'targets': tensors, 'padded_targets': padded_targets}

    def tensor2idx(self, outputs, img_metas=None):
        batch_size = outputs.size(0)
        ignore_indexes = [self.padding_idx]
        indexes, scores = [], []
        for idx in range(batch_size):
            seq = outputs[(idx), :, :]
            seq = seq.softmax(-1)
            max_value, max_idx = torch.max(seq, -1)
            str_index, str_score = [], []
            output_index = max_idx.cpu().detach().numpy().tolist()
            output_score = max_value.cpu().detach().numpy().tolist()
            for char_index, char_score in zip(output_index, output_score):
                if char_index in ignore_indexes:
                    continue
                if char_index == self.end_idx:
                    break
                str_index.append(char_index)
                str_score.append(char_score)
            indexes.append(str_index)
            scores.append(str_score)
        return indexes, scores


class TableMasterConvertor(MasterConvertor):

    def __init__(self, dict_type='DICT90', dict_file=None, dict_list=None,
        with_unknown=True, max_seq_len=500, lower=False, start_end_same=
        False, cell_dict_file=None, max_seq_len_cell=100, **kwargs):
        self.start_end_same = start_end_same
        self.checker()
        super().__init__(dict_type, dict_file, dict_list, with_unknown,
            max_seq_len, lower, start_end_same)
        self.max_seq_len_cell = max_seq_len_cell
        assert cell_dict_file is None or isinstance(cell_dict_file, str)
        self.idx2char_cell = []
        if cell_dict_file is not None:
            for line in utils.list_from_file(cell_dict_file):
                line = line.strip('\n')
                if line != '':
                    self.idx2char_cell.append(line)
        self.char2idx_cell = {}
        for idx, char in enumerate(self.idx2char_cell):
            self.char2idx_cell[char] = idx
        self.update_cell_dict()

    def num_classes_cell(self):
        return len(self.idx2char_cell)

    def update_cell_dict(self):
        start_token = '<SOS>'
        end_token = '<EOS>'
        unknown_token = '<UKN>'
        padding_token = '<PAD>'
        self.unknown_idx_cell = None
        if self.with_unknown:
            self.idx2char_cell.append(unknown_token)
            self.unknown_idx_cell = len(self.idx2char_cell) - 1
        self.idx2char_cell.append(start_token)
        self.start_idx_cell = len(self.idx2char_cell) - 1
        if not self.start_end_same:
            self.idx2char_cell.append(end_token)
        self.end_idx_cell = len(self.idx2char_cell) - 1
        self.idx2char_cell.append(padding_token)
        self.padding_idx_cell = len(self.idx2char_cell) - 1
        self.char2idx_cell = {}
        for idx, char in enumerate(self.idx2char_cell):
            self.char2idx_cell[char] = idx

    def checker(self):
        try:
            assert self.start_end_same is False
        except AssertionError:
            raise

    def _pad_bbox(self, bboxes):
        padded_bboxes = []
        for bbox in bboxes:
            bbox = torch.from_numpy(bbox)
            bbox_pad = torch.Tensor([0.0, 0.0, 0.0, 0.0]).float()
            padded_bbox = torch.zeros(self.max_seq_len, 4)
            padded_bbox[:] = bbox_pad
            if bbox.shape[0] > self.max_seq_len - 2:
                padded_bbox[1:self.max_seq_len - 1] = bbox[:self.
                    max_seq_len - 2]
            else:
                padded_bbox[1:len(bbox) + 1] = bbox
            padded_bboxes.append(padded_bbox)
        padded_bboxes = torch.stack(padded_bboxes, 0).float()
        return padded_bboxes

    def _pad_bbox_mask(self, bbox_masks):
        padded_bbox_masks = []
        for bbox_mask in bbox_masks:
            bbox_mask = torch.from_numpy(bbox_mask)
            bbox_mask_pad = torch.Tensor([0])
            padded_bbox_mask = torch.zeros(self.max_seq_len)
            padded_bbox_mask[:] = bbox_mask_pad
            if bbox_mask.shape[0] > self.max_seq_len - 2:
                padded_bbox_mask[1:self.max_seq_len - 1] = bbox_mask[:self.
                    max_seq_len - 2]
            else:
                padded_bbox_mask[1:len(bbox_mask) + 1] = bbox_mask
            padded_bbox_masks.append(padded_bbox_mask)
        padded_bbox_masks = torch.stack(padded_bbox_masks, 0).long()
        return padded_bbox_masks

    def idx2str(self, indexes):
        assert isinstance(indexes, list)
        strings = []
        for index in indexes:
            string = [self.idx2char[i] for i in index]
            string = ','.join(string)
            strings.append(string)
        return strings

    def idx2str_cell(self, indexes):
        assert isinstance(indexes, list)
        strings = []
        for index in indexes:
            string = [self.idx2char_cell[i] for i in index]
            string = ''.join(string)
            strings.append(string)
        return strings

    def str2idx_cell(self, strings):
        assert isinstance(strings, list)
        indexes = []
        for string in strings:
            if self.lower:
                string = string.lower()
            index = []
            for char in string:
                char_idx = self.char2idx_cell.get(char, self.unknown_idx_cell)
                if char_idx is None:
                    raise Exception(
                        f'Chararcter: {char} not in dict, please check gt_label and use custom dict file, or set "with_unknown=True"'
                        )
                index.append(char_idx)
            indexes.append(index)
        return indexes

    def str2tensor_cell(self, strings):
        assert utils.is_type_list(strings, list)
        cell_targets, cell_padded_targets = [], []
        indexes = [self.str2idx_cell(item) for item in strings]
        for item_sample in indexes:
            item_targets = []
            item_padded_targets = []
            for index in item_sample:
                tensor = torch.LongTensor(index)
                item_targets.append(tensor)
                src_target = torch.LongTensor(tensor.size(0) + 2).fill_(0)
                src_target[-1] = self.end_idx_cell
                src_target[0] = self.start_idx_cell
                src_target[1:-1] = tensor
                padded_target = (torch.ones(self.max_seq_len_cell) * self.
                    padding_idx_cell).long()
                char_num = src_target.size(0)
                if char_num > self.max_seq_len_cell:
                    padded_target = src_target[:self.max_seq_len_cell].clone()
                    padded_target[-1] = self.end_idx_cell
                else:
                    padded_target[:char_num] = src_target
                item_padded_targets.append(padded_target)
            cell_targets.append(item_targets)
            cell_padded_targets.append(item_padded_targets)
        return {'cell_targets': cell_targets, 'cell_padded_targets':
            cell_padded_targets}

    def idx_tag_cell(self):
        idx_tag_cell = [self.char2idx['<td></td>'], self.char2idx['<td']]
        return idx_tag_cell

    def tensor2idx_cell(self, outputs, img_metas=None):
        batch_size = outputs.size(0)
        ignore_indexes = [self.padding_idx_cell]
        indexes, scores = [], []
        for idx in range(batch_size):
            seq = outputs[(idx), :, :]
            seq = seq.softmax(-1)
            max_value, max_idx = torch.max(seq, -1)
            str_index, str_score = [], []
            output_index = max_idx.cpu().detach().numpy().tolist()
            output_score = max_value.cpu().detach().numpy().tolist()
            for char_index, char_score in zip(output_index, output_score):
                if char_index in ignore_indexes:
                    continue
                if char_index == self.end_idx_cell:
                    break
                str_index.append(char_index)
                str_score.append(char_score)
            indexes.append(str_index)
            scores.append(str_score)
        return indexes, scores

    def _get_pred_bbox_mask(self, strings):
        assert isinstance(strings, list)
        pred_bbox_masks = []
        SOS = self.idx2char[self.start_idx]
        EOS = self.idx2char[self.end_idx]
        PAD = self.idx2char[self.padding_idx]
        for string in strings:
            pred_bbox_mask = []
            char_list = string.split(',')
            for char in char_list:
                if char == EOS:
                    pred_bbox_mask.append(0)
                    break
                elif char == PAD:
                    pred_bbox_mask.append(0)
                    continue
                elif char == SOS:
                    pred_bbox_mask.append(0)
                    continue
                elif char == '<td></td>' or char == '<td':
                    pred_bbox_mask.append(1)
                else:
                    pred_bbox_mask.append(0)
            pred_bbox_masks.append(pred_bbox_mask)
        return [np.asarray(m, dtype=np.int64) for m in pred_bbox_masks]

    def _filter_invalid_bbox(self, output_bbox, pred_bbox_mask):
        low_mask = (output_bbox >= 0.0) * 1
        high_mask = (output_bbox <= 1.0) * 1
        mask = np.sum(low_mask + high_mask, axis=1)
        value_mask = np.where(mask == 2 * 4, 1, 0)
        output_bbox_len = output_bbox.shape[0]
        pred_bbox_mask_len = pred_bbox_mask.shape[0]
        padded_pred_bbox_mask = np.zeros(output_bbox_len, dtype='int64')
        padded_pred_bbox_mask[:pred_bbox_mask_len] = pred_bbox_mask
        filtered_output_bbox = output_bbox * np.expand_dims(value_mask, 1
            ) * np.expand_dims(padded_pred_bbox_mask, 1)
        return filtered_output_bbox

    def _decode_bboxes(self, outputs_bbox, pred_bbox_masks, img_metas):
        pred_bboxes = []
        for output_bbox, pred_bbox_mask, img_meta in zip(outputs_bbox,
            pred_bbox_masks, img_metas):
            output_bbox = output_bbox.detach().cpu().numpy()
            scale_factor = img_meta['scale_factor']
            pad_shape = img_meta['pad_shape']
            ori_shape = img_meta['ori_shape']
            output_bbox = self._filter_invalid_bbox(output_bbox, pred_bbox_mask
                )
            output_bbox[:, 0::2] = output_bbox[:, 0::2] * pad_shape[1]
            output_bbox[:, 1::2] = output_bbox[:, 1::2] * pad_shape[0]
            output_bbox[:, 0::2] = output_bbox[:, 0::2] / scale_factor[1]
            output_bbox[:, 1::2] = output_bbox[:, 1::2] / scale_factor[0]
            pred_bboxes.append(output_bbox)
        return pred_bboxes

    def _adjsut_bboxes_len(self, bboxes, strings):
        new_bboxes = []
        for bbox, string in zip(bboxes, strings):
            string = string.split(',')
            string_len = len(string)
            bbox = bbox[:string_len, :]
            new_bboxes.append(bbox)
        return new_bboxes

    def _get_strings_scores(self, str_scores):
        strings_scores = []
        for str_score in str_scores:
            if len(str_score) > 0:
                score = sum(str_score) / len(str_score)
            else:
                score = 0.0
            strings_scores.append(score)
        return strings_scores

    def str_bbox_format(self, img_metas):
        gt_labels = [[char for char in img_meta['text'].split(',')] for
            img_meta in img_metas]
        tmp_dict = self.str2tensor(gt_labels)
        text_target = tmp_dict['targets']
        text_padded_target = tmp_dict['padded_targets']
        bboxes = [img_meta['bbox'] for img_meta in img_metas]
        bboxes = self._pad_bbox(bboxes)
        bbox_masks = [img_meta['bbox_masks'] for img_meta in img_metas]
        bbox_masks = self._pad_bbox_mask(bbox_masks)
        gt_cell_contents = [img_meta['cell_content'] for img_meta in img_metas]
        tmp_dict_cell = self.str2tensor_cell(gt_cell_contents)
        cell_targets = tmp_dict_cell['cell_targets']
        cell_padded_targets = tmp_dict_cell['cell_padded_targets']
        format_dict = {'targets': text_target, 'padded_targets':
            text_padded_target, 'bbox': bboxes, 'bbox_masks': bbox_masks,
            'cell_targets': cell_targets, 'cell_padded_targets':
            cell_padded_targets}
        return format_dict

    def output_format(self, outputs, out_bbox, out_cell, img_metas=None):
        str_indexes, str_scores = self.tensor2idx(outputs, img_metas)
        strings = self.idx2str(str_indexes)
        scores = self._get_strings_scores(str_scores)
        pred_bbox_masks = self._get_pred_bbox_mask(strings)
        pred_bboxes = self._decode_bboxes(out_bbox, pred_bbox_masks, img_metas)
        pred_bboxes = self._adjsut_bboxes_len(pred_bboxes, strings)
        cell_strings = []
        cell_scores = []
        for idx_, out_cell_i in enumerate(out_cell):
            Nc = out_cell_i.size(0)
            if Nc == 0:
                cell_strings_i = []
                cell_scores_i = []
            else:
                str_cell_indexes, str_cell_scores = self.tensor2idx_cell(
                    out_cell_i, img_metas)
                cell_strings_i = self.idx2str_cell(str_cell_indexes)
                cell_scores_i = self._get_strings_scores(str_cell_scores)
            cell_strings.append(cell_strings_i)
            cell_scores.append(cell_scores_i)
        return strings, scores, pred_bboxes, cell_strings, cell_scores
