# Adapted from MTL-TabNet, TableMASTER, and OpenMMLab components.
# Changes: rps package imports and framework-independent integration.
# Upstream notices: THIRD_PARTY_NOTICES.md and licenses/.
import copy
import math
import torch
from torch import nn
import torch.nn.functional as F
from .base import BaseDecoder
from .position import PositionalEncoding

class Embeddings(nn.Module):

    def __init__(self, d_model, vocab):
        super().__init__()
        self.lut = nn.Embedding(vocab, d_model)
        self.d_model = d_model

    def forward(self, x):
        return self.lut(x) * math.sqrt(self.d_model)


def clones(module, N):
    return nn.ModuleList([copy.deepcopy(module) for _ in range(N)])


class SubLayerConnection(nn.Module):

    def __init__(self, size, dropout):
        super().__init__()
        self.norm = nn.LayerNorm(size)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x, sublayer):
        return x + self.dropout(sublayer(self.norm(x)))


class FeedForward(nn.Module):

    def __init__(self, d_model, d_ff, dropout):
        super().__init__()
        self.w_1 = nn.Linear(d_model, d_ff)
        self.w_2 = nn.Linear(d_ff, d_model)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        return self.w_2(self.dropout(F.relu(self.w_1(x))))


def self_attention(query, key, value, mask=None, dropout=None):
    d_k = value.size(-1)
    score = torch.matmul(query, key.transpose(-2, -1)) / math.sqrt(d_k)
    score = score.clamp(min=-50, max=50)
    if mask is not None:
        score = score.masked_fill(~mask, -50.0)
    p_attn = F.softmax(score, dim=-1)
    if dropout is not None:
        p_attn = dropout(p_attn)
    return torch.matmul(p_attn, value)


class MultiHeadAttention(nn.Module):

    def __init__(self, headers, d_model, dropout):
        super().__init__()
        assert d_model % headers == 0
        self.d_k = d_model // headers
        self.headers = headers
        self.linears = clones(nn.Linear(d_model, d_model), 4)
        self.dropout = nn.Dropout(dropout)

    def _shape(self, x, B):
        return x.view(B, -1, self.headers, self.d_k).transpose(1, 2
            ).contiguous()

    def forward(self, query, key, value, mask=None, cache=None, static_kv:
        bool=False):
        B = query.size(0)
        q = self._shape(self.linears[0](query), B)
        if cache is None:
            k = self._shape(self.linears[1](key), B)
            v = self._shape(self.linears[2](value), B)
        elif static_kv:
            if cache.get('k', None) is None or cache.get('v', None) is None:
                k = self._shape(self.linears[1](key), B)
                v = self._shape(self.linears[2](value), B)
                cache['k'], cache['v'] = k, v
            else:
                k, v = cache['k'], cache['v']
        else:
            k_new = self._shape(self.linears[1](key), B)
            v_new = self._shape(self.linears[2](value), B)
            if cache.get('k', None) is None or cache.get('v', None) is None:
                cache['k'], cache['v'] = k_new, v_new
            else:
                cache['k'] = torch.cat([cache['k'], k_new], dim=2)
                cache['v'] = torch.cat([cache['v'], v_new], dim=2)
            k, v = cache['k'], cache['v']
        x = self_attention(q, k, v, mask=mask, dropout=self.dropout)
        x = x.transpose(1, 2).contiguous().view(B, -1, self.headers * self.d_k)
        return self.linears[-1](x)


class MultiHeadAttentionCell(nn.Module):

    def __init__(self, headers, d_model, dropout):
        super().__init__()
        assert d_model % headers == 0
        self.d_k = d_model // headers
        self.headers = headers
        self.linears = clones(nn.Linear(d_model, d_model), 4)
        self.dropout = nn.Dropout(dropout)

    def forward(self, query, key, value, mask=None):
        nbatches = query.size(0)
        query, key, value = [l(x) for l, x in zip(self.linears, (query, key,
            value))]
        query = query.view(nbatches, -1, self.headers, self.d_k).transpose(1, 2
            )
        key = key.contiguous().view(1, -1, self.headers, self.d_k).transpose(
            1, 2)
        value = value.contiguous().view(1, -1, self.headers, self.d_k
            ).transpose(1, 2)
        x = self_attention(query, key, value, mask=mask, dropout=self.dropout)
        x = x.transpose(1, 2).contiguous().view(nbatches, -1, self.headers *
            self.d_k)
        return self.linears[-1](x)


class DecoderLayer(nn.Module):

    def __init__(self, size, self_attn, src_attn, feed_forward, dropout):
        super().__init__()
        self.size = size
        self.self_attn = MultiHeadAttention(**self_attn)
        self.src_attn = MultiHeadAttention(**src_attn)
        self.feed_forward = FeedForward(**feed_forward)
        self.sublayer = clones(SubLayerConnection(size, dropout), 3)

    def forward(self, x, feature, src_mask, tgt_mask, layer_cache=None,
        incremental: bool=False):
        if layer_cache is None:
            x = self.sublayer[0](x, lambda x: self.self_attn(x, x, x, tgt_mask)
                )
            x = self.sublayer[1](x, lambda x: self.src_attn(x, feature,
                feature, src_mask))
            return self.sublayer[2](x, self.feed_forward)
        self_cache = layer_cache.setdefault('self', {'k': None, 'v': None})
        cross_cache = layer_cache.setdefault('cross', {'k': None, 'v': None})
        x = self.sublayer[0](x, lambda x: self.self_attn(x, x, x, mask=None,
            cache=self_cache, static_kv=False))
        x = self.sublayer[1](x, lambda x: self.src_attn(x, feature, feature,
            mask=src_mask, cache=cross_cache, static_kv=True))
        return self.sublayer[2](x, self.feed_forward)


class DecoderLayerCell(nn.Module):

    def __init__(self, size, self_attn, src_attn, feed_forward, dropout):
        super().__init__()
        self.size = size
        self.self_attn = MultiHeadAttention(**self_attn)
        self.src_attn = MultiHeadAttentionCell(**src_attn)
        self.feed_forward = FeedForward(**feed_forward)
        self.sublayer = clones(SubLayerConnection(size, dropout), 3)

    def forward(self, x, feature, src_mask, tgt_mask):
        x = self.sublayer[0](x, lambda x: self.self_attn(x, x, x, tgt_mask))
        x = self.sublayer[1](x, lambda x: self.src_attn(x, feature, feature,
            src_mask))
        return self.sublayer[2](x, self.feed_forward)


class my_momodecoder(BaseDecoder):

    def __init__(self, N, decoder, d_model, num_classes, num_classes_cell,
        start_idx, padding_idx, end_idx, max_seq_len, start_idx_cell,
        padding_idx_cell, end_idx_cell, max_seq_len_cell, idx_tag_cell,
        decode_cell_content=False):
        super().__init__()
        self.max_length = max_seq_len
        self.decode_cell_content = bool(decode_cell_content)
        self.decode_bbox = False
        self.layers = clones(DecoderLayer(**decoder), N - 1)
        self.cls_layer = clones(DecoderLayer(**decoder), 1)
        self.bbox_layer = clones(DecoderLayer(**decoder), 1)
        self.cell_layer = clones(DecoderLayerCell(**decoder), 1)
        self.cls_fc = nn.Linear(d_model, num_classes)
        self.bbox_fc = nn.Sequential(nn.Linear(d_model, 4), nn.Sigmoid())
        self.cell_fc = nn.Linear(d_model, num_classes_cell)
        self.norm = nn.LayerNorm(decoder.size)
        self.embedding = Embeddings(d_model=d_model, vocab=num_classes)
        self.positional_encoding = PositionalEncoding(d_model=d_model)
        was_train = self.positional_encoding.training
        self.positional_encoding.eval()
        with torch.no_grad():
            dummy_x = torch.zeros(1, self.max_length + 1, d_model)
            pe_added = self.positional_encoding(dummy_x).clone()
        self.positional_encoding.train(was_train)
        self.register_buffer('_pos_cache', pe_added, persistent=False)
        self.embedding_cell = Embeddings(d_model=d_model, vocab=
            num_classes_cell)
        self.positional_encoding_cell = PositionalEncoding(d_model=d_model)
        self.SOS_CELL = start_idx_cell
        self.EOS_CELL = end_idx_cell
        self.PAD_CELL = padding_idx_cell
        self.max_length_cell = max_seq_len_cell
        self.cell_input_fc = nn.Linear(2 * d_model, d_model)
        self.batch_size_cell = 64
        self.idx_tag_cell = idx_tag_cell
        self.SOS = start_idx
        self.EOS = end_idx
        self.PAD = padding_idx
        self._last_t_ids = None
        self._last_t_logits = None
        self.va_enabled = True
        self._va_html_feats = None
        self._va_pos_index = None
        self.vis_proj = nn.Linear(512, d_model)
        self._va_text_feats = None

    def make_mask(self, src, tgt, padding_idx):
        tgt_len = tgt.size(1)
        trg_pad_mask = (tgt != padding_idx).unsqueeze(1).unsqueeze(3)
        trg_sub_mask = torch.tril(torch.ones((tgt_len, tgt_len), device=tgt
            .device, dtype=torch.bool))
        tgt_mask = trg_pad_mask & trg_sub_mask
        return None, tgt_mask

    def decode_test(self, input_ids, feature, src_mask):
        x = self.embedding(input_ids)
        x = self.positional_encoding(x)
        _, target_mask = self.make_mask(feature, input_ids, self.PAD)
        for layer in self.layers:
            x = layer(x, feature, src_mask, target_mask)
        for layer in self.cls_layer:
            cls_x = layer(x, feature, src_mask, target_mask)
        cls_x = self.norm(cls_x)
        tag_logits = self.cls_fc(cls_x)
        return tag_logits

    @torch.inference_mode()
    def greedy_forward_with_ids(self, out_enc, use_amp: bool=False):
        was_training = self.training
        if was_training:
            self.eval()
        try:
            device = out_enc.device
            B = out_enc.size(0)
            V = self.cls_fc.out_features
            Lmax = self.max_length
            logits_tok = torch.empty((B, Lmax, V), device=device, dtype=
                torch.float32)
            ids_tok = torch.full((B, Lmax), self.PAD, dtype=torch.long,
                device=device)
            n_layers = len(self.layers) + len(self.cls_layer)
            caches = [{'self': {'k': None, 'v': None}, 'cross': {'k': None,
                'v': None}} for _ in range(n_layers)]
            finished = torch.zeros((B,), dtype=torch.bool, device=device)
            prev = torch.full((B,), self.SOS, dtype=torch.long, device=device)
            pos = self._pos_cache
            if pos.device != device:
                pos = pos.to(device)
            amp_enabled = bool(use_amp) and out_enc.is_cuda
            with torch.cuda.amp.autocast(enabled=amp_enabled):
                for t in range(Lmax):
                    x = self.embedding(prev.unsqueeze(1))
                    x = x + pos[:, t:t + 1, :]
                    for i in range(len(self.layers)):
                        x = self.layers[i](x, out_enc, src_mask=None,
                            tgt_mask=None, layer_cache=caches[i],
                            incremental=True)
                    cls_x = self.cls_layer[0](x, out_enc, src_mask=None,
                        tgt_mask=None, layer_cache=caches[-1], incremental=True
                        )
                    cls_x = self.norm(cls_x)
                    last_logits = self.cls_fc(cls_x)[:, (-1), :]
                    next_tok = torch.argmax(last_logits, dim=-1)
                    next_tok = torch.where(finished, torch.full_like(
                        next_tok, self.PAD), next_tok)
                    logits_tok[:, (t), :] = last_logits.float()
                    ids_tok[:, (t)] = next_tok
                    just_finished = (next_tok == self.EOS) & ~finished
                    finished |= just_finished
                    if finished.all():
                        logits_tok = logits_tok[:, :t + 1, :]
                        ids_tok = ids_tok[:, :t + 1]
                        break
                    prev = next_tok
            return logits_tok, ids_tok
        finally:
            if was_training:
                self.train()

    @torch.inference_mode()
    def forward_pseudo(self, out_enc):
        logits_tok, ids_tok = self.greedy_forward_with_ids(out_enc)
        B, Lcur, V = logits_tok.shape
        Lmax = self.max_length
        logits_pad = logits_tok.new_full((B, Lmax, V), -50.0)
        ids_pad = torch.full((B, Lmax), self.PAD, dtype=torch.long, device=
            out_enc.device)
        copy_len = min(Lcur, Lmax)
        logits_pad[:, :copy_len, :] = logits_tok[:, :copy_len, :]
        ids_pad[:, :copy_len] = ids_tok[:, :copy_len]
        eos_mask = ids_pad == self.EOS
        after_or_at = eos_mask.cumsum(dim=1) >= 1
        after = after_or_at & ~eos_mask
        ids_pad[after] = self.PAD
        if 0 <= self.PAD < V:
            pad_pos = ids_pad == self.PAD
            logits_pad[pad_pos] = -50.0
            b_idx, t_idx = torch.where(pad_pos)
            if b_idx.numel() > 0:
                logits_pad[b_idx, t_idx, self.PAD] = 50.0
        return logits_pad, ids_pad

    def decode(self, input_ids, feature, src_mask):
        x = self.embedding(input_ids)
        x = self.positional_encoding(x)
        _, target_mask = self.make_mask(feature, input_ids, self.PAD)
        for layer in self.layers:
            x = layer(x, feature, src_mask, target_mask)
        for layer in self.cls_layer:
            cls_x = layer(x, feature, src_mask, target_mask)
        cls_x = self.norm(cls_x)
        tag_logits = self.cls_fc(cls_x)
        return tag_logits

    def forward_train(self, feat, out_enc, collect_va, targets_dict=None,
        img_metas=None):
        device = out_enc.device if out_enc is not None else feat.device
        if targets_dict is None:
            if out_enc is None:
                raise ValueError(
                    '[my_momodecoder] out_enc is None in pseudo mode.')
            t_logits, t_ids = self.forward_pseudo(out_enc)
            self._last_t_logits = t_logits
            self._last_t_ids = t_ids
            return t_logits, None, None
        padded_targets = targets_dict['padded_targets'].to(device)
        inp = padded_targets[:, :-1]
        tag_logits = self.decode(inp, out_enc, src_mask=None)
        self._last_t_logits = None
        self._last_t_ids = None
        return tag_logits, None, None

    def forward_test(self, feat, out_enc, img_metas):
        t_logits, t_ids = self.forward_pseudo(out_enc)
        self._last_t_logits = t_logits
        self._last_t_ids = t_ids
        return t_logits, None, None

    def forward(self, feat, out_enc, collect_va, targets_dict=None,
        img_metas=None, train_mode=False):
        self.train_mode = train_mode
        if train_mode:
            return self.forward_train(feat, out_enc, collect_va,
                targets_dict, img_metas)
        return self.forward_test(feat, out_enc, img_metas)
