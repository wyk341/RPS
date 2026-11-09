"""Manifest-backed image datasets with table resize and top-left padding."""
import copy
import json
import math
from pathlib import Path

import cv2
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader, Sampler


class TableDataset(Dataset):
    def __init__(self, manifest, image_root, image_size=480, labeled=True):
        self.manifest = Path(manifest).resolve()
        self.image_root = Path(image_root).resolve()
        self.image_size = int(image_size)
        self.labeled = labeled
        self.records = []
        with self.manifest.open(encoding='utf-8') as stream:
            for number, line in enumerate(stream, 1):
                if not line.strip():
                    continue
                record = json.loads(line)
                if not isinstance(record.get('image'), str):
                    raise ValueError('Missing image field at manifest line {}'.format(number))
                if labeled and not isinstance(record.get('structure'), (list, str)):
                    raise ValueError('Missing structure field at manifest line {}'.format(number))
                self.records.append(record)
        if not self.records:
            raise ValueError('Image manifest is empty: {}'.format(self.manifest))
        names = [record['image'] for record in self.records]
        if len(set(names)) != len(names):
            raise ValueError('A manifest contains duplicate image paths.')
        # The group flag corresponds to the aspect ratio before padding.
        self.flag = np.asarray([self._shape_flag(record) for record in self.records], dtype=np.uint8)

    def path(self, record):
        return self.image_root / record['image']

    def _shape_flag(self, record):
        width, height = record.get('width'), record.get('height')
        if not width or not height:
            image = cv2.imread(str(self.path(record)), cv2.IMREAD_COLOR)
            if image is None:
                raise FileNotFoundError(str(self.path(record)))
            height, width = image.shape[:2]
        return int(width > height)

    def __len__(self):
        return len(self.records)

    def __getitem__(self, index):
        record = self.records[index]
        path = self.path(record)
        image = cv2.imread(str(path), cv2.IMREAD_COLOR)
        if image is None:
            raise FileNotFoundError(str(path))
        height, width = image.shape[:2]
        if width < height:
            resized_h, resized_w = self.image_size, max(1, int(self.image_size / height * width))
        else:
            resized_h, resized_w = max(1, int(self.image_size / width * height)), self.image_size
        resized = cv2.resize(image, (resized_w, resized_h), interpolation=cv2.INTER_LINEAR)
        padded = np.zeros((self.image_size, self.image_size, 3), dtype=np.uint8)
        padded[:resized_h, :resized_w] = resized
        tensor = torch.from_numpy(padded.transpose(2, 0, 1).copy()).float().div_(255.)
        tensor.sub_(.5).div_(.5)
        metadata = dict(filename=str(path), ori_filename=record['image'],
                        ori_shape=(height, width, 3), img_shape=(resized_h, resized_w, 3),
                        pad_shape=(self.image_size, self.image_size, 3),
                        scale_factor=(resized_h / height, resized_w / width),
                        img_norm_cfg=dict(mean=[.5]*3, std=[.5]*3))
        if self.labeled:
            tokens = record['structure']
            tokens = tokens.split(',') if isinstance(tokens, str) else list(tokens)
            if not tokens or not all(isinstance(token, str) for token in tokens):
                raise ValueError('Structure must contain vocabulary tokens: '+record['image'])
            metadata.update(text=','.join(tokens),
                            bbox=np.zeros((len(tokens), 4), dtype=np.float32),
                            bbox_masks=np.zeros(len(tokens), dtype=np.int64),
                            cell_content=[])
        return dict(img=tensor, img_metas=metadata)


def collate(batch):
    return dict(img=torch.stack([record['img'] for record in batch]),
                img_metas=[record['img_metas'] for record in batch])


class GroupSampler(Sampler):
    """Aspect-ratio groups with an isolated, checkpointable NumPy RNG."""
    def __init__(self, dataset, batch_size, seed):
        self.flag = np.asarray(dataset.flag)
        self.batch_size = int(batch_size)
        self.group_sizes = np.bincount(self.flag)
        self.num_samples = sum(int(math.ceil(size / self.batch_size))*self.batch_size
                               for size in self.group_sizes)
        self.state = np.random.RandomState(seed).get_state()

    def __len__(self):
        return self.num_samples

    def __iter__(self):
        rng = np.random.RandomState()
        rng.set_state(self.state)
        groups = []
        for group, size in enumerate(self.group_sizes):
            if not size:
                continue
            indices = np.where(self.flag == group)[0]
            rng.shuffle(indices)
            extra = int(math.ceil(size / self.batch_size))*self.batch_size-len(indices)
            indices = np.concatenate([indices, rng.choice(indices, extra)])
            groups.append(indices)
        indices = np.concatenate(groups)
        chunks = [indices[i*self.batch_size:(i+1)*self.batch_size]
                  for i in rng.permutation(len(indices)//self.batch_size)]
        self.state = rng.get_state()
        return iter(np.concatenate(chunks).astype(np.int64).tolist())


def make_loader(dataset, batch_size, seed, shuffle=True):
    sampler = GroupSampler(dataset, batch_size, seed) if shuffle else None
    return DataLoader(dataset, batch_size=batch_size, sampler=sampler, num_workers=0,
                      drop_last=False, collate_fn=collate,
                      generator=torch.Generator().manual_seed(seed))


def check_disjoint(labeled, unlabeled, validation):
    collections = [{str(dataset.path(record).resolve()) for record in dataset.records}
                   for dataset in (labeled, unlabeled, validation)]
    if any(collections[i] & collections[j] for i, j in ((0,1),(0,2),(1,2))):
        raise ValueError('Labeled, unlabeled, and validation image lists must be disjoint.')
