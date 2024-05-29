import os
import pickle
from typing import Union, List

import torch
from PIL import Image
from avalanche.benchmarks.utils import AvalancheDataset
from torch.utils.data import Dataset

import numpy as np

# class FileListDataset(Dataset):
#     def __init__(self, filelist: List[str], targets: List[int]):
#         self.filelist = filelist
#         self.targets = targets
#         assert len(filelist) == len(targets)

#     def __len__(self):
#         return len(self.targets)

#     def __getitem__(self, item):
#         img = Image.open(self.filelist[item])
#         target = self.targets[item]
#         return img, target

class FileListDataset(Dataset):
    def __init__(self, filelist: List[str], targets: List[int], preprocess=False):
        self.filelist = filelist
        self.imgs = []
        self.preprocess = preprocess
        if preprocess:
            imgs = []
            for file in self.filelist:
                img = Image.open(file).convert("RGB")
                imgs.append(img)
            self.imgs = imgs
            
        self.targets = targets
        assert len(filelist) == len(targets)

    def __len__(self):
        return len(self.targets)

    def __getitem__(self, item):
        if self.preprocess:
            img = self.imgs[item]
        else:
            img = Image.open(self.filelist[item]).convert("RGB")
        target = self.targets[item]
        return img, target


class UnlabelledDataset(Dataset):
    def __init__(self, dataset: Dataset):
        self.dataset = dataset

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, index):
        return self.dataset[index][0],  # just return Data needs to be tuple otherwise TransformGroups unpack is not working!


class UnlabelledAvalancheDataset(AvalancheDataset):
    def __getitem__(self, idx: Union[int, slice]):
        res = super().__getitem__(idx)
        if len(res) == 1:
            return res[0]  # unpack!
        return res

class MemoryDataset(Dataset):
    def __init__(self, transform=None, device='cpu'):
        self.datalist = []
        self.logits = []
        self.labels = []
        self.transform = transform
        self.device=device
        
        # if save multiple data per class
        self.cls_list = []
        self.cls_count = []
        self.cls_idx = []
        self.cls_train_cnt = np.array([])
    
    def __len__(self):
        return len(self.datalist)

    def add_new_class(self, cls_list):
        self.cls_list = cls_list
        self.cls_count.append(0)
        self.cls_idx.append([])
        self.cls_train_cnt = np.append(self.cls_train_cnt, 0)
    
    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.value()
        label = self.labels[idx]
        logit = self.logits[idx]
        data = self.datalist[idx]
        if self.transform:
            data = self.transform(data)
        return data, logit, label
    
    def replace_sample(self, sample, label, idx=None):
        self.cls_count[self.cls_list.index(label)] += 1
        if idx is None:
            self.cls_idx[self.cls_list.index(label)].append(len(self.datalist))
            self.datalist.append(sample[0])
            self.logits.append(sample[1])
            self.labels.append(label)
            return len(self.datalist) - 1
        else:
            self.cls_count[self.cls_list.index(self.labels[idx])] -= 1
            self.cls_idx[self.cls_list.index(self.labels[idx])].remove(idx)
            self.datalist[idx] = sample[0]
            self.logits[idx] = sample[1]
            self.cls_idx[self.cls_list.index(label)].append(idx)
            self.labels[idx] = label
            return idx
            
    @torch.no_grad()
    def get_batch(self, batch_size, transform=None):
        batch_size = min(batch_size, len(self.datalist))
        if batch_size > 0:
            indices = np.random.choice(range(len(self.datalist)), size=batch_size, replace=False)
        datas = []
        logits = []
        labels = []
        if batch_size > 0:
            for i in indices:
                datas.append(self.datalist[i])
                logits.append(self.logits[i])
                labels.append(self.labels[i])
                # self.cls_train_cnt[self.labels[i]] += 1
                # self.usage_cnt[i] += 1

            datas = torch.stack(datas)
            logits = torch.stack(logits)
            labels = torch.stack(labels)
        return datas, logits, labels