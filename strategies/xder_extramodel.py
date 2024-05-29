import copy
import torch
import numpy as np
import torch.nn as nn
import torch.nn.functional as F
from collections import OrderedDict
from torch.utils.data import DataLoader
from utils.randaug import DataAugmentation
from benchmarks.benchmark_utils import MemoryDataset
from strategies.competition_template import CompetitionTemplate
from avalanche.training.plugins.evaluation import default_evaluator
from avalanche.benchmarks.utils.data_loader import (
    TaskBalancedDataLoader,
)
import random

DATALOADER_SEED = 42 # random.randint()

class xder_extramodel(CompetitionTemplate):
    """
    xder + ace + distill + ssl (36.711)
    """
    def __init__(self, model, optimizer, criterion=nn.CrossEntropyLoss(), train_mb_size=1,
                 train_epochs=1, eval_mb_size=1, device="cpu",
                 plugins=None, evaluator=default_evaluator(), eval_every=-1,
                 peval_mode="epoch", ):
        super().__init__(model, optimizer, criterion, train_mb_size, train_epochs, eval_mb_size, device,
                         plugins, evaluator, eval_every, peval_mode, )
        
        self.randaug = DataAugmentation(224, (0.485, 0.456, 0.406), (0.229, 0.224, 0.225)).to(self.device)
        self.lr = self.optimizer.param_groups[0]['lr']
        self.exposed_cls = []
        
        self.memory = MemoryDatasetWithLogits(device=self.device)
        self.mem_size = 200
        self.mem_batch_size = 10
        self.memory_loader = None
        
        self.model.Rot = torch.nn.Linear(512, 4).to(self.device)
        self.sdp_model = copy.deepcopy(self.model).to(self.device)
        self.prev_model_list = []
        
        self.ema_ratio = 0.996
        self.der_weight = 0.4
        self.ssl_weight = 0.5
        self.distill_weight = 0.002
    
    def make_train_dataloader(
        self,
        num_workers=0,
        shuffle=True,
        pin_memory=None,
        persistent_workers=False,
        drop_last=False,
        **kwargs
    ):

        assert self.adapted_dataset is not None

        torch.utils.data.DataLoader

        # generator
        def seed_worker(worker_id):
            worker_seed = torch.initial_seed() % 2**32
            np.random.seed(worker_seed)
            random.seed(worker_seed)

        g = torch.Generator()
        g.manual_seed(DATALOADER_SEED)

        other_dataloader_args = self._obtain_common_dataloader_parameters(
            batch_size=self.train_mb_size,
            num_workers=num_workers,
            shuffle=shuffle,
            pin_memory=pin_memory,
            persistent_workers=persistent_workers,
            drop_last=drop_last,
            worker_init_fn=seed_worker,
            generator=g,
        )

        self.dataloader = TaskBalancedDataLoader(
            self.adapted_dataset, oversample_small_groups=True, **other_dataloader_args
        )

    def training_epoch(self, **kwargs):
        super().training_epoch(**kwargs)

    def _before_training_exp(self, **kwargs):      
        super()._before_training_exp(**kwargs)
        # generator
        def seed_worker(worker_id):
            worker_seed = torch.initial_seed() % 2**32
            np.random.seed(worker_seed)
            random.seed(worker_seed)

        g = torch.Generator()
        g.manual_seed(DATALOADER_SEED)
        
        self.unlabelled_dl = DataLoader(self.unlabelled_ds, batch_size=100, shuffle=True, num_workers=kwargs['num_workers'], worker_init_fn=seed_worker, generator=g)
        
        self.unlabelled_iterator = iter(self.unlabelled_dl)

        if self.experience.current_experience > 1:
            self.prev_model_list.append(copy.deepcopy(self.sdp_model))
            if len(self.prev_model_list) > 2:
                del self.prev_model_list[0]

        self.sdp_model = copy.deepcopy(self.model).to(self.device)

        self.cur_task = self.experience.current_experience
        self.cur_cls = self.experience.classes_in_this_experience
        self.prev_cls = self.exposed_cls[:]
        self.seen_cls = self.experience.classes_seen_so_far
        self.unseen_cls = list(set(range(100)) - set(self.seen_cls))
        for cls in self.cur_cls:
            if cls not in self.exposed_cls:
                self.exposed_cls.append(cls)
                self.memory.add_new_class(self.exposed_cls)

        if self.train_epochs > 25:
            self.train_epochs -= 1
        
    def _after_training_exp(self, **kwargs):
        super()._after_training_exp(**kwargs)
        self.ssl_weight *= 0.95
        self.distill_weight += 0.002
        with torch.no_grad():
            for idx, (x, y, _) in enumerate(self.dataloader):
                logit, feat = self.model(self.randaug(x.to(self.device), random=False), get_feature=True)
                logit[:, self.unseen_cls] = 1e-9
                for idx in range(len(logit)):
                    self.update_memory(feat[idx].detach().cpu(), y[idx].detach().cpu(), logit[idx].detach())
        
    def forward(self):
        self.model.train()
        self.sdp_model.train()
        for prevmodel in self.prev_model_list:
            prevmodel.train()
        x = self.randaug(self.mbatch[0])
        self.y = self.mbatch[1]
        self.logit, self.feature = self.model(x, get_feature=True)
        self.xder_logit = self.logit.clone()
        self.logit[:, self.unseen_cls] = 1e-9
        
        invalid_idx = []
        for idx in range(self.logit.shape[1]):
            if idx in self.prev_cls and idx not in self.y:
                invalid_idx.append(idx)
        self.logit[:, invalid_idx] = -1e9
        
        self.bs = min(len(self.memory), self.mem_batch_size)
        if self.bs > 0:
            mem_feat, mem_y, self.mem_logit, self.mem_index = self.memory.get_batch(self.bs)
            mem_feat, mem_y, self.mem_logit, self.mem_index = mem_feat.to(self.device), mem_y.to(self.device), self.mem_logit.to(self.device), self.mem_index.to(self.device)
            mem_logit = self.model.fc(mem_feat)
            mem_logit[:, self.unseen_cls] = 1e-9
            self.y = torch.cat([self.y, mem_y])
            self.logit = torch.cat([self.logit, mem_logit])
    
        self.sdp_model.zero_grad()
        with torch.no_grad():
            logit2, self.feature2 = self.sdp_model(x, get_feature=True)
            logit2[:, self.unseen_cls] = 1e-9
            if len(self.prev_model_list) > 0:
                feature2_list = []
                logit2_list = []
                for prev_model in self.prev_model_list:
                    logit_prev, feature_prev = prev_model(x, get_feature=True)
                    logit_prev[:, self.unseen_cls] = 1e-9
                    feature2_list.append(feature_prev)
                    logit2_list.append(logit_prev)
                
                # decide best feature to distill
                cur_best_prob = F.softmax(logit2, 1)[list(range(self.feature2.shape[0])), self.y[:-self.bs]]
                for i in range(len(feature2_list)):
                    cur_prob = F.softmax(logit2_list[i], 1)[list(range(self.feature2.shape[0])), self.y[:-self.bs]]
                    idx = cur_best_prob < cur_prob
                    cur_best_prob[idx] = cur_prob[idx]
                    self.feature2[idx] = feature2_list[i][idx]

        return self.logit[:-self.bs] if self.bs > 0 else self.logit
    
    def criterion(self):
        if self.bs > 0:
            loss_current = F.cross_entropy(self.logit[:-self.bs], self.y[:-self.bs], reduction='none')
            loss_buffer = F.cross_entropy(self.logit[-self.bs:], self.y[-self.bs:], reduction='none')
            cls_loss = torch.cat((loss_current, loss_buffer), dim=0).mean()
            der_loss = self.der_weight * F.kl_div(F.log_softmax(self.logit[-self.bs:], dim=1), F.softmax(self.mem_logit, dim=1), reduction='none').sum(1).mean()
            xder_loss = self.get_logit_constraint_loss(self.xder_logit[:-self.bs])
        else:
            cls_loss = F.cross_entropy(self.logit, self.y)
            der_loss = 0
            xder_loss = self.get_logit_constraint_loss(self.xder_logit)

        distill_loss = 0
        distill_loss += self.distill_weight * ((self.feature - self.feature2.detach()) ** 2).sum(dim=1).mean()
        return (1-0.1*self.ssl_weight)*cls_loss + der_loss + xder_loss + distill_loss
    
    def distill_constraint_loss(self, feature, feature2):
        Temp_list = [ 1.0, ]
        final_loss = 0
        for tmp in Temp_list:
            logit = F.softmax(feature/tmp, dim=1)
            logit2 = F.softmax(feature2/tmp, dim=1)

            student_matrix = torch.mm(logit, logit2.transpose(1, 0))
            teacher_matrix = torch.mm(logit2, logit.transpose(1, 0))
            bc_loss = ((teacher_matrix - student_matrix) ** 2).sum() / logit.size(0)
            student_matrix = torch.mm(logit.transpose(1, 0), logit2)
            teacher_matrix = torch.mm(logit2.transpose(1, 0), logit)
            cc_loss = ((teacher_matrix - student_matrix) ** 2).sum() / logit.size(1)

            final_loss += bc_loss + cc_loss
    
        return final_loss
    
    def _before_backward(self, **kwargs):
        super()._before_backward(**kwargs)
        if self.unlabelled_iterator is not None:
            try:
                batch_unlabelled = next(self.unlabelled_iterator)
            except StopIteration:
                try:
                    self.unlabelled_iterator = iter(self.unlabelled_dl)
                    batch_unlabelled = next(self.unlabelled_iterator)
                except Exception:
                    self.unlabelled_iterator = None
                    batch_unlabelled = None
                    
            if batch_unlabelled is not None:
                teacher_batch = self.randaug(batch_unlabelled.to(self.device), random=False)
                batch_unlabelled = self.randaug(batch_unlabelled.to(self.device))
                rot_x = []
                rot_label = []
                x = batch_unlabelled[:batch_unlabelled.shape[0]//4]
                with torch.no_grad():
                    for k in range(4):
                        x = torch.rot90(x, 1, dims=[2,3])
                        rot_x.append(copy.deepcopy(x))
                        rot_label.append(torch.LongTensor(np.full(x.shape[0], k)).to(x.device))
                    rot_x = torch.stack(rot_x, dim=0)
                    rot_x = rot_x.reshape(rot_x.shape[0]*rot_x.shape[1], rot_x.shape[2], rot_x.shape[3], rot_x.shape[4])
                    rot_label = torch.stack(rot_label, dim=0)
                    rot_label = rot_label.reshape(rot_label.shape[0]*rot_label.shape[1])
                _, feat = self.model(rot_x, get_feature=True)
                out = self.model.Rot(feat)
                ssl_loss = F.cross_entropy(out, rot_label)
                self.loss += self.ssl_weight * ssl_loss

                if self.experience.current_experience > 0:
                    logit, feature = self.model(batch_unlabelled, get_feature=True)
                    logit[:, self.unseen_cls] = 1e-9
                    with torch.no_grad():
                        _, feature2 = self.sdp_model(batch_unlabelled, get_feature=True)
                        logit2 = self.sdp_model(teacher_batch)
                        logit2[:, self.unseen_cls] = 1e-9
                        if len(self.prev_model_list) > 0:
                            feature2_list = []
                            logit2_list = []
                            for prev_model in self.prev_model_list:
                                logit_prev, feature_prev = prev_model(batch_unlabelled, get_feature=True)
                                logit_prev[:, self.unseen_cls] = 1e-9
                                feature2_list.append(feature_prev)
                                logit2_list.append(logit_prev)
                            cur_best_prob = torch.max(F.softmax(logit2, 1),dim=1)[0]
                            for i in range(len(feature2_list)):
                                cur_prob = torch.max(F.softmax(logit2_list[i], 1), dim=1)[0]
                                idx = cur_best_prob < cur_prob
                                cur_best_prob[idx] = cur_prob[idx]
                                feature2[idx] = feature2_list[i][idx]
                                logit2[idx] = logit2_list[i][idx]

                    distill_loss = ((feature - feature2.detach()) ** 2).sum(dim=1)
                        
                    # distill_loss /= total
                    self.loss += self.distill_weight * distill_loss.mean()
                    self.loss += 0.1 * self.distill_constraint_loss(logit, logit2)
                    
    def _after_update(self, **kwargs):
        self.update_sdp_model()
        return super()._after_update(**kwargs)
        
    def update_memory(self, feature, label, logit=None):
        self.balanced_replace_memory(feature, label, logit)

    def balanced_replace_memory(self, feature, label, logit=None):
        if len(self.memory) >= self.mem_size:
            label_frequency = copy.deepcopy(self.memory.cls_count)
            label_frequency[self.exposed_cls.index(label)] += 1
            cls_to_replace = np.random.choice(
                np.flatnonzero(np.array(label_frequency) == np.array(label_frequency).max()))
            idx_to_replace = np.random.choice(self.memory.cls_idx[cls_to_replace])
            self.memory.replace_sample(feature, label, logit, idx_to_replace)
        else:
            self.memory.replace_sample(feature, label, logit)

    def get_logit_constraint_loss(self, outputs):
        loss_constr_past = torch.tensor(0.)
        if self.cur_task > 0:
            good_head = F.softmax(outputs[:, self.cur_cls], 1)
            bad_head = F.softmax(outputs[:, self.prev_cls], 1)
            loss_constr = bad_head.max(1)[0].detach() + 0.1 - good_head.max(1)[0]
            mask = loss_constr > 0
            if (mask).any():
                loss_constr_past = 0.01 * loss_constr[mask].mean()
            
        loss_constr_futu = torch.tensor(0.)
        if len(self.unseen_cls) > 0:
            good_head = outputs[:, self.cur_cls]
            bad_head = outputs[:, self.unseen_cls]
            loss_constr = bad_head.max(1)[0] + 0.1 - good_head.max(1)[0]
            mask = loss_constr > 0
            if (mask).any():
                loss_constr_futu = 0.01 * loss_constr[mask].mean()
        return loss_constr_past + loss_constr_futu

    def update_schedule(self):
        for param_group in self.optimizer.param_groups:
            param_group["initial_lr"] = self.lr * (1 - self.distill_weight)
    
    @torch.no_grad()
    def update_sdp_model(self):
        model_params = OrderedDict(self.model.named_parameters())
        ema_params = OrderedDict(self.sdp_model.named_parameters())
        for name, param in model_params.items():
            ema_params[name].copy_(self.ema_ratio * ema_params[name] +  (1-self.ema_ratio)* param)
        self.sdp_model.fc = copy.deepcopy(self.model.fc)

        model_buffers = OrderedDict(self.model.named_buffers())
        shadow_buffers = OrderedDict(self.sdp_model.named_buffers())
        for name, buffer in model_buffers.items():
            shadow_buffers[name].copy_(buffer)
        
        if len(self.prev_model_list) > 0:
            for prev_model in self.prev_model_list:
                model_params = OrderedDict(self.model.named_parameters())
                ema_params = OrderedDict(prev_model.named_parameters())
            
                assert model_params.keys() == ema_params.keys()
                for name, param in model_params.items():
                    ema_params[name].copy_(
                        self.ema_ratio * ema_params[name] +  (1-self.ema_ratio)* param)
                prev_model.fc = copy.deepcopy(self.model.fc)

                model_buffers = OrderedDict(self.model.named_buffers())
                shadow_buffers = OrderedDict(prev_model.named_buffers())

                assert model_buffers.keys() == shadow_buffers.keys()

                for name, buffer in model_buffers.items():
                    shadow_buffers[name].copy_(buffer)


class MemoryDatasetWithLogits(MemoryDataset):
    def __init__(self, transform=None, device='cpu'):
        super().__init__(transform, device)
        self.features = []
        self.logits = []

    def __len__(self):
        return len(self.features)
    
    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.value()
        feature = self.features[idx]
        label = self.labels[idx]
        logit = self.logits[idx]
        if self.transform:
            data = self.transform(data)
        return feature, label, logit, idx
    
    def update_logits(self, indices, logits):
        for logit, index in zip(logits, indices):
            self.logits[index] = logit

    def replace_sample(self, feature, label, logit=None, idx=None):
        self.cls_count[self.cls_list.index(label)] += 1
        if idx is None:
            self.cls_idx[self.cls_list.index(label)].append(len(self.features))
            self.features.append(feature)
            self.labels.append(label)
            self.logits.append(logit)
        else:
            self.cls_count[self.cls_list.index(self.labels[idx])] -= 1
            self.cls_idx[self.cls_list.index(self.labels[idx])].remove(idx)
            self.features[idx] = feature
            self.cls_idx[self.cls_list.index(label)].append(idx)
            self.labels[idx] = label
            self.logits[idx] = logit
            
    @torch.no_grad()
    def get_batch(self, batch_size):
        batch_size = min(batch_size, len(self.features))
        if batch_size > 0:
            indices = np.random.choice(range(len(self.features)), size=batch_size, replace=False)
        features = []
        labels = []
        logits = []
        if batch_size > 0:
            for i in indices:
                features.append(self.features[i])
                labels.append(self.labels[i])
                logits.append(self.logits[i])
            features = torch.stack(features)
            labels = torch.stack(labels)
            logits = torch.stack(logits)
            indices = torch.tensor(indices)
        return features, labels, logits, indices
        