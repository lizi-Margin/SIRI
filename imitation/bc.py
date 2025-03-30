import os
import copy
import torch
import shutil, random
import numpy as np
import torch.optim as optim
from torch.distributions.categorical import Categorical
from UTIL.tensor_ops import _2tensor
from UTIL.colorful import *
from imitation.utils import print_dict

from imitation_bc.bc import AlgorithmConfig, TrainerBase

class wasd_xy_Trainer(TrainerBase):
    def train_on_data_(self, data: dict):
        """ BC """
        num_epoch = AlgorithmConfig.num_epoch_per_update
        assert 'obs' in data
        all_obs = data.pop('obs')
        assert 'obs' not in data

        for epoch in range(num_epoch):
            N = len(next(iter(data.values())))
            n = min(AlgorithmConfig.sample_size, N)
            start = np.random.choice(max(N-n, 1))

            if isinstance(all_obs, (tuple, list,)): obs = tuple([_2tensor(f[start:start+n]) for f in all_obs])
            else: obs = _2tensor(all_obs[start:start+n])
            act = {key: value[start:start+n] for key, value in data.items()}

            self.optimizer.zero_grad()

            index_x = _2tensor(act['x'])
            index_y = _2tensor(act['y'])
            index_wasd = _2tensor(act['wasd'])

            logit_wasd, logit_x, logit_y = self.policy(obs)
            dist_wasd = Categorical(logits=logit_wasd)
            dist_x = Categorical(logits=logit_x)
            dist_y = Categorical(logits=logit_y)

            actLogProbs = dist_wasd.log_prob(index_wasd) + dist_x.log_prob(index_x) + dist_y.log_prob(index_y)
            distEntropy = dist_wasd.entropy() + dist_x.entropy() + dist_y.entropy()

            cross_entropy_loss = -actLogProbs.mean() # mean log probility of the expert actions -> cross entropy loss -> max likelyhood
            mean_distEntropy = distEntropy.mean()
            dist_entropy_loss = -mean_distEntropy * AlgorithmConfig.dist_entropy_loss_coef

            loss = cross_entropy_loss + dist_entropy_loss

            if epoch==0: print('[bc.py] Memory Allocated %.2f GB'%(torch.cuda.memory_allocated()/1073741824))
            loss.backward()
            self.optimizer.step()
            if self.sheduler:
                self.sheduler.step()
                current_lr = self.sheduler.get_last_lr()[0]
            else:
                current_lr = AlgorithmConfig.lr

            self.epoch_cnt += 1
            print(f"bc_train: epoch{self.epoch_cnt} finished, current lr: {round(current_lr, 5)}")
            # print(f"bc_train: epoch{self.epoch_cnt} finished, current lr: {current_lr}")

            log = {
                "Cross Entropy Loss": float(cross_entropy_loss.detach().to("cpu").numpy()),
                "Cross Entropy Loss clip 5": float(np.clip(cross_entropy_loss.detach().to("cpu").numpy(), -10, 5)),
                "Dist Entropy Loss (Regulization)": float(dist_entropy_loss.detach().to("cpu").numpy()),
                "lr": float(current_lr),
                "Dist Entropy": float(mean_distEntropy.detach().to("cpu").numpy())
            }
            self.logs.append(log)
            # print_dict(log)
            # print(str(log))
            self.log_trivial(dictionary=log)
            self.log_trivial_finalize()
            self.mcv.rec(self.epoch_cnt, 'time')
            self.mcv.rec_show()
        # torch.cuda.empty_cache()
                
        assert self.epoch_cnt%AlgorithmConfig.num_epoch_per_update == 0
        update_cnt = int(self.epoch_cnt/AlgorithmConfig.num_epoch_per_update)
        print(f"update {update_cnt} finished")
        # self.save_model(self.epoch_cnt)
        return update_cnt

