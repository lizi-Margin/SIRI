import os
import copy
import torch
import shutil, random
import numpy as np
import torch.optim as optim
from torch.distributions.categorical import Categorical
from torch.distributions.bernoulli import Bernoulli
from UTIL.tensor_ops import _2tensor
from UTIL.colorful import *
from imitation.utils import print_dict
from imitation.bc import AlgorithmConfig, wasd_xy_Trainer
from siri.utils.logger import lprint_

def t3n(x):
    return float(x.detach().mean().to("cpu").numpy())

class FullTrainer(wasd_xy_Trainer):
    def __init__(self, policy):
        super().__init__(policy)
        self.scaler = torch.amp.GradScaler('cuda', init_scale = 2.0**16)

    def train_on_data_(self, data: dict):
        """ BC """
        num_epoch = AlgorithmConfig.num_epoch_per_update
        assert 'obs' in data
        all_obs = data.pop('obs')
        assert 'obs' not in data
        sample_size = AlgorithmConfig.sample_size
        for epoch in range(num_epoch):
            N = len(next(iter(data.values())))
            not_pass = True
            while not_pass:
                n = min(sample_size, N)
                start = np.random.choice(max(N-n, 1))

                if isinstance(all_obs, (tuple, list,)): obs = tuple([_2tensor(f[start:start+n]) for f in all_obs])
                else: obs = _2tensor(all_obs[start:start+n])
                act = {key: value[start:start+n] for key, value in data.items()}

                self.optimizer.zero_grad()

                try:
                    with torch.amp.autocast_mode.autocast("cuda", dtype=torch.float16):
                        loss, log = self.establish_torch_graph(obs, act)

                    if epoch==0: print('[bc.py] Memory Allocated %.2f GB'%(torch.cuda.memory_allocated()/1073741824))
                    self.scaler.scale(loss).backward()
                    # del loss; torch.cuda.empty_cache()
                    self.scaler.step(self.optimizer)
                    self.scaler.update()
                    del loss; torch.cuda.empty_cache()
                    not_pass = False
                except torch.OutOfMemoryError:
                    print亮红(lprint_(self, f"Error: cuda out of memory, sample_size={sample_size}"))
                    sample_size = int(sample_size * 0.9)
                    if sample_size < AlgorithmConfig.sample_size/10:
                        raise torch.OutOfMemoryError
                    print亮红(lprint_(self, f"adjust sample_size to {sample_size}"))
                finally:
                    del obs, act
                

            if self.sheduler:
                self.sheduler.step()
                current_lr = self.sheduler.get_last_lr()[0]
            else:
                current_lr = AlgorithmConfig.lr

            self.epoch_cnt += 1
            print(f"bc_train: epoch{self.epoch_cnt} finished, current lr: {round(current_lr, 5)}")
            # print(f"bc_train: epoch{self.epoch_cnt} finished, current lr: {current_lr}")


            log_ = {"lr": float(current_lr), "sample_size": float(sample_size)}
            log.update(log_)
            self.logs.append(log)
            # print_dict(log)
            # print(str(log))
            self.log_trivial(dictionary=log)
            
        self.log_trivial_finalize()
        self.mcv.rec(self.epoch_cnt, 'time')
        self.mcv.rec_show()
        torch.cuda.empty_cache()
                
        update_cnt = int(self.epoch_cnt/AlgorithmConfig.num_epoch_per_update)
        print(f"update {update_cnt} finished")
        # self.save_model(self.epoch_cnt)
        return update_cnt

    def establish_torch_graph(self, obs, act):
        (
            logit_wasd,
            logit_x,
            logit_y,
            logit_jump,
            logit_crouch,
            logit_reload,
            logit_r,
            logit_l 
        ) = self.policy(obs)

        dist_wasd = Categorical(logits=logit_wasd)
        dist_x = Categorical(logits=logit_x)
        dist_y = Categorical(logits=logit_y)
        dist_jump = Bernoulli(logits=logit_jump)
        dist_crouch = Bernoulli(logits=logit_crouch)
        dist_reload = Bernoulli(logits=logit_reload)
        dist_r = Bernoulli(logits=logit_r)
        dist_l = Bernoulli(logits=logit_l)


        index_x = _2tensor(act['x'])
        index_y = _2tensor(act['y'])
        index_wasd = _2tensor(act['wasd'])

        index_jump = _2tensor(act['jump'])
        index_crouch =  _2tensor(act['crouch'])
        index_reload =  _2tensor(act['reload'])
        index_r =  _2tensor(act['r'])
        index_l =  _2tensor(act['l'])


        new_coef = 0.2

        category_actLogProbs = (
            dist_wasd.log_prob(index_wasd) + dist_x.log_prob(index_x) + dist_y.log_prob(index_y)
        )/3
        binary_actLogProbs = new_coef * (
            dist_jump.log_prob(index_jump) + dist_crouch.log_prob(index_crouch) + dist_reload.log_prob(index_reload)
            + dist_r.log_prob(index_r) + dist_l.log_prob(index_l)
        )/5
        cross_entropy_loss = -(category_actLogProbs + binary_actLogProbs).mean() # mean log probility of the expert actions -> cross entropy loss -> max likelyhood
        binary_actLogProbs = t3n(binary_actLogProbs)
        category_actLogProbs = t3n(category_actLogProbs)
        

        category_distEntropy = (
            dist_wasd.entropy() + dist_x.entropy() + dist_y.entropy()
        )/3
        binary_distEntropy = new_coef * (
            dist_jump.entropy() + dist_crouch.entropy() + dist_reload.entropy()
            + dist_r.entropy() + dist_l.entropy()
        )/5
        mean_distEntropy = (category_distEntropy + binary_distEntropy).mean()
        binary_distEntropy = t3n(binary_distEntropy)
        category_distEntropy = t3n(category_distEntropy)
        dist_entropy_loss = -mean_distEntropy * AlgorithmConfig.dist_entropy_loss_coef
        mean_distEntropy = t3n(mean_distEntropy)

        

        loss = cross_entropy_loss + dist_entropy_loss
        cross_entropy_loss = t3n(cross_entropy_loss)
        dist_entropy_loss = t3n(dist_entropy_loss)



        log = {
            "Cross Entropy Loss": cross_entropy_loss,
            "Cross Entropy Loss clip 5": np.clip(cross_entropy_loss, -10, 5),
            "Dist Entropy Loss (Regulization)": dist_entropy_loss,

            "binary_actLogProbs": binary_actLogProbs,
            "category_actLogProbs": category_actLogProbs,

            "binary_distEntropy": binary_distEntropy,
            "category_distEntropy": category_distEntropy,

            "Dist Entropy": mean_distEntropy
        }


        return loss, log