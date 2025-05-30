import torch
import numpy as np
from torch.distributions.categorical import Categorical
from torch.distributions.bernoulli import Bernoulli
import torch.optim as optim
from UTIL.tensor_ops import _2tensor
from UTIL.colorful import *
from imitation.utils import print_dict
from siri.utils.logger import lprint_

from .conf import AlgorithmConfig
from imitation.trainer_base import TrainerBase

def t2n_mean(x):
    return float(x.detach().mean().to("cpu").numpy())


class FullTrainer(TrainerBase):
    def __init__(self, policy: torch.nn.Module):
        super().__init__(AlgorithmConfig.logdir, AlgorithmConfig.device)
        self.policy = policy
        self.policy.to(AlgorithmConfig.device)
        self.lr = AlgorithmConfig.lr
        self.all_parameter = list(policy.named_parameters())

        # if not self.freeze_body:
        self.parameter = [p for p_name, p in self.all_parameter]
        # self.optimizer = optim.Adam(self.parameter, lr=self.lr)
        self.optimizer = optim.SGD(self.parameter, lr=self.lr)

        self.sheduler = None
        if AlgorithmConfig.lr_sheduler:
            # def linear_decay(epoch):
            #     coef = max(1. - epoch/1000, 0.)
            #     return max(coef, AlgorithmConfig.lr_sheduler_min_lr/AlgorithmConfig.lr)
            def linear_decay_and_jump(epoch):
                coef = max(1. - epoch/5000, 0.)
                min_coef = AlgorithmConfig.lr_sheduler_min_lr/AlgorithmConfig.lr
                if coef <= min_coef:
                    coef = min_coef + max(0, float(np.sin(epoch/500))) * (1-min_coef) * 0.25
                return coef
            self.sheduler = optim.lr_scheduler.LambdaLR(self.optimizer, lr_lambda=linear_decay_and_jump)

        self.epoch_cnt = 0
        self.logs = []
        self.mcv = AlgorithmConfig.mcom
        self.scaler = torch.GradScaler('cuda', init_scale = 2.0**16)

    def train_on_data_(self, data: dict):
        """ BC """
        num_epoch = AlgorithmConfig.num_epoch_per_update
        assert 'obs' in data
        all_obs = data.pop('obs')
        assert 'obs' not in data
        sample_size = int(np.random.uniform(low=AlgorithmConfig.sample_size_min, high=AlgorithmConfig.sample_size_max))
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
                    with torch.autocast("cuda", dtype=torch.float16):
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
                    if sample_size < AlgorithmConfig.sample_size_min/2:
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



        category_actLogProbs = (
            dist_wasd.log_prob(index_wasd) + dist_x.log_prob(index_x) + dist_y.log_prob(index_y)
        )/3
        binary_actLogProbs = AlgorithmConfig.binary_coef * (
            dist_jump.log_prob(index_jump) + dist_crouch.log_prob(index_crouch) + dist_reload.log_prob(index_reload)
            + dist_r.log_prob(index_r) + dist_l.log_prob(index_l)
        )/5
        cross_entropy_loss = -(category_actLogProbs + binary_actLogProbs).mean() # mean log probility of the expert actions -> cross entropy loss -> max likelyhood
        binary_actLogProbs = t2n_mean(binary_actLogProbs)
        category_actLogProbs = t2n_mean(category_actLogProbs)
        

        category_distEntropy = (
            dist_wasd.entropy() + dist_x.entropy() + dist_y.entropy()
        )/3
        binary_distEntropy = AlgorithmConfig.binary_coef * (
            dist_jump.entropy() + dist_crouch.entropy() + dist_reload.entropy()
            + dist_r.entropy() + dist_l.entropy()
        )/5
        mean_distEntropy = (category_distEntropy + binary_distEntropy).mean()
        binary_distEntropy = t2n_mean(binary_distEntropy)
        category_distEntropy = t2n_mean(category_distEntropy)
        dist_entropy_loss = -mean_distEntropy * AlgorithmConfig.dist_entropy_loss_coef
        mean_distEntropy = t2n_mean(mean_distEntropy)

        

        loss = cross_entropy_loss + dist_entropy_loss
        cross_entropy_loss = t2n_mean(cross_entropy_loss)
        dist_entropy_loss = t2n_mean(dist_entropy_loss)



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