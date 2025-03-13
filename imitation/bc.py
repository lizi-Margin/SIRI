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

def get_a_logger():
    from VISUALIZE.mcom import mcom, logdir
    mcv = mcom( path='%s/logger/'%logdir,
                    digit=16,
                    rapid_flush=True,
                    draw_mode='Img',
                    tag='[task_runner.py]',
                    resume_mod=False)
    mcv.rec_init(color='b')
    return mcv



class AlgorithmConfig:
    logdir = './imitation_TRAIN/BC/'
    device = 'cuda'
    
    sample_size = 50

    # behavior cloning part
    lr = 0.01 
    lr_sheduler_min_lr = 0.004
    # lr = 0.005 
    # lr_sheduler_min_lr = 0.0008
    lr_sheduler = True  # whether to use lr_sheduler
    num_epoch_per_update = 16
    beta_base = 0.
    dist_entropy_loss_coef = 1e-4
    

    mcom = get_a_logger()
    


class wasd_xy_Trainer():
    def __init__(self, policy: torch.nn.Module):
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
                    coef = min_coef + max(0, float(np.sin(epoch/100))) * (1-min_coef) * 0.35
                return coef
            self.sheduler = optim.lr_scheduler.LambdaLR(self.optimizer, lr_lambda=linear_decay_and_jump)

        self.epoch_cnt = 0
        self.logs = []
        self.mcv = AlgorithmConfig.mcom
        self.trivial_dict = {}
        self.smooth_trivial_dict = {}



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
        self.save_model(self.epoch_cnt)
        return update_cnt


    def save_model(self, update_cnt, info=None):
        if not os.path.exists('%s/history_cpt/' % AlgorithmConfig.logdir): 
            os.makedirs('%s/history_cpt/' % AlgorithmConfig.logdir)

        # dir 1
        pt_path = '%s/model.pt' % AlgorithmConfig.logdir
        print绿('saving model to %s' % pt_path)
        torch.save({
            'policy': self.policy.state_dict(),
            'optimizer': self.optimizer.state_dict(),
        }, pt_path)

        # dir 2
        if update_cnt % 20*AlgorithmConfig.num_epoch_per_update == 0 :
            info = str(update_cnt) if info is None else ''.join([str(update_cnt), '_', info])
            pt_path2 = '%s/history_cpt/model_%s.pt' % (AlgorithmConfig.logdir, info)
            shutil.copyfile(pt_path, pt_path2)

            print绿('save_model fin')
    

    def load_model(self):
        # if not os.path.exists('%s/history_cpt/' % AlgorithmConfig.logdir): 
        #     assert False, "file does not exists"

        # dir 1
        pt_path = '%s/model.pt' % AlgorithmConfig.logdir
        cpt = torch.load(pt_path, map_location=AlgorithmConfig.device)
        self.policy.load_state_dict(cpt['policy'], strict=True)
        # https://github.com/pytorch/pytorch/issues/3852
        self.optimizer.load_state_dict(cpt['optimizer'])

        print绿(f'loaded model {pt_path}')


    def log_trivial(self, dictionary):
        for key in dictionary:
            if key not in self.trivial_dict: self.trivial_dict[key] = []
            item = dictionary[key].item() if hasattr(dictionary[key], 'item') else dictionary[key]
            self.trivial_dict[key].append(item)

    def log_trivial_finalize(self, print=True):
        for key in self.trivial_dict:
            self.trivial_dict[key] = np.array(self.trivial_dict[key])
        
        print_buf = ['[bc.py] ']
        for key in self.trivial_dict:
            self.trivial_dict[key] = self.trivial_dict[key].mean()
            print_buf.append(' %s:%.3f, '%(key, self.trivial_dict[key]))
            if self.mcv is not None:  
                alpha = 0.98
                if key in self.smooth_trivial_dict:
                    self.smooth_trivial_dict[key] = alpha*self.smooth_trivial_dict[key] + (1-alpha)*self.trivial_dict[key]
                else:
                    self.smooth_trivial_dict[key] = self.trivial_dict[key]
                self.mcv.rec(self.trivial_dict[key], key)
                self.mcv.rec(self.smooth_trivial_dict[key], key + ' - smooth')
        if print: print紫(''.join(print_buf))
        if self.mcv is not None:
            self.mcv.rec_show()

        self.trivial_dict = {}
