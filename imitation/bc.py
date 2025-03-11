import os
import torch
import shutil
import numpy as np
import torch.optim as optim
from torch.distributions.categorical import Categorical
from UTIL.tensor_ops import _2tensor
from UTIL.colorful import *
from imitation.utils import print_dict

class AlgorithmConfig:
    logdir = './imitation_TRAIN/BC/'
    device = 'cuda'
    
    sample_size = 128

    # behavior cloning part
    lr = 0.02 
    num_epoch_per_update = 20
    drop_data = True
    MAX_MEM_TRAJ_NUM = 128
    beta_base = 0.
    dist_entropy_loss_coef = 1e-4
    lr_sheduler = True  # whether to use lr_sheduler
    lr_sheduler_min_lr = 0.005
    


class wasd_xy_Trainer():
    def __init__(self, policy: torch.nn.Module):
        self.policy = policy
        self.policy.to(AlgorithmConfig.device)
        self.lr = AlgorithmConfig.lr
        self.all_parameter = list(policy.named_parameters())

        # if not self.freeze_body:
        self.parameter = [p for p_name, p in self.all_parameter]
        self.optimizer = optim.Adam(self.parameter, lr=self.lr)

        self.sheduler = None
        if AlgorithmConfig.lr_sheduler:
            def linear_decay(epoch):
                coef = max(1. - epoch/1000, 0.)
                return max(coef, AlgorithmConfig.lr_sheduler_min_lr/AlgorithmConfig.lr)
            self.sheduler = optim.lr_scheduler.LambdaLR(self.optimizer, lr_lambda=linear_decay)

        self.epoch_cnt = 0
        self.logs = []



    def train_on_data_(self, data: dict):
        """ BC """
        num_epoch = AlgorithmConfig.num_epoch_per_update


        for epoch in range(num_epoch):
            N = len(next(iter(data.values())))
            n = min(AlgorithmConfig.sample_size, N)
            start = np.random.choice(max(N-n, 1))
            sample = {key: value[start:start+n] for key, value in data.items()}

            self.optimizer.zero_grad()

            frame_center = _2tensor(sample['frame_center'])
            index_x = _2tensor(sample['x'])
            index_y = _2tensor(sample['y'])
            index_wasd = _2tensor(sample['wasd'])

            logit_wasd, logit_x, logit_y = self.policy(frame_center)
            dist_wasd = Categorical(logits=logit_wasd)
            dist_x = Categorical(logits=logit_x)
            dist_y = Categorical(logits=logit_y)

            actLogProbs = dist_wasd.log_prob(index_wasd) + dist_x.log_prob(index_x) + dist_y.log_prob(index_y)
            distEntropy = dist_wasd.entropy() + dist_x.entropy() + dist_y.entropy()

            cross_entropy_loss = -actLogProbs.mean() # mean log probility of the expert actions -> cross entropy loss -> max likelyhood
            dist_entropy_loss = -distEntropy.mean() * AlgorithmConfig.dist_entropy_loss_coef

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
                "Cross Entropy Loss": round(float(cross_entropy_loss.detach().to("cpu").numpy()), 4),
                "Dist Entropy Loss": round(float(dist_entropy_loss.detach().to("cpu").numpy()), 4),
            }
            self.logs.append(log)
            # print_dict(log)
            print(str(log))
        torch.cuda.empty_cache()
                
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
        info = str(update_cnt) if info is None else ''.join([str(update_cnt), '_', info])
        pt_path2 = '%s/history_cpt/model_%s.pt' % (AlgorithmConfig.logdir, info)
        shutil.copyfile(pt_path, pt_path2)

        print绿('save_model fin')
    

    def load_model(self):
        if not os.path.exists('%s/history_cpt/' % AlgorithmConfig.logdir): 
            assert False, "file does not exists"

        # dir 1
        pt_path = '%s/model.pt' % AlgorithmConfig.logdir
        cpt = torch.load(pt_path, map_location=AlgorithmConfig.device)
        self.policy.load_state_dict(cpt['policy'], strict=True)
        # https://github.com/pytorch/pytorch/issues/3852
        self.optimizer.load_state_dict(cpt['optimizer'])

        print绿(f'loaded model {pt_path}')