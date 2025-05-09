import os
import torch
import shutil
import numpy as np
import torch.optim as optim
from UTIL.colorful import *
from .conf import AlgorithmConfig

class TrainerBase():
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
                    coef = min_coef + max(0, float(np.sin(epoch/500))) * (1-min_coef) * 0.25
                return coef
            self.sheduler = optim.lr_scheduler.LambdaLR(self.optimizer, lr_lambda=linear_decay_and_jump)

        self.epoch_cnt = 0
        self.logs = []
        self.mcv = AlgorithmConfig.mcom
        self.trivial_dict = {}
        self.smooth_trivial_dict = {}


    def save_model(self,info=None):
        # update_cnt = int(self.epoch_cnt/AlgorithmConfig.num_epoch_per_update)
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
        info = str(self.epoch_cnt) if info is None else ''.join([str(self.epoch_cnt), '_', info])
        pt_path2 = '%s/history_cpt/model_%s.pt' % (AlgorithmConfig.logdir, info)
        shutil.copyfile(pt_path, pt_path2)

        # dir 3
        pt_path3 = '%s/model-%s-AUTOSAVED.pt' % (AlgorithmConfig.logdir, self.policy.__class__.__name__)
        shutil.copyfile(pt_path, pt_path3)

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
