import os
import torch
import shutil
import numpy as np
from UTIL.colorful import *

class TrainerBase():
    def __init__(self, logdir, device):
        self.logdir = logdir
        self.device = device
        self.trivial_dict = {}
        self.smooth_trivial_dict = {}


    def save_model(self,info=None):
        # update_cnt = int(self.epoch_cnt/self.num_epoch_per_update)
        if not os.path.exists('%s/history_cpt/' % self.logdir): 
            os.makedirs('%s/history_cpt/' % self.logdir)

        # dir 1
        pt_path = '%s/model.pt' % self.logdir
        print绿('saving model to %s' % pt_path)
        torch.save({
            'policy': self.policy.state_dict(),
            'optimizer': self.optimizer.state_dict(),
        }, pt_path)

        # dir 2
        info = str(self.epoch_cnt) if info is None else ''.join([str(self.epoch_cnt), '_', info])
        pt_path2 = '%s/history_cpt/model_%s.pt' % (self.logdir, info)
        shutil.copyfile(pt_path, pt_path2)

        # dir 3
        pt_path3 = '%s/model-%s-AUTOSAVED.pt' % (self.logdir, self.policy.__class__.__name__)
        shutil.copyfile(pt_path, pt_path3)

        print绿('save_model fin')
    

    def load_model(self):
        # if not os.path.exists('%s/history_cpt/' % self.logdir): 
        #     assert False, "file does not exists"

        # dir 1
        pt_path = '%s/model.pt' % self.logdir
        cpt = torch.load(pt_path, map_location=self.device)
        if 'optimizer' not in cpt:
            self.policy.load_state_dict(cpt, strict=True)
        else:
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
