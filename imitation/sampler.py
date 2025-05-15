import math
import torch.nn.functional as F
import numpy as np
from copy import copy
from torch.utils.data.sampler import BatchSampler, SubsetRandomSampler
from UTIL.colorful import *
from UTIL.tensor_ops import _2tensor, __hash__, repeat_at
from .utils import get_container_from_traj_pool
class TrajPoolSampler():
    return_rename = "return"
    reward_rename = "reward"
    value_rename =  "state_value"
    advantage_rename = "advantage"
    req_dict_rename = ['obs', 'action','action_index', 'actionLogProb', return_rename, reward_rename, value_rename]
    req_dict = ['obs', 'action','action_index', 'actionLogProb', 'return', 'reward', 'value']

    def __init__(self, n_div, traj_pool, flag, prevent_batchsize_oom=False, mcv=None):
        self.n_pieces_batch_division = n_div
        self.prevent_batchsize_oom = prevent_batchsize_oom    
        self.mcv = mcv
        if self.prevent_batchsize_oom:
            assert self.n_pieces_batch_division==1, ('?')

        self.num_batch = None
        self.container = {}
        self.warned = False
        assert flag=='train'

        self.container = get_container_from_traj_pool(traj_pool, self.req_dict_rename)

        # normalize advantage inside the batch
        self.container[self.advantage_rename] = self.container[self.return_rename] - self.container[self.value_rename]
        self.container[self.advantage_rename] = ( self.container[self.advantage_rename] - self.container[self.advantage_rename].mean() ) / (self.container[self.advantage_rename].std() + 1e-5)

        self.big_batch_size = self._get_big_batch_size()

        # size of minibatch for each agent
        self.mini_batch_size = math.ceil(self.big_batch_size / self.n_pieces_batch_division)  

    def __len__(self):
        return self.n_pieces_batch_division
    
    def _get_big_batch_size(self):
        # check batch size
        big_batch_size = self.container[self.req_dict_rename[1]].shape[0]
        for key_index, key in enumerate(self.req_dict_rename):
            if key in self.container:
                if not isinstance(self.container[key], tuple):
                    assert big_batch_size==self.container[key].shape[0], (key,key_index)
        return big_batch_size

    def determine_max_n_sample(self):
        assert self.prevent_batchsize_oom
        if not hasattr(TrajPoolSampler,'MaxSampleNum'):
            # initialization
            TrajPoolSampler.MaxSampleNum =  [int(self.big_batch_size*(i+1)/50) for i in range(50)]
            max_n_sample = self.big_batch_size
        elif TrajPoolSampler.MaxSampleNum[-1] > 0:  
            # meaning that oom never happen, at least not yet
            # only update when the batch size increases
            if self.big_batch_size > TrajPoolSampler.MaxSampleNum[-1]: TrajPoolSampler.MaxSampleNum.append(self.big_batch_size)
            max_n_sample = self.big_batch_size
        else:
            # meaning that oom already happened, choose TrajPoolSampler.MaxSampleNum[-2] to be the limit
            assert TrajPoolSampler.MaxSampleNum[-2] > 0
            max_n_sample = TrajPoolSampler.MaxSampleNum[-2]
        return max_n_sample

    def reset_and_get_iter(self):
        if not self.prevent_batchsize_oom:
            self.sampler = BatchSampler(SubsetRandomSampler(range(self.big_batch_size)), self.mini_batch_size, drop_last=False)
        else:
            max_n_sample = self.determine_max_n_sample()
            n_sample = min(self.big_batch_size, max_n_sample)
            if not hasattr(self,'reminded'):
                self.reminded = True
                drop_percent = (self.big_batch_size-n_sample)/self.big_batch_size*100
                if self.mcv is not None:
                    self.mcv.rec(drop_percent, 'drop percent')
                if drop_percent > 20: 
                    print_ = print亮红
                    print_('droping %.1f percent samples..'%(drop_percent))
                    assert False, "GPU OOM!"
                else:
                    print_ = print
                    print_('droping %.1f percent samples..'%(drop_percent))
            self.sampler = BatchSampler(SubsetRandomSampler(range(n_sample)), n_sample, drop_last=False)

        for indices in self.sampler:
            selected = {}
            for key in self.container:
                selected[key] = self.container[key][indices]
            for key in [key for key in selected if '>' in key]:
                # 重新把子母键值组合成二重字典
                mainkey, subkey = key.split('>')
                if not mainkey in selected: selected[mainkey] = {}
                selected[mainkey][subkey] = selected[key]
                del selected[key]
            yield selected


class AdversarialTrajPoolSampler(TrajPoolSampler):
    label_dict_rename = 'label'

    def __init__(self, n_div, policy_traj_pool, expert_container, flag, prevent_batchsize_oom=False, mcv=None):
        self.n_pieces_batch_division = n_div
        self.prevent_batchsize_oom = prevent_batchsize_oom    
        self.mcv = mcv
        if self.prevent_batchsize_oom:
            assert self.n_pieces_batch_division==1, ('?')

        self.num_batch = None
        self.warned = False
        assert flag=='train'
        
        # adv_req_dict = copy(self.req_dict)
        # adv_req_dict.remove('return')
        # adv_req_dict.remove('avail_act')
        # adv_req_dict_rename = copy(TrajPoolSampler.req_dict_rename)
        # adv_req_dict_rename.remove(self.return_rename)
        # adv_req_dict_rename.remove('avail_act')

        req_dict = ['obs', 'action_for_reward', 'actionLogProb']
        req_dict_rename = req_dict
        for name in req_dict:
            assert name in expert_container, (name, expert_container.keys())

        self.expert_container = expert_container
        self.policy_container = get_container_from_traj_pool(policy_traj_pool, req_dict_rename, req_dict=req_dict)

        _expert_shape = self.expert_container[req_dict_rename[0]].shape
        _policy_shape = self.policy_container[req_dict_rename[0]].shape
        self.expert_container[self.label_dict_rename] = np.ones((_expert_shape[0], _expert_shape[1], 1,))
        self.policy_container[self.label_dict_rename] = np.zeros((_policy_shape[0], _policy_shape[1], 1,))
        req_dict_rename.append(self.label_dict_rename)

        # concatenate expert data and policy(generator) data
        self.container = {}
        for key in req_dict_rename:
            self.container[key] = np.concatenate([self.expert_container[key], self.policy_container[key]])
        
        self.big_batch_size = self._get_big_batch_size()
        self.mini_batch_size = math.ceil(self.big_batch_size / self.n_pieces_batch_division)  
        


class ContainerSampler(TrajPoolSampler):
    def __init__(self, n_div, container, flag, prevent_batchsize_oom=False, mcv=None):
        self.n_pieces_batch_division = n_div
        self.prevent_batchsize_oom = prevent_batchsize_oom    
        self.mcv = mcv
        if self.prevent_batchsize_oom:
            assert self.n_pieces_batch_division==1, ('?')

        self.num_batch = None
        self.warned = False
        assert flag=='train'

        assert isinstance(container, dict)
        self.container = container

        # normalize advantage inside the batch
        self.container[self.advantage_rename] = self.container[self.return_rename] - self.container[self.value_rename]
        self.container[self.advantage_rename] = ( self.container[self.advantage_rename] - self.container[self.advantage_rename].mean() ) / (self.container[self.advantage_rename].std() + 1e-5)

        self.big_batch_size = self._get_big_batch_size()

        # size of minibatch for each agent
        self.mini_batch_size = math.ceil(self.big_batch_size / self.n_pieces_batch_division)  