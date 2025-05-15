import torch, traceback
# import math, traceback
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import numpy as np
import random
# import tqdm
# from random import randint, sample
# from torch.utils.data.sampler import BatchSampler, SubsetRandomSampler
from UTIL.colorful import *
from .conf import AlgorithmConfig
from UTIL.tensor_ops import _2tensor, __hash__, __hashn__
from imitation.sampler import TrajPoolSampler, AdversarialTrajPoolSampler, ContainerSampler
# from VISUALIZE.mcom import mcom
from imitation.utils import get_container_from_traj_pool, safe_load_traj_pool
from data_loader import get_data

class AIRL_PPO():
    def __init__(self, policy_and_critic, reward_net):
        self.policy_and_critic = policy_and_critic
        self.reward_net = reward_net
        self.clip_param = AlgorithmConfig.clip_param
        self.ppo_epoch = AlgorithmConfig.ppo_epoch
        self.use_avail_act = AlgorithmConfig.ppo_epoch
        self.n_pieces_batch_division = AlgorithmConfig.n_pieces_batch_division
        self.value_loss_coef = AlgorithmConfig.value_loss_coef
        self.entropy_coef = AlgorithmConfig.entropy_coef
        self.max_grad_norm = AlgorithmConfig.max_grad_norm
        self.add_prob_loss = AlgorithmConfig.add_prob_loss
        self.prevent_batchsize_oom = AlgorithmConfig.prevent_batchsize_oom
        # self.freeze_body = ppo_config.freeze_body
        self.ppo_lr = AlgorithmConfig.ppo_lr
        self.disc_lr = AlgorithmConfig.disc_lr
        policy_all_parameter = list(policy_and_critic.named_parameters())

        # if not self.freeze_body:
        self.policy_parameter = [p for p_name, p in policy_all_parameter]
        self.policy_optimizer = optim.Adam(self.policy_parameter, lr=self.ppo_lr)
        self.reward_optimizer = optim.Adam(self.reward_net.parameters(), lr=self.disc_lr)

        self.g_update_delayer = 0
        self.g_initial_value_loss = 0
        
        # 轮流训练式
        self.mcv = AlgorithmConfig.mcom
        self.ppo_update_cnt = 0
        self.batch_size_reminder = True
        self.trivial_dict = {}

        assert self.n_pieces_batch_division == 1
        self.train_hook = self.TrainHook(self.prevent_batchsize_oom)

        self.safe_load_traj_pool = safe_load_traj_pool(traj_dir=AlgorithmConfig.datasets)

        # self.train_on_data(None, "train", n_epoch=10)

    def get_train_hook(self, name):
        if   name == "train": name = "train_on_data_"
        elif name == "test":  name = "train_on_traj_"

        assert hasattr(self, name), 'no such train hook'
        self.train_hook.hook_func = getattr(self, name)
        return self.train_hook

    class TrainHook:
        def __init__(self, prevent_batchsize_oom):
            self.hook_func = None
            self.prevent_batchsize_oom = prevent_batchsize_oom

        def __call__(self, traj_pool, task):
            assert self.hook_func is not None
            while True:
                try:
                    self.hook_func(traj_pool, task) 
                    break # 运行到这说明显存充足
                except RuntimeError as err:
                    print(traceback.format_exc())
                    if self.prevent_batchsize_oom:
                        # in some cases, reversing MaxSampleNum a single time is not enough
                        if TrajPoolSampler.MaxSampleNum[-1] < 0: TrajPoolSampler.MaxSampleNum.pop(-1)
                        assert TrajPoolSampler.MaxSampleNum[-1] > 0
                        TrajPoolSampler.MaxSampleNum[-1] = -1
                        print亮红('Insufficient gpu memory, using previous sample size !')
                    else:
                        assert False
                torch.cuda.empty_cache() 

    def train_on_data_(self, policy_traj_pool, task):
        """ train AIRL """
        assert task == "train"
        assert False, "failed"

        # train reward_net(disciminator)
        
        # N_EXPERT_TRAJ = AlgorithmConfig.train_traj_needed
        # expert_traj_pool = load_all_traj_pool(idx=[_ for _ in range(25)])
        # assert expert_traj_pool is not None
        # expert_traj_pool = random.sample(expert_traj_pool, k=min(N_EXPERT_TRAJ, len(expert_traj_pool)))
        expert_traj_pool = self.safe_load_traj_pool(n_samples=AlgorithmConfig.train_traj_needed)
        expert_data = get_data(expert_traj_pool, self.policy_and_critic)
        expert_data['obs'] = self.policy_and_critic.preprocess(expert_data['obs'])

        req_dict = ['obs', 'action_for_reward', 'actionLogProb']
        req_dict_rename = req_dict
        for name in req_dict:
            assert name in expert_data, (name, expert_data.keys())
        policy_data = get_container_from_traj_pool(policy_traj_pool, req_dict_rename, req_dict=req_dict)
        policy_data['obs'] = self.policy_and_critic.get_frame_centers(policy_data['obs'])
        _expert_shape = expert_data[req_dict_rename[0]].shape
        _policy_shape = policy_data[req_dict_rename[0]].shape
        expert_data['label'] = np.ones((_expert_shape[0], _expert_shape[1], 1,))
        policy_data['label'] = np.zeros((_policy_shape[0], _policy_shape[1], 1,))

        container = {}
        for key in req_dict_rename:
            container[key] = np.concatenate([expert_data[key], policy_data[key]])


        print(f"policy_traj_pool size: {len(policy_traj_pool)}, expert_traj_pool size: {len(expert_traj_pool)}")


        for epoch in range(AlgorithmConfig.num_epoch_per_update):
            print(f"AIRL reward_net training, epoch{epoch}")

            sample = container
            
            self.reward_optimizer.zero_grad()

            obs = _2tensor(sample['obs'])
            # next_obs = torch.roll(obs, shifts=-1, dims=0); next_obs[-1] = torch.nan
            action_for_reward = _2tensor(sample['action_for_reward'].astype(np.float32))
            actionLogProb = _2tensor(sample['actionLogProb']); assert not np.any(np.isnan(sample['actionLogProb'])), 'The actionLogProb in expert data must not be Nan!'
            label = _2tensor(sample['label'].astype(np.float32)) # expert data is 1

            reward_net_logits = self.reward_net(
                obs, action_for_reward, use_hs=False
            )

            disc_logits = reward_net_logits - actionLogProb

            loss = F.binary_cross_entropy_with_logits(disc_logits, label)
            if epoch==0: print('[airl.py] Memory Allocated %.2f GB'%(torch.cuda.memory_allocated()/1073741824))
            loss.backward()
            self.reward_optimizer.step()

            log = {
                "Cross Entropy Loss": loss.detach().to("cpu").numpy(),
            }
            print(log)
            self.log_trivial(dictionary=log)
        # self.log_trivial_finalize()

        # predict the reward
        
        ppo_sampler = ContainerSampler(
            n_div=1,
            container=policy_data,
            flag=task,
            prevent_batchsize_oom=self.prevent_batchsize_oom,
            mcv=self.mcv
        )
        torch.cuda.empty_cache()

        self._ppo_train(ppo_sampler=ppo_sampler, task=task)

        self.log_trivial_finalize()
        self.ppo_update_cnt += 1
        torch.cuda.empty_cache()

        return self.ppo_update_cnt
        
    def train_on_traj_(self, traj_pool, task):
        """ test AIRL """
        assert False, "failed"
        policy_container = get_container_from_traj_pool(traj_pool, TrajPoolSampler.req_dict_rename)
        policy_container['obs'] = self.policy_and_critic.get_frame_centers(policy_container['obs'])
        policy_container = ContainerSampler(
            n_div=1,
            container=policy_container,
            flag=task,
            prevent_batchsize_oom=self.prevent_batchsize_oom,
            mcv=self.mcv
        ).container
        

        torch.cuda.empty_cache()
        
        self._ppo_train(task=task, sample=policy_container)

        self.log_trivial_finalize()
        self.ppo_update_cnt += 1
        torch.cuda.empty_cache()
                
        return self.ppo_update_cnt

    def _ppo_train(self, task, ppo_sampler=None, sample=None):
        # train policy
        ppo_valid_percent_list = []
        for epoch in range(self.ppo_epoch):
            print(f"PPO policy training, epoch{epoch}")

            if sample is None:
                sample_iter = ppo_sampler.reset_and_get_iter()
                self.policy_optimizer.zero_grad()
                # ! get traj fragment
                sample = next(sample_iter)
            # ! build graph, then update network
            loss_final, others = self.establish_pytorch_graph(task, sample, epoch)
            loss_final = loss_final*0.5
            if epoch==0: print('[airl.py] Memory Allocated %.2f GB'%(torch.cuda.memory_allocated()/1073741824))
            loss_final.backward()
            # log
            ppo_valid_percent_list.append(others.pop('PPO valid percent').item())
            self.log_trivial(dictionary=others); others = None
            nn.utils.clip_grad_norm_(self.policy_parameter, self.max_grad_norm)
            self.policy_optimizer.step()
            
            if ppo_valid_percent_list[-1] < 0.70: 
                print亮黄('policy change too much, epoch terminate early'); break 
        pass # finish all epoch update
        print亮黄(np.array(ppo_valid_percent_list))

    def freeze_body(self):
        assert False, "function forbidden"
        self.freeze_body = True
        self.parameter_pv = [p_name for p_name, p in self.all_parameter if not any(p_name.startswith(kw)  for kw in ('obs_encoder', 'attention_layer'))]
        self.parameter = [p for p_name, p in self.all_parameter if not any(p_name.startswith(kw)  for kw in ('obs_encoder', 'attention_layer'))]
        self.optimizer = optim.Adam(self.parameter, lr=self.lr)
        print('change train object')

    def log_trivial(self, dictionary):
        for key in dictionary:
            if key not in self.trivial_dict: self.trivial_dict[key] = []
            item = dictionary[key].item() if hasattr(dictionary[key], 'item') else dictionary[key]
            self.trivial_dict[key].append(item)

    def log_trivial_finalize(self, print=True):
        for key in self.trivial_dict:
            self.trivial_dict[key] = np.array(self.trivial_dict[key])
        
        print_buf = ['[airl.py] ']
        for key in self.trivial_dict:
            self.trivial_dict[key] = self.trivial_dict[key].mean()
            print_buf.append(' %s:%.3f, '%(key, self.trivial_dict[key]))
            if self.mcv is not None:  self.mcv.rec(self.trivial_dict[key], key)
        if print: print紫(''.join(print_buf))
        if self.mcv is not None:
            self.mcv.rec_show()
        self.trivial_dict = {}


    def establish_pytorch_graph(self, flag, sample, n):
        obs = _2tensor(sample['obs'])
        advantage = _2tensor(sample['advantage'])
        action = _2tensor(sample['action_index'])
        oldPi_actionLogProb = _2tensor(sample['actionLogProb'])
        real_value = _2tensor(sample['return'])
        # avail_act = _2tensor(sample['avail_act']) if 'avail_act' in sample else None

        # batchsize = advantage.shape[0]#; print亮紫(batchsize)
        batch_agent_size = advantage.shape[0]*advantage.shape[1]

        assert flag == 'train'
        newPi_value, newPi_actionLogProb, entropy, probs, others = \
            self.policy_and_critic.evaluate_actions(
                obs=obs, 
                eval_actions=action, 
                # test_mode=False, 
                # avail_act=avail_act
            )
        entropy_loss = entropy.mean()


        n_actions = probs.shape[-1]
        if self.add_prob_loss: assert n_actions <= 15  # 
        penalty_prob_line = (1/n_actions)*0.12
        probs_loss = (penalty_prob_line - torch.clamp(probs, min=0, max=penalty_prob_line)).mean()
        if not self.add_prob_loss:
            probs_loss = torch.zeros_like(probs_loss)

        # dual clip ppo core
        E = newPi_actionLogProb - oldPi_actionLogProb
        E_clip = torch.zeros_like(E)
        E_clip = torch.where(advantage > 0, torch.clamp(E, max=np.log(1.0+self.clip_param)), E_clip)
        E_clip = torch.where(advantage < 0, torch.clamp(E, min=np.log(1.0-self.clip_param), max=np.log(5) ), E_clip)
        ratio  = torch.exp(E_clip)
        policy_loss = -(ratio*advantage).mean()

        # add all loses
        value_loss = 0.5 * F.mse_loss(real_value, newPi_value)


        AT_net_loss = policy_loss - entropy_loss*self.entropy_coef # + probs_loss*20
        CT_net_loss = value_loss * 1.0
        # AE_new_loss = ae_loss * 1.0

        loss_final =  AT_net_loss + CT_net_loss  # + AE_new_loss

        ppo_valid_percent = ((E_clip == E).int().sum()/batch_agent_size)

        nz_mask = real_value!=0
        value_loss_abs = (real_value[nz_mask] - newPi_value[nz_mask]).abs().mean()
        others = {
            'Value loss Abs':           value_loss_abs,
            'PPO valid percent':        ppo_valid_percent,
            'CT_net_loss':              CT_net_loss,
            'AT_net_loss':              AT_net_loss,
        }

        return loss_final, others
