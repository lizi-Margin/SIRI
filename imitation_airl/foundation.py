import os, time, torch, traceback, shutil
import random
import numpy as np
from UTIL.colorful import *
from .conf import AlgorithmConfig
from UTIL.tensor_ops import repeat_at
    
def str_array_to_num(str_arr):
    out_arr = []
    buffer = {}
    for str in str_arr:
        if str not in buffer:
            buffer[str] = len(buffer)
        out_arr.append(buffer[str])
    return out_arr

def itemgetter(*items):
    # same with operator.itemgetter
    def g(obj): return tuple(obj[item] if item in obj else None for item in items)
    return g

class ReinforceAlgorithmFoundation:
    def __init__(self):
        self.device = AlgorithmConfig.device
        self.logdir = AlgorithmConfig.logdir

        # initialize policy
        from .AC import DoubleBranchMapAC as AC
        self.policy = AC()
        self.policy = self.policy.to(self.device)

        # initialize reward
        from .reward_net import DoubleBranchMapReward as RewardNet
        self.reward_net = RewardNet(action_dim=6)
        # self.reward_net = ShapedRewardNet(rawob_dim=rawob_dim, action_dim=rawact_dim)
        self.reward_net.to(self.device)

        # initialize optimizer and trajectory (batch) manager
        from .airl import AIRL_PPO
        from .trajectory import BatchTrajManager
        self.trainer = AIRL_PPO(
            policy_and_critic=self.policy,
            reward_net=self.reward_net,
        )
        trainer_hook = self.trainer.get_train_hook("test" if AlgorithmConfig.test_reward_net else "train")
        self.traj_manager = BatchTrajManager(
            n_env=1, traj_limit=int(200),
            trainer_hook=trainer_hook
        )

    @staticmethod
    def hmap_multi_env_compat(traj_frag):
        assert isinstance(traj_frag, dict)
        for key in traj_frag:
            if key in ("_DONE_", "_SKIP_", "_TOBS_"):
                traj_frag[key] = np.array([traj_frag[key]])
            else:
                if isinstance(traj_frag[key], torch.Tensor):
                    traj_frag[key] = traj_frag[key].detach().cpu().numpy()
                
                if isinstance(traj_frag[key], tuple):
                    assert False
                else:
                    traj_frag[key] = [traj_frag[key]]
        return traj_frag


    def interact_with_env(self, State):
        '''
            Interfacing with marl, standard method that you must implement
            (redirect to shell_env to help with history rolling)
        '''
        self.policy.eval()

        # read obs
        frames = State['obs']
        obs = self.policy.get_frame_centers(frames)
        done = State['done']

        # make decision
        wasd, xy, info = self.policy.act(frames=frames, frame_centers=obs)
        index_x = info['index_x']; assert isinstance(index_x, int), f"type: {type(index_x)}"
        index_y = info['index_y']; assert isinstance(index_y, int), f"type: {type(index_y)}"
        index_wasd = info['index_wasd']; assert isinstance(index_wasd, int), f"type: {type(index_wasd)}"
        # from .map_net_base import x_box, y_box
        index_xy = np.array([index_x, index_y])
        index_xy_norm = np.array([index_x/12, index_y/4])
        action = (wasd, xy,)
        action_raw = np.concatenate([wasd, xy], axis=0)
        action_for_reward = np.concatenate([wasd, index_xy_norm], axis=0)
        action_index = np.concatenate([np.array([index_wasd]), index_xy], axis=0)
        
        actLogProbs = info['actLogProbs']
        value = info['value']

        fake_reward = np.array([0.0])
        # generated_reward = self.reward_net.get_reward(
        #     frames=frames,
        #     frame_centers=obs,
        #     actions=action_for_reward
        # )
        if done:
            traj_frag = {
                '_DONE_': True,      # 这表示episode结束
                '_SKIP_': False,     # 不跳过这个数据
                '_TOBS_': frames,  # 终止时的观察值
                'obs': frames,
                'action': action_index,
                'action_for_reward': action_for_reward,
                'action_index': action_index,
                'action_raw': action_raw,
                'actionLogProb': actLogProbs,
                'value': value,
                'reward': fake_reward,
                'human_reward': fake_reward,
                # 'generated_reward': generated_reward,
            }
        else:
            traj_frag = {
                '_DONE_': False, 
                '_SKIP_': False, 
                '_TOBS_': None, 
                'obs': frames,
                'action': action_index,
                'action_for_reward': action_for_reward,
                'action_index': action_index,
                'action_raw': action_raw,
                'actionLogProb': actLogProbs,
                'value': value,
                'reward': fake_reward,
                'human_reward': fake_reward,
                # 'generated_reward': generated_reward,
            }
        traj_frag = self.hmap_multi_env_compat(traj_frag)
        self.traj_manager.feed_traj_framedata(traj_frag, require_hook=False)
        if done:
            print绿('[ReinforceAlgorithmFoundation] episode done, all nets reset')
            self.policy.reset()
            self.reward_net.reset()
        return action


    def train(self):
        if self.traj_manager.can_exec_training():
            print绿(f'start training, traj_pool len: {len(self.traj_manager.traj_pool)}')
            self.traj_manager.train_and_clear_traj_pool()
            self.save_model()
        else:
            # print黄(f'traj_manager is not ready to train, traj_pool len: {len(self.traj_manager.traj_pool)}')
            pass


    def save_model(self, info=None):
        update_cnt = self.traj_manager.update_cnt
        if not os.path.exists('%s/history_cpt/' % self.logdir): 
            os.makedirs('%s/history_cpt/' % self.logdir)

        # dir 1
        pt_path = '%s/model.pt' % self.logdir
        print绿('saving model to %s' % pt_path)
        torch.save({
            'policy': self.policy.state_dict(),
            'optimizer': self.trainer.policy_optimizer.state_dict(),
            'reward_net': self.trainer.reward_net.state_dict(),
            'reward_optimizer': self.trainer.reward_optimizer.state_dict()
        }, pt_path)

        # dir 2
        info = str(update_cnt) if info is None else ''.join([str(update_cnt), '_', info])
        pt_path2 = '%s/history_cpt/model_%s.pt' % (self.logdir, info)
        shutil.copyfile(pt_path, pt_path2)

        print绿('save_model fin')

    def load_model(self, policy_only=False):
        '''
            load model now
        '''
        ckpt_dir = '%s/model.pt' % self.logdir
        cuda_n = 'cpu' if 'cpu' in self.device else self.device
        strict = True
        cpt = torch.load(ckpt_dir, map_location=cuda_n)

        if policy_only:
            print黄('loading policy only...')
            self.policy.load_state_dict(cpt['policy'], strict=strict)
        else:
            self.policy.load_state_dict(cpt['policy'], strict=strict)
            self.reward_net.load_state_dict(cpt['reward_net'], strict=strict)
            # https://github.com/pytorch/pytorch/issues/3852
            self.trainer.policy_optimizer.load_state_dict(cpt['optimizer'])
            self.trainer.reward_optimizer.load_state_dict(cpt['reward_optimizer'])
        print黄('loaded checkpoint:', ckpt_dir)
