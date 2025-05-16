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

        # initialize optimizer and trajectory (batch) manager
        from .bc import Trainer
        from .trajectory import BatchTrajManager
        self.trainer = Trainer(
            policy=self.policy,
        )
        trainer_hook = self.trainer.train_on_data
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
        human_active = State['human_active']

        # read obs
        frames = State['obs']
        obs = self.policy.get_frame_centers(frames)
        done = State['done']

        if not human_active:
            # make decision
            wasd, xy, info = self.policy.act(frames=frames, frame_centers=obs)
            action = (wasd, xy,)
            # index_x = info['index_x']; assert isinstance(index_x, int), f"type: {type(index_x)}"
            # index_y = info['index_y']; assert isinstance(index_y, int), f"type: {type(index_y)}"
            # index_wasd = info['index_wasd']; assert isinstance(index_wasd, int), f"type: {type(index_wasd)}"
        else:
            self.policy.reset()
            wasd = State['rec']['act_wasd']
            xy = State['rec']['xy']
            action = (wasd, xy,)

        index_x = self.policy.x_discretizer.discretize(State['rec']['xy'][0])
        index_y = self.policy.y_discretizer.discretize(State['rec']['xy'][1])
        index_wasd = self.policy.wasd_discretizer.action_to_index(State['rec']['act_wasd'])
        

        index_xy = np.array([index_x, index_y])
        # action_raw = np.concatenate([wasd, xy], axis=0)
        action_index = np.concatenate([np.array([index_wasd]), index_xy], axis=0)
        

        if done:
            traj_frag = {
                '_DONE_': True,      # 这表示episode结束
                '_SKIP_': False,     # 不跳过这个数据
                '_TOBS_': frames,  # 终止时的观察值
                'obs': frames,
                'action': action_index,
                'action_index': action_index,
            }
        else:
            traj_frag = {
                '_DONE_': False, 
                '_SKIP_': False, 
                '_TOBS_': None, 
                'obs': frames,
                'action': action_index,
                'action_index': action_index,
            }
        traj_frag = self.hmap_multi_env_compat(traj_frag)
        # self.traj_manager.feed_traj_framedata(traj_frag, require_hook=False)
        if done:
            print绿('[ReinforceAlgorithmFoundation] episode done, all nets reset')
            self.policy.reset()
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
        self.trainer.save_model(info=info)

    def load_model(self):
        self.trainer.load_model()