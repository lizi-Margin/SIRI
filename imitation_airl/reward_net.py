import torch, copy, cv2, os, time, abc
import numpy as np
import torch.nn as nn
import torch.nn.functional as F
from typing import Union, List
from torchvision.transforms import Compose

from UTIL.colorful import *
from imitation.conv_lstm import ConvLSTM
from UTIL.tensor_ops import Args2tensor_Return2numpy

from .map_net_base import MapNetBase

class RewardNet(nn.Module, abc.ABC):
    @Args2tensor_Return2numpy
    def get_reward_test(self, *args, **kargs):
        """reward without shaping used to train RL policy"""
        return self.forward(*args, **kargs)

    @Args2tensor_Return2numpy
    def get_reward_train(self, *args, **kargs):
        """shaped reward used to train generator policy"""
        return self.forward(*args, **kargs)



class MapRewardBase(MapNetBase, RewardNet):
    def init_head(self, thw):
        hdim = min(1024, thw//2)
        self.head = nn.Sequential(
            nn.Linear(thw, hdim), nn.ReLU(inplace=True),
            nn.Linear(hdim, 1)
        )
    
    @torch.no_grad()
    def get_reward(self, frames, actions, frame_centers=None):
        if isinstance(frames, list):
            assert len(frames) == 1
            assert isinstance(frames[0], np.ndarray)
            assert len(frames[0].shape) == 3
        if isinstance(frames, np.ndarray):
            assert len(frames.shape) == 3
            frames = [frames]
            actions = [actions]

        actions = torch.from_numpy(np.array(actions)).float().to('cuda')
        if frame_centers is None:
            frame_centers = [self.get_center(f.copy()) for f in frames]
            frame_centers = self.preprocess(frame_centers, train=False)
        reward = self.forward(frame_centers, actions, use_hs=True)
        return reward.cpu().numpy()


# def normalize_actions(actions):
#     assert isinstance(actions, torch.Tensor)
#     n, action_dim = actions.size()
#     if action_dim == 6:
#         wasd, x, y = actions[:, 0:4], actions[:, 4], actions[:, 5]
#     else:
#         raise NotImplementedError(f"action_dim: {action_dim}")


class DoubleBranchMapReward(MapRewardBase):
    def __init__(self, action_dim):
        super().__init__()
        self.action_dim = action_dim

        self.features, t, h, w = self.get_efficientnet_b5() 
        self.map_features, mt, _, _ = self.get_efficientnet_b5() 
        mh, mw = 8, 12

        self.conv_lstm = ConvLSTM(
            input_dim=t,
            hidden_dim=t,
            kernel_size=(3, 3),
            num_layers=1,
            batch_first=True
        )

        self.map_conv_lstm = ConvLSTM(
            input_dim=mt,
            hidden_dim=mt,
            kernel_size=(3, 3),
            num_layers=1,
            batch_first=True
        ) 
        thw = (t * h * w) + (mt * mh * mw)
        obs_feature_dim = min(1520, thw)
        act_feature_dim = max(action_dim, obs_feature_dim//5)
        who = f"[{self.__class__.__name__}]: "
        print黄(f"{who}vision_feature_dim: {obs_feature_dim}")
        print黄(f"{who}action_feature_dim: {act_feature_dim}")
        print黄(f"{who}action_dim: {action_dim}")

        self.obs_emb = torch.nn.Sequential(
            torch.nn.Linear(thw, obs_feature_dim),
            torch.nn.ReLU(inplace=True)
        )
        self.action_emb = torch.nn.Sequential(
            torch.nn.Linear(action_dim, act_feature_dim),
            torch.nn.ReLU(inplace=True)
        )
        self.obs_act_lstm = torch.nn.LSTM(
            input_size=obs_feature_dim + act_feature_dim,
            hidden_size=obs_feature_dim + act_feature_dim,
            num_layers=1,
            batch_first=True,
        )

        self.init_head(obs_feature_dim + act_feature_dim)
        self.reset()
    
    def reset(self):
        self.hs = [None, None, None]

    def forward(self, obs, action, use_hs=False):
        obs, map_in = obs
        seq_len, channels, height, width = obs.size() 

        features = self.features(obs)
        map_features = self.map_features(map_in)

        if not use_hs:
            hidden_state = None
            map_hidden_state = None
            obs_act_hs = None
        else:
            assert seq_len == 1
            hidden_state = self.hs[0]
            map_hidden_state = self.hs[1]
            obs_act_hs = self.hs[2]


        o, hs = self.conv_lstm(features.unsqueeze(0), hidden_state=hidden_state)
        map_o, map_hs = self.map_conv_lstm(map_features.unsqueeze(0), hidden_state=map_hidden_state)
        o = o[0].squeeze(0)
        map_o = map_o[0].squeeze(0)
        assert (o.shape[0] == seq_len)
        assert (map_o.shape[0] == seq_len)
        o = o.reshape(o.shape[0], -1)
        map_o = map_o.reshape(map_o.shape[0], -1)

        o = torch.cat([o, map_o], dim=1)

        o = self.obs_emb(o)
        act_feature = self.action_emb(action)
        o, obs_act_hs = self.obs_act_lstm(torch.cat([o, act_feature], dim=1).unsqueeze(0), hx=obs_act_hs)
        o = o[0].squeeze(0)
        if use_hs: self.hs = [hs, map_hs, obs_act_hs]

        return self.head(o)
    