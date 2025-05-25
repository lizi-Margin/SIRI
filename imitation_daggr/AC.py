import torch
import numpy as np
import torch.nn as nn
import torch.nn.functional as F
from torchvision.transforms import Compose
from torch.distributions.bernoulli import Bernoulli
from torch.distributions.categorical import Categorical
from siri.vision.preprocess import crop_wh
from imitation.utils import iterable_eq

import cv2
from imitation.transform import center_transform_train, center_transform_test, center_transform_train_f
from typing import Union, List
from .conf import AlgorithmConfig
from imitation.conv_lstm import ConvLSTM

from .map_net_base import x_box, y_box
from .map_net_base import Y_MAX, Y_D_MAX
from .map_net_base import MapNetBase

from UTIL.colorful import *
from UTIL.tensor_ops import Args2tensor
from imitation.conv_lstm import ConvLSTM

class MapACBase(MapNetBase):
    def get_frame_centers(self, frames):
        if isinstance(frames, list):
            assert len(frames) == 1
            assert isinstance(frames[0], np.ndarray)
            assert len(frames[0].shape) == 3
        if isinstance(frames, np.ndarray):
            if len(frames.shape) != 4:
                assert len(frames.shape) == 3, f"frames.shape: {frames.shape}"
                frames = [frames]

        frame_centers = [self.get_center(f.copy()) for f in frames]
        frame_centers = self.preprocess(frame_centers, train=False)
        return frame_centers

    @torch.no_grad()
    def act(self, frames, frame_centers=None):
        if frame_centers is None: 
            frame_centers = self.get_frame_centers(frames)
        index, value, actLogProbs = self._act(frame_centers, eval_mode=False, use_hs=True)
        index = tuple(int(x[0]) for x in index)
        index_wasd, index_x, index_y = index[0], index[1], index[2]
        wasd = self.wasd_discretizer.index_to_action_(index_wasd)
        x = self.x_discretizer.index_to_action_(index_x)
        y = self.y_discretizer.index_to_action_(index_y)
        info = {
            # 'jump' : index[3],
            # 'crouch' : index[4],
            # 'reload' : index[5],
            # 'mouse_right' : index[6],
            # 'mouse_left' : index[7]
            'value': value,
            'actLogProbs': actLogProbs,
            'index_x': index_x,
            'index_y': index_y,
            'index_wasd': index_wasd,
        }
        return wasd, np.array([x, y]), info

    @Args2tensor
    def evaluate_actions(self, *args, **kargs):
        return self._act(*args, **kargs, eval_mode=True, use_hs=False)

    def _act(self, obs, eval_mode=False, eval_actions=None, use_hs=False):
        seq_len, channels, height, width = obs[0].size()
        logits, value = self.forward(obs, use_hs=use_hs)
        act, actLogProbs, distEntropy, probs = self._logit2act(
            logits,
            eval_mode=eval_mode,
            eval_actions=eval_actions,
        )
        if not eval_mode: return act, value, actLogProbs
        else:             return value, actLogProbs, distEntropy, probs

    
    def init_head(self, thw):
        self.wasd_n_actions = self.wasd_discretizer.n_actions
        self.x_n_actions = self.x_discretizer.n_actions
        self.y_n_actions = self.y_discretizer.n_actions
        self.n_actions = self.wasd_n_actions + self.x_n_actions + self.y_n_actions
        shared_dim = min(1024, thw//2)
        print黄(f"[{self.__class__.__name__}]: 输出头 shared_dim={shared_dim}")

        self.shared_head_layer = nn.Sequential(
            nn.Linear(thw, shared_dim),
            nn.ReLU(inplace=True)
        )

        head_h_dim = shared_dim//2
        self.act_head = nn.Sequential(
            nn.Linear(shared_dim, head_h_dim),
            nn.ReLU(inplace=True),
            nn.Linear(head_h_dim, self.n_actions),
        )
        self.value_head = nn.Sequential(
            nn.Linear(shared_dim, head_h_dim),
            nn.ReLU(inplace=True),
            nn.Linear(head_h_dim, 1)
        )
    
    @staticmethod
    def logit_from_to(logit, from_, to_):
        return logit[:, from_:to_]

    def _logit2act(self, logit, argmax=True,
                   eval_mode=False, eval_actions=None):
        if eval_mode:
            assert eval_actions is not None
                   
        logit_wasd = self.logit_from_to(logit, 0, self.wasd_n_actions)
        logit_x = self.logit_from_to(logit, self.wasd_n_actions, self.wasd_n_actions + self.x_n_actions)        
        logit_y = self.logit_from_to(logit, self.wasd_n_actions + self.x_n_actions, self.wasd_n_actions + self.x_n_actions + self.y_n_actions)

        dist_wasd = Categorical(logits=logit_wasd)
        dist_x = Categorical(logits=logit_x)
        dist_y = Categorical(logits=logit_y)
        # dist_jump = Bernoulli(logits=logit_jump)
        # dist_crouch = Bernoulli(logits=logit_crouch)
        # dist_reload = Bernoulli(logits=logit_reload)
        # dist_r = Bernoulli(logits=logit_r)
        # dist_l = Bernoulli(logits=logit_l)

        if not eval_mode:
            if argmax:
                act_wasd = torch.argmax(logit_wasd, dim=-1)
                act_x = torch.argmax(logit_x, dim=-1)
                # act_x = dist_x.sample()
                act_y = torch.argmax(logit_y, dim=-1)
                # act_jump = torch.sigmoid(logit_jump) > 0.5,
                # act_crouch = torch.sigmoid(logit_crouch) > 0.5,
                # act_reload = torch.sigmoid(logit_reload) > 0.5,
                # act_r = torch.sigmoid(logit_r) > 0.5,
                # act_l = torch.sigmoid(logit_l) > 0.5,
                act = (
                    act_wasd,
                    act_x,
                    act_y,
                    # act_jump,
                    # act_crouch,
                    # act_reload,
                    # act_r,
                    # act_l,
                )
            else:
                act_wasd = dist_wasd.sample()
                act_x = dist_x.sample()
                act_y = dist_y.sample()
                # act_jump = dist_jump.sample()
                # act_crouch = dist_crouch.sample()
                # act_reload = dist_reload.sample()
                # act_r = dist_r.sample()
                # act_l = dist_l.sample()
                act = (
                    act_wasd,
                    act_x,
                    act_y,
                    # act_jump,
                    # act_crouch,
                    # act_reload,
                    # act_r,
                    # act_l,
                )
        else:
            assert eval_actions is not None
            act_wasd = eval_actions[:, 0]
            act_x = eval_actions[:, 1]
            act_y = eval_actions[:, 2]
            # act_jump = eval_actions[3]
            # act_crouch = eval_actions[4]
            # act_reload = eval_actions[5]
            # act_r = eval_actions[6]
            # act_l = eval_actions[7]
            act = (
                act_wasd,
                act_x,
                act_y,
                # act_jump,
                # act_crouch,
                # act_reload,
                # act_r,
                # act_l,
            )

        probs = (
            dist_wasd.probs,
            dist_x.probs,
            dist_y.probs,
            # dist_jump.probs,
            # dist_crouch.probs,
            # dist_reload.probs,
            # dist_r.probs,
            # dist_l.probs,
        )

        category_actLogProbs = (
            dist_wasd.log_prob(act_wasd)
            + dist_x.log_prob(act_x)
            + dist_y.log_prob(act_y)
        )/3
        # binary_actLogProbs = AlgorithmConfig.binary_coef * (
        #     dist_jump.log_prob(index_jump) + dist_crouch.log_prob(index_crouch) + dist_reload.log_prob(index_reload)
        #     + dist_r.log_prob(index_r) + dist_l.log_prob(index_l)
        # )/5
        # cross_entropy_loss = -(category_actLogProbs + binary_actLogProbs).mean() # mean log probility of the expert actions -> cross entropy loss -> max likelyhood
        # binary_actLogProbs = t2n_mean(binary_actLogProbs)
        # category_actLogProbs = t2n_mean(category_actLogProbs)
        

        category_distEntropy = (
            dist_wasd.entropy() + dist_x.entropy() + dist_y.entropy()
        )/3
        # binary_distEntropy = AlgorithmConfig.binary_coef * (
        #     dist_jump.entropy() + dist_crouch.entropy() + dist_reload.entropy()
        #     + dist_r.entropy() + dist_l.entropy()
        # )/5
        # mean_distEntropy = (category_distEntropy + binary_distEntropy).mean()
        return act, category_actLogProbs, category_distEntropy, probs


class DoubleBranchMapAC(MapACBase):
    def __init__(self):
        super().__init__()
        self.features, t, h, w = self.get_efficientnet_b5() 
        self.map_features, mt, _, _ = self.get_efficientnet_b5() 
        mh, mw = 8, 12

        # lstm_o_chns = t//2, mt//2
        # conv_lstm_layers = 2

        lstm_o_chns = t, mt
        conv_lstm_layers = 1

        self.conv_lstm = ConvLSTM(
            input_dim=t,
            hidden_dim=lstm_o_chns[0],
            kernel_size=(3, 3),
            num_layers=conv_lstm_layers,
            batch_first=True
        )
        self.map_conv_lstm = ConvLSTM(
            input_dim=mt,
            hidden_dim=lstm_o_chns[1],
            kernel_size=(3, 3),
            num_layers=conv_lstm_layers,
            batch_first=True
        ) 
        self.init_head((lstm_o_chns[0] * h * w) + (lstm_o_chns[1] * mh * mw))

        self.train()
    
    def reset(self):
        self.hs = [None, None]

    def forward(self, x_and_map, use_hs=False):
        x, map_in = x_and_map # screen center of the game and the corner of map
        seq_len, channels, height, width = x.size() 

        x = torch.nan_to_num_(x, 0)
        map_in = torch.nan_to_num_(map_in, 0)

        features = self.features(x)
        map_features = self.map_features(map_in)

        if not use_hs:
            hs = None
            map_hs = None
        else:
            hs = self.hs[0]
            map_hs = self.hs[1]


        o, hs = self.conv_lstm(features.unsqueeze(0), hidden_state=hs)
        map_o, map_hs = self.map_conv_lstm(map_features.unsqueeze(0), hidden_state=map_hs)
        if use_hs: self.hs = [hs, map_hs]
        o = o[0].squeeze(0)
        map_o = map_o[0].squeeze(0)
        assert (o.shape[0] == seq_len)
        assert (map_o.shape[0] == seq_len)
        o = o.reshape(o.shape[0], -1)
        map_o = map_o.reshape(map_o.shape[0], -1)

        o = torch.cat([o, map_o], dim=1)
        o = self.shared_head_layer(o)

        return self.act_head(o), self.value_head(o)


class YoloMapAC(MapACBase):
    @classmethod
    def preprocess(cls, imgs: Union[np.ndarray, List[np.ndarray]], train=True) -> torch.Tensor: # to('cuda')
        assert isinstance(imgs, list)
        assert isinstance(imgs[0], (tuple, list,))
        assert len(imgs[0]) == 3
        center_imgs, map_imgs, raw_frames = [], [], []
        for (center_img, map_img, raw_frame) in imgs:
            center_imgs.append(center_img)
            map_imgs.append(map_img)
            raw_frames.append(raw_frame)

        def raw_trans(image):
            image = image[..., ::-1].transpose((2, 0, 1))  # BGR to RGB, BHWC to BCHW, (n, 3, h, w)
            image = np.ascontiguousarray(image)  # contiguous
            image = torch.from_numpy(image).float().to(AlgorithmConfig.device) / 255.
            assert image.shape[0] == 3
            return image
        raw_t = torch.stack([raw_trans(img) for img in raw_frames])
        raw_t = raw_t.to(AlgorithmConfig.device).float()

        def map_trans(image):
            image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            transform = center_transform_train if train else center_transform_test
            image = transform(image)
            assert image.shape[0] == 3
            return image
        map_t = torch.stack([map_trans(img) for img in map_imgs])
        map_t = map_t.to(AlgorithmConfig.device).float()

        def trans(image):
            image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            transform = center_transform_train if train else center_transform_test
            image = transform(image)
            assert image.shape[0] == 3
            return image
        center_t = torch.stack([trans(img) for img in center_imgs])
        center_t = center_t.to(AlgorithmConfig.device).float()

        

        if not cls._showed and train:
            for i in range(5):
                center_0 = ((center_t[i].cpu().permute(1, 2, 0).numpy() + 1)/2 * 255).astype(np.uint8)[..., ::-1]
                map_0 = ((map_t[i].cpu().permute(1, 2, 0).numpy() + 1)/2 * 255).astype(np.uint8)[..., ::-1]
                raw_0 = ((raw_t[i].cpu().permute(1, 2, 0).numpy() + 1)/2 * 255).astype(np.uint8)[..., ::-1]
                cv2.imshow('center', center_0)
                cv2.imshow('map', map_0)
                cv2.imshow('raw', raw_0)
                cv2.waitKey(0)
                cv2.destroyAllWindows()
            cls._showed = True
        return center_t, map_t, raw_t

    
    @classmethod
    def get_yolo_field(cls, frame):
        if not iterable_eq(frame.shape, (578, 1280, 3)):
            print(f"[MapNetActor] Warning: input shape is {frame.shape}, use resize")
            frame = cv2.resize(frame, (1280, 578))
        assert frame.shape[0] == 578, frame.shape[1] == 1280
        frame = crop_wh(frame, 240, 100)
        assert frame.shape[0] == 378, frame.shape[1] == 800
        frame = cv2.resize(frame, (384, 160,))
        return frame

    @classmethod
    def get_center(cls, frame):
        assert isinstance(frame, np.ndarray)
        assert len(frame.shape) == 3
        assert frame.shape[0] == 578
        assert frame.shape[1] == 1280
        assert frame.shape[2] == 3
        return (cls.get_center_(frame.copy()), cls.get_map(frame.copy()), cls.get_yolo_field(frame))
    
    def get_yolo(self):
        from ultralytics import YOLO
        m = YOLO(model='best.pt').model.to(AlgorithmConfig.device)
        for param in m.parameters():
            param.requires_grad = False
        return m


    def __init__(self):
        super().__init__()
        self.features, t, h, w = self.get_efficientnet_b5() 
        self.map_features, mt, _, _ = self.get_efficientnet_b5() 
        mh, mw = 8, 12

        self.yolo_features = self.get_yolo()
        yt, yh, yw = 150, 10, 24
        # yt, yh, yw = 18, 4, 10
        # yt, yh, yw = 18, 18, 40
        # yt, yh, yw = 54, 18, 40
        

        lstm_o_chns = t, mt, yt
        conv_lstm_layers = 1
        self.conv_lstm = ConvLSTM(
            input_dim=t,
            hidden_dim=lstm_o_chns[0],
            kernel_size=(3, 3),
            num_layers=conv_lstm_layers,
            batch_first=True
        )
        self.map_conv_lstm = ConvLSTM(
            input_dim=mt,
            hidden_dim=lstm_o_chns[1],
            kernel_size=(3, 3),
            num_layers=conv_lstm_layers,
            batch_first=True
        ) 
        self.yolo_conv_lstm = ConvLSTM(
            input_dim=yt,
            hidden_dim=lstm_o_chns[2],
            kernel_size=(3, 3),
            num_layers=conv_lstm_layers,
            batch_first=True
        ) 
        self.init_head((lstm_o_chns[0] * h * w) + (lstm_o_chns[1] * mh * mw) + (lstm_o_chns[2] * yh * yw))

        self.train()
    
    def reset(self):
        self.hs = [None, None, None]

    def forward(self, x_and_map_and_frame, use_hs=False):
        x, map_in, frame = x_and_map_and_frame # screen center of the game and the corner of map
        seq_len, channels, height, width = x.size() 

        x = torch.nan_to_num_(x, 0)
        map_in = torch.nan_to_num_(map_in, 0)

        features = self.features(x)
        map_features = self.map_features(map_in)
        with torch.no_grad():
            # yolo_features = self.get_yolo_feature_class_0_p3(self.yolo_features(frame))
            yolo_features = self.get_yolo_feature_p2_p3(self.yolo_features(frame))
            print(yolo_features.shape)

        if not use_hs:
            hs = None
            map_hs = None
            yolo_hs = None
        else:
            hs = self.hs[0]
            map_hs = self.hs[1]
            yolo_hs = self.hs[2]


        o, hs = self.conv_lstm(features.unsqueeze(0), hidden_state=hs)
        map_o, map_hs = self.map_conv_lstm(map_features.unsqueeze(0), hidden_state=map_hs)
        yolo_o, yolo_hs = self.yolo_conv_lstm(yolo_features.unsqueeze(0), hidden_state=yolo_hs)
        if use_hs: self.hs = [hs, map_hs, yolo_hs]
        o = o[0].squeeze(0)
        map_o = map_o[0].squeeze(0)
        yolo_o = yolo_o[0].squeeze(0)
        assert (o.shape[0] == seq_len)
        assert (map_o.shape[0] == seq_len)
        assert (yolo_o.shape[0] == seq_len), f"{yolo_o.shape}"
        o = o.reshape(o.shape[0], -1)
        map_o = map_o.reshape(map_o.shape[0], -1)
        yolo_o = yolo_o.reshape(yolo_o.shape[0], -1)

        o = torch.cat([o, map_o, yolo_o], dim=1)
        o = self.shared_head_layer(o)

        return self.act_head(o), self.value_head(o)

    @staticmethod
    def get_p1_p2_p3(yolo_logit):
        if len(yolo_logit) == 2:
            # batch = 1
            p1 = yolo_logit[1][0]
            p2 = yolo_logit[1][1]
            p3 = yolo_logit[1][2]
        elif len(yolo_logit) == 3:
            # batch > 1 
            # print("len(yolo_logit): " , len(yolo_logit))
            # print("yolo_logit[0].shape: ", yolo_logit[0].shape)
            # print("len(yolo_logit[1]): ", len(yolo_logit[1]))
            # print("len(yolo_logit[2]): ", len(yolo_logit[2]))
            # print("yolo_logit[1][0].shape: ", yolo_logit[1][0].shape)
            # print("yolo_logit[1][1].shape: ", yolo_logit[1][1].shape)
            # print("yolo_logit[1][2].shape: ", yolo_logit[1][2].shape)
            # print("yolo_logit[2][0].shape: ", yolo_logit[2][0].shape)
            # print("yolo_logit[2][1].shape: ", yolo_logit[2][1].shape)
            # print("yolo_logit[2][2].shape: ", yolo_logit[2][2].shape)
            p1 = yolo_logit[0]
            p2 = yolo_logit[1]
            p3 = yolo_logit[2]
            assert p1.shape[0] == p1.shape[0] == p3.shape[0]
            assert p1.shape[1] == p1.shape[1] == p3.shape[1]  # channels
        else:
            raise NotImplementedError
        return p1, p2, p3

    @classmethod
    def get_yolo_feature_class_0_p3(cls, yolo_logit):
        p1, p2, p3 = YoloMapAC.get_p1_p2_p3(yolo_logit)
        Seq, C, H, W = p3.shape
        def _extract_class(feat):
            feat = feat.view(Seq, 3, 25, *feat.shape[-2:])
            # 提取坐标偏移(0:3)、置信度(4)和player概率(5)
            return torch.cat([
                feat[:,:,:4],  # 位置/尺寸信息
                feat[:,:,4:5], # 置信度
                feat[:,:,5:6]  # player分类logit
            ], dim=2)  # [1,3,6,H,W]
        p3_cls = _extract_class(p3) # [1,18,18,40]
        p3_cls = p3_cls.flatten(1,2) 
        return p3_cls

    @classmethod
    def get_yolo_feature_p3(cls, yolo_logit):
        p1, p2, p3 = YoloMapAC.get_p1_p2_p3(yolo_logit)
        return p3

    @classmethod
    def get_yolo_feature_p2_p3(cls, yolo_logit):
        p1, p2, p3 = YoloMapAC.get_p1_p2_p3(yolo_logit)
        Seq, C, H, W = p2.shape
        p3_up = F.interpolate(p3, size=(H,W), mode='bilinear')
        fused = torch.concatenate([
            p2, p3_up
        ], dim = 1)
        return fused  

    @classmethod
    def get_yolo_feature_class_0(cls, yolo_logit):
        p1, p2, p3 = YoloMapAC.get_p1_p2_p3(yolo_logit)
        """
        专注提取player(类别0)相关特征
        输入: 
            p1: [1,75,72,160] (P3)
            p2: [1,75,36,80]  (P4)
            p3: [1,75,18,40]  (P5)
        输出: 
            [1,32,72,160] player专属热力图特征
        """
        H, W = p1.shape[-2:]
        def _extract_class(feat):
            feat = feat.view(1, 3, 25, *feat.shape[-2:])
            # 提取坐标偏移(0:3)、置信度(4)和player概率(5)
            return torch.cat([
                feat[:,:,:4],  # 位置/尺寸信息
                feat[:,:,4:5], # 置信度
                feat[:,:,5:6]  # player分类logit
            ], dim=2)  # [1,3,6,H,W]
        
        # 三尺度特征处理
        p1_cls = _extract_class(p1).flatten(1,2)  # [1,18,72,160]
        p2_cls = _extract_class(p2).flatten(1,2)  # [1,18,36,80]
        p3_cls = _extract_class(p3).flatten(1,2)  # [1,18,18,40]
        
        # 上采样并融合
        p2_up = F.interpolate(p2_cls, size=(H,W), mode='bilinear')
        p3_up = F.interpolate(p3_cls, size=(H,W), mode='bilinear')
        
        fused = torch.concat([
            p1_cls,
            p2_up,
            p3_up,
        ], dim=1)
        return fused

class TransformerMapAC(MapACBase):
    def __init__(self, max_seq_len=45):
        super().__init__()
        hid_size = 4096
        self.hid_size = hid_size
        self.max_seq_len = max_seq_len 

        self.features, t, h, w = self.get_efficientnet_b5() 
        self.map_features, mt, mh, mw = self.get_efficientnet_b5() 
        mh, mw = 8, 12
        
        # 降维编码器
        from imitation.impala import ImpalaStyleEncoder
        self.encoder = ImpalaStyleEncoder(
            in_shape=(t, h, w),
            chns=[256, 512, 1024],
            hidsize=hid_size
        )
        self.map_encoder = ImpalaStyleEncoder(
            in_shape=(mt, mh, mw),
            chns=[256, 512, 1024],
            hidsize=hid_size//2
        )
        
        # 使用改进后的TransformerBlock
        from imitation.transformer import TransformerProcessor
        self.temporal_processor = TransformerProcessor(
            input_dim=hid_size, 
            num_heads=8, 
            num_layers=2,
            max_seq_len=max_seq_len
        )
        self.map_temporal_processor = TransformerProcessor(
            input_dim=hid_size//2, 
            num_heads=4, 
            num_layers=1,
            max_seq_len=max_seq_len
        )
        
        # 上下文记忆
        self.hs = None  # 用于存储历史信息
        self.current_seq_len = 0  # 当前记忆长度
        
        # 输出头初始化
        total_size = hid_size + hid_size//2 
        self.init_head(total_size)
        
    def reset(self):
        """重置历史记忆"""
        self.hs = None
        self.current_seq_len = 0
        
    def _process_with_memory(self, x, processor, is_map=False):
        """
        带记忆的时序处理
        Args:
            x: 当前输入 (seq_len, features)
            processor: TransformerBlock处理器
            is_map: 是否是地图分支
        """
        if self.hs is None:
            # 第一次调用，初始化记忆
            self.hs = {
                'main': None,
                'map': None
            }
        
        # 获取当前分支的记忆
        branch = 'map' if is_map else 'main'
        prev_memory = self.hs[branch]
        
        if prev_memory is None:
            # 没有历史记忆，直接处理当前输入
            combined = x.unsqueeze(0)  # (1, seq_len, features)
            self.current_seq_len = x.size(0)
        else:
            # 合并历史记忆和当前输入
            combined = torch.cat([prev_memory, x.unsqueeze(0)], dim=1)
            self.current_seq_len = combined.size(1)
            
            # 如果超过最大长度，截断最早的记忆
            if self.current_seq_len > self.max_seq_len:
                combined = combined[:, -self.max_seq_len:, :]
                self.current_seq_len = self.max_seq_len
        
        # 生成因果掩码
        mask = None
        if self.current_seq_len > 1:
            mask = torch.triu(torch.ones(self.current_seq_len, self.current_seq_len), diagonal=1)
            mask = mask.float().masked_fill(mask == 1, float('-inf')).to(x.device)
        
        # 通过Transformer处理
        output = processor(combined, mask=mask)
        
        # 更新记忆
        self.hs[branch] = output.detach()  # 分离计算图，避免梯度回传
        
        # 返回最后一步的输出
        # return output[:, -x.size(0):, :].squeeze(0)
        return output[:, -1:, :].squeeze(0)
        
    def forward(self, x_and_map, use_hs=False):
        x, map_in = x_and_map
        x = torch.nan_to_num(x, 0)
        map_in = torch.nan_to_num(map_in, 0)
        
        # 特征提取
        features = self.features(x)  # (seq_len, t, h, w)
        map_features = self.map_features(map_in)  # (seq_len, mt, mh, mw)
        
        # 降维编码
        encoded = self.encoder(features)  # (seq_len, hid_size)
        map_encoded = self.map_encoder(map_features)  # (seq_len, hid_size//2)
        
        # 时序处理
        if use_hs:
            # 使用记忆模式，seq_len应为1
            assert x.size(0) == 1, "In use_hs mode, seq_len must be 1"
            
            encoded = self._process_with_memory(encoded, self.temporal_processor)
            map_encoded = self._process_with_memory(
                map_encoded, self.map_temporal_processor, is_map=True
            )
        else:
            # 普通模式，处理整个序列
            seq_len = encoded.size(0)
            
            # 生成因果掩码
            mask = None
            if seq_len > 1:
                mask = torch.triu(torch.ones(seq_len, seq_len), diagonal=1)
                mask = mask.float().masked_fill(mask == 1, float('-inf')).to(x.device)
            
            encoded = self.temporal_processor(encoded.unsqueeze(0), mask=mask).squeeze(0)
            map_encoded = self.map_temporal_processor(
                map_encoded.unsqueeze(0), mask=mask
            ).squeeze(0)
        print(encoded.shape)
        print(map_encoded.shape)
        
        # 合并特征
        combined = torch.cat([encoded, map_encoded], dim=1)
        combined = self.shared_head_layer(combined)
        
        # 输出头
        return self.act_head(combined), self.value_head(combined)




class SwinTMapAC(MapACBase):
    def __init__(self, max_seq_len=45):
        super().__init__()
        hid_size = 4096
        self.hid_size = hid_size
        self.max_seq_len = max_seq_len

        # 初始化SwinT主干网络
        self.features = self.get_swint_backbone()
        self.map_features = self.get_swint_backbone(shallow=True)
        
        # 获取特征维度 (根据SwinT结构调整)
        t, h, w = 768, 6, 6  # 主分支输出形状 (400x189输入)
        mt, mh, mw = 384, 3, 3  # 小地图分支输出 (181x124输入)
        

    def get_swint_backbone(self, shallow=False) -> nn.Module:
        """构建SwinT主干网络"""
        from torchvision.models import swin_t, Swin_T_Weights
        model = swin_t(weights=Swin_T_Weights.IMAGENET1K_V1)
        
        # 分解特征提取部分
        layers = list(model.children())[:-2]  # 移除最后的head和norm
        
        # 构建特征提取器
        feature_extractor = nn.Sequential(
            *layers[:4] if shallow else layers,
            # 添加自适应池化保证输出尺寸一致
            nn.AdaptiveAvgPool2d((6, 6) if not shallow else (3, 3))
        )
        
        # # 冻结部分层 (可选)
        # for param in feature_extractor[:3].parameters():
        #     param.requires_grad = False

        # # 获取特征维度 (根据SwinT结构调整)
        # t, h, w = 768, 6, 6  # 主分支输出形状 (400x189输入)
        # mt, mh, mw = 384, 3, 3  # 小地图分支输出 (181x124输入)           
        return feature_extractor
    
    def preprocess_input(self, x: torch.Tensor, is_map: bool = False) -> torch.Tensor:
        """输入预处理"""
        # 标准化 (使用ImageNet统计量)
        mean = torch.tensor([0.485, 0.456, 0.406], device=x.device).view(1,3,1,1)
        std = torch.tensor([0.229, 0.224, 0.225], device=x.device).view(1,3,1,1)
        x = (x - mean) / std
        
        # 调整尺寸 (SwinT需要32的倍数)
        target_size = (384, 192) if not is_map else (128, 64)
        return nn.functional.interpolate(x, size=target_size, mode='bilinear')
