import torch, copy, cv2, os, time
import numpy as np
import torch.nn as nn
import torch.nn.functional as F
from typing import Union, List
from torchvision.transforms import Compose

from .net import x_box, y_box
from .net import Y_MAX, Y_D_MAX
from .net import NetActor

from imitation.transform import center_transform_train, center_transform_test
from siri.vision.preprocess import crop_wh, apply_gamma, apply_contrast, flood_fill_segment, find_darkest_pixel
from imitation.utils import iterable_eq
from UTIL.colorful import *
from imitation.conv_lstm import ConvLSTM

class MapNetBase(NetActor):
    MAP_SZ_WH = (181, 124)
    use_map_aug = True
    
    @classmethod
    def preprocess(cls, imgs: Union[np.ndarray, List[np.ndarray]], train=True) -> torch.Tensor: # to('cuda')
        assert isinstance(imgs, list)
        assert isinstance(imgs[0], (tuple, list,))
        assert len(imgs[0]) == 2
        center_imgs, map_imgs = [], []
        for (center_img, map_img) in imgs:
            center_imgs.append(center_img.copy())
            map_imgs.append(map_img.copy())

        # center_imgs = np.array([cls.get_center_(frame.copy()) for frame in imgs])
        # map_imgs = np.array([cls.get_map(frame.copy()) for frame in imgs])

        def map_trans(image):
            image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            transform = center_transform_train if train else center_transform_test
            image = transform(image)
            assert image.shape[0] == 3
            return image

            
        map_t = torch.stack([map_trans(img) for img in map_imgs])
        map_t = map_t.to('cuda').float()

        def trans(image):
            image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            transform = center_transform_train if train else center_transform_test
            image = transform(image)
            assert image.shape[0] == 3
            return image
        center_t = torch.stack([trans(img) for img in center_imgs])
        center_t = center_t.to('cuda').float()

        

        if not cls._showed and train:
            for i in range(5):
                center_0 = ((center_t[i].cpu().permute(1, 2, 0).numpy() + 1)/2 * 255).astype(np.uint8)[..., ::-1]
                # midas_0 = ((midas_t[i].cpu().permute(1, 2, 0).numpy() + 1)/2 * 255).astype(np.uint8)[..., ::-1]
                map_0 = ((map_t[i].cpu().permute(1, 2, 0).numpy() + 1)/2 * 255).astype(np.uint8)[..., ::-1]
                cv2.imshow('raw', center_0)
                # cv2.imshow('midas', midas_0)
                cv2.imshow('map', map_0)
                cv2.waitKey(0)
                cv2.destroyAllWindows()
            cls._showed = True
        return center_t, map_t


    @classmethod
    def get_center(cls, frame):
        return (cls.get_center_(frame.copy()), cls.get_map(frame),)

    @classmethod
    def get_center_(cls, frame):
        if not iterable_eq(frame.shape, (578, 1280, 3)):
            print(f"[MapNetActor] Warning: input shape is {frame.shape}, use resize")
            frame = cv2.resize(frame, (1280, 578))
        assert frame.shape[0] == 578, frame.shape[1] == 1280
        frame = crop_wh(frame, 240, 100)
        assert frame.shape[0] == 378, frame.shape[1] == 800
        frame = cv2.resize(frame, cls.CENTER_SZ_WH)
        return frame
    
    @classmethod
    def get_map(cls, image: np.ndarray):
        assert isinstance(image, np.ndarray)
        assert len(image.shape) == 3
        if not iterable_eq(image.shape, (578, 1280, 3)):
            print(f"[MapNetActor] Warning: input shape is {image.shape}, use resize")
            image = cv2.resize(image, (1280, 578))
        h, w = image.shape[:2]
        # btm = int(h/3.5)
        # left = int(w - w/7.5)
        # right = int(w - w/100)
        left, right = 1092, 1273
        top, btm = 9, 133
        # print(left, right, btm)
        corner = image[top:btm, left:right, :]
        corner = cls.map_aug(corner)
        sz = corner.shape
        assert len(sz) == 3
        assert sz[0] == cls.MAP_SZ_WH[1], sz[1] == cls.MAP_SZ_WH[0]
        return corner

    @classmethod
    def map_aug(cls, image):
        if cls.use_map_aug:
            image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            image[..., 0] = np.clip(image[..., 0], 0, 160)
            image[..., 1] = np.clip(image[..., 1], 0, 160)
            image[..., 2] = np.clip(image[..., 2], 0, 160)
            image = apply_gamma(image, gamma=0.5)
            # image = apply_contrast(image, contrast_coef=1.5)
            image[..., 0] = np.clip(image[..., 0], 0, 100)
            image[..., 1] = np.clip(image[..., 1], 0, 100)
            image[..., 2] = np.clip(image[..., 2], 0, 180)
            image = cv2.normalize(image, None, 0, 255, cv2.NORM_MINMAX)
            image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)


            # m_r_gt_140 = cv2.inRange(image[..., 0], 140, 255)
            # m_g_gt_140 = cv2.inRange(image[..., 1], 140, 255)
            # m_b_gt_140 = cv2.inRange(image[..., 2], 140, 255)

            # m_enm = cv2.bitwise_and(m_r_gt_140, cv2.bitwise_not(cv2.bitwise_or(m_g_gt_140, m_b_gt_140)))
            # m_self = cv2.bitwise_and(cv2.bitwise_and(m_r_gt_140, m_g_gt_140), cv2.bitwise_not(m_b_gt_140))
            # m_friend = cv2.bitwise_and(cv2.bitwise_and(m_g_gt_140, m_b_gt_140), cv2.bitwise_not(m_r_gt_140))

            # # m_obj = cv2.bitwise_or(m_enm, cv2.bitwise_or(m_self, m_friend))
            # # image[..., 0] = cv2.bitwise_and(image[..., 0], cv2.bitwise_not(m_obj))
            # # image[..., 1] = cv2.bitwise_and(image[..., 1], cv2.bitwise_not(m_obj))
            # # image[..., 2] = cv2.bitwise_and(image[..., 2], cv2.bitwise_not(m_obj))

            # image[..., 0] = np.clip(image[..., 0], 0, 100)
            # image[..., 1] = np.clip(image[..., 1], 0, 100)
            # image[..., 2] = np.clip(image[..., 2], 0, 100)
            # image = apply_contrast(image, contrast_coef=1.9)


            # _, targe_w, target_h = find_darkest_pixel(cv2.cvtColor((image.copy()), cv2.COLOR_RGB2BGR), 91, 53, radius=10)
            # print(target_h, targe_w)
            # image, _ = flood_fill_segment(cv2.cvtColor(image, cv2.COLOR_RGB2BGR), (targe_w, target_h), tolerance=(100, 100, 100)); image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            # image = cv2.normalize(image, None, 0, 255, cv2.NORM_MINMAX)

            # m = cv2.inRange(image[..., 2], 150, 255)
            # image[..., 2] = cv2.bitwise_or(image[..., 2], m)

            # m = cv2.inRange(image[..., 0], 150, 255)
            # image[..., 0] = cv2.bitwise_or(image[..., 0], m)

            # m = cv2.inRange(image[..., 0], 0, 15)
            # m = cv2.bitwise_and(m, cv2.inRange(image[..., 1], 0, 15))
            # m = cv2.bitwise_and(m, cv2.inRange(image[..., 2], 0, 25))
            # m = cv2.bitwise_not(m)
            # image[..., 0] = cv2.bitwise_and(image[..., 0], m)
            # image[..., 1] = cv2.bitwise_and(image[..., 1], m)
            # image[..., 2] = cv2.bitwise_and(image[..., 2], m)

            # image = apply_gamma(image, 1.6)
            # image = cv2.normalize(image, None, 0, 255, cv2.NORM_MINMAX)

            # # 锐化处理（使用拉普拉斯算子）
            # kernel = np.array([[0, -1, 0], [-1, 5, -1], [0, -1, 0]])
            # image = cv2.filter2D(image, -1, kernel)
        return image

    def act(self, frames):
        assert len(frames) == 1

        frame_centers = [self.get_center(f.copy()) for f in frames]
        frame_centers = self.preprocess(frame_centers, train=False)
        index = self._act(frame_centers)
        index = tuple(int(x[0]) for x in index)
        index_wasd, index_x, index_y = index[0], index[1], index[2]
        wasd = self.wasd_discretizer.index_to_action_(index_wasd)
        x = self.x_discretizer.index_to_action_(index_x)
        y = self.y_discretizer.index_to_action_(index_y)
        info = {
            'jump' : index[3],
            'crouch' : index[4],
            'reload' : index[5],
            'mouse_right' : index[6],
            'mouse_left' : index[7]
        }
        return wasd, np.array([x, y]), info


class DoubleBranchMapNet(MapNetBase):
    def __init__(self):
        super().__init__()
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
        self.init_act_head((t * h * w) + (mt * mh * mw))
    
    def reset(self):
        self.hs = [None, None]

    def forward(self, x, train=True):
        x, map_in = x
        seq_len, channels, height, width = x.size() 

        features = self.features(x)
        map_features = self.map_features(map_in)

        if train:
            hidden_state = None
            map_hidden_state = None
        else:
            hidden_state = self.hs[0]
            map_hidden_state = self.hs[1]


        o, hs = self.conv_lstm(features.unsqueeze(0), hidden_state=hidden_state)
        map_o, map_hs = self.map_conv_lstm(map_features.unsqueeze(0), hidden_state=map_hidden_state)
        if not train: self.hs = [hs, map_hs]
        o = o[0].squeeze(0)
        map_o = map_o[0].squeeze(0)
        assert (o.shape[0] == seq_len)
        assert (map_o.shape[0] == seq_len)
        o = o.reshape(o.shape[0], -1)
        map_o = map_o.reshape(map_o.shape[0], -1)

        o = torch.cat([o, map_o], dim=1)

        return tuple(self.fc_layers[name](o) for name in self.fc_layers)
    