import torch, copy, cv2, os, time
import numpy as np
import torch.nn as nn
import torch.nn.functional as F
from typing import Union, List

from torchsummary import summary
from torchvision.models import efficientnet_b0, efficientnet_b5
from siri.utils.logger import lprint
from siri.vision.preprocess import crop_wh, apply_gamma, apply_contrast, flood_fill_segment, find_darkest_pixel
from imitation.discretizer import SimpleDiscretizer, wasd_Discretizer
from imitation.utils import iterable_eq
from UTIL.colorful import *
from .conf import AlgorithmConfig


x_box = [225, 195, 155, 130, 105, 85, 65, 45, 32.5, 20, 12.5, 5]; x_box = np.array(x_box + [0] + (-1 * np.array(list(reversed(copy.copy(x_box))))).tolist(), dtype=np.float32)
y_box = [50, 25, 12.5, 5]; y_box = np.array(y_box + [0] + (-1 * np.array(list(reversed(copy.copy(y_box))))).tolist(), dtype=np.float32)

Y_MAX=300
Y_D_MAX=200

from imitation.transform import center_transform_train, center_transform_test, center_transform_train_f

def load_model(m, pt_path, device='cuda'):
    if not os.path.exists(pt_path): 
        assert False, "file does not exists"

    cpt = torch.load(pt_path, map_location=device)
    m.load_state_dict(cpt['policy'], strict=True)
    print绿(f'loaded model {pt_path}')
    return m

class MapNetBase(nn.Module):
    _showed = not AlgorithmConfig.show_preprocessed_preview

    x_discretizer = SimpleDiscretizer(x_box)
    y_discretizer = SimpleDiscretizer(y_box, MAX=Y_MAX, D_MAX=Y_D_MAX)
    wasd_discretizer = wasd_Discretizer()


    CENTER_SZ_WH = (400, 189,)
    MAP_SZ_WH = (181, 124)
    use_map_aug = True
    def __init__(self):
        super(MapNetBase, self).__init__()
        input_sz_wh = self.CENTER_SZ_WH
        self.input_frame_shape = (3,) + tuple(reversed(input_sz_wh))
        lprint(self, f"input frame shape: {self.input_frame_shape}, if it's changed, errors may occur")
    
        self.reset()

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
        return image
    

    def reset(self): self.hs = None
    
    def load_model(self, path):
        load_model(self, path, device='cuda')


    
    @staticmethod
    def get_efficientnet_b5():
        base_model = efficientnet_b5(weights='IMAGENET1K_V1')
        features = list(base_model.features.children())[:6]
        features = nn.Sequential(*features)
        t, h, w = 176, 12, 25
        return features, t, h, w
