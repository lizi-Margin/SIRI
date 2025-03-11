import torch, copy, cv2, os
import numpy as np
import torch.nn as nn
from typing import Union, List
from torchsummary import summary
from torchvision.models import efficientnet_b0
from torch.distributions.categorical import Categorical
from siri.utils.logger import lprint
from siri.vision.preprocess import crop_wh
from imitation.discretizer import SimpleDiscretizer, wasd_Discretizer
from imitation.utils import iterable_eq
from UTIL.colorful import *

def load_model(m, pt_path, device='cuda'):
    if not os.path.exists(pt_path): 
        assert False, "file does not exists"

    cpt = torch.load(pt_path, map_location=device)
    m.load_state_dict(cpt['policy'], strict=True)
    print绿(f'loaded model {pt_path}')
    return m


class SimpleNet(nn.Module):
    def __init__(self, input_sz_wh, wasd_n_actions, x_n_actions, y_n_actions):
        super(SimpleNet, self).__init__()
        self.input_frame_shape = (3,) + tuple(reversed(input_sz_wh))
        lprint(self, f"input frame shape: {self.input_frame_shape}, if it's changed, errors may occur")
    

        base_model = efficientnet_b0(weights='IMAGENET1K_V1')
        # feature_extractor = nn.Sequential(*list(base_model.features.children())[:161])
        # summary(feature_extractor, input_size=self.input_frame_shape)

        features = list(base_model.features.children())[:6]
        self.features = nn.Sequential(*features)
        t, h, w = 112, 12, 25
        self.wasd_fc = nn.Linear(t * h * w, wasd_n_actions)
        self.x_fc = nn.Linear(t * h * w, x_n_actions)
        self.y_fc = nn.Linear(t * h * w, y_n_actions)

    def forward(self, x):
        # x的形状应该是 (batch_size, channels, height, width)
        batch_size, channels, height, width = x.size()

        x = self.features(x)
        # print(x.shape)
        x = x.view(batch_size, -1)
        # print(x.shape)

        logit_wasd = self.wasd_fc(x)
        logit_x = self.x_fc(x)
        logit_y = self.y_fc(x)

        return logit_wasd, logit_x, logit_y
    
    @torch.no_grad
    def act(self, x):
        batch_size, channels, height, width = x.size()
        assert batch_size == 1
        logit_wasd, logit_x, logit_y = self.forward(x)
        return torch.argmax(logit_wasd, dim=-1), torch.argmax(logit_x, dim=-1), torch.argmax(logit_y, dim=-1)

class SimpleNetActor(SimpleNet):
    x_box = [450, 350, 250, 150, 100, 50, 25, 10, 5]; x_box = np.array(x_box + [0] + (-1 * np.array(list(reversed(copy.copy(x_box))))).tolist(), dtype=np.float32)
    y_box = [50, 15, 5]; y_box = np.array(y_box + [0] + (-1 * np.array(list(reversed(copy.copy(y_box))))).tolist(), dtype=np.float32)

    x_discretizer = SimpleDiscretizer(x_box)
    y_discretizer = SimpleDiscretizer(y_box)
    wasd_discretizer = wasd_Discretizer()

    CENTER_SZ_WH = (400, 189,)

    @staticmethod
    def preprocess(im: Union[np.ndarray, List[np.ndarray]]) -> torch.Tensor: # to('cuda')
        assert len(im[0].shape) == 3 and im[0].shape[-1] == 3, "im shape should be (n, h, w, 3)"
        
        im = im[..., ::-1].transpose((0, 3, 1, 2))  # BGR to RGB, BHWC to BCHW, (n, 3, h, w)
        # im = im.transpose((0, 3, 1, 2))  # BHWC to BCHW, (n, 3, h, w)
        im = np.ascontiguousarray(im)  # contiguous
        im = torch.from_numpy(im)

        im = im.to('cuda')
        im = im.float()  # uint8 to fp16/32
        im /= 255  # 0 - 255 to 0.0 - 1.0
        return im

    @staticmethod
    def get_center(frame):
        if not iterable_eq(frame.shape, (578, 1280, 3)):
            print(f"[SimpleNetActor] Warning: input shape is {frame.shape}, use resize")
            frame = cv2.resize(frame, (1280, 578))
        assert frame.shape[0] == 578, frame.shape[1] == 1280
        frame = crop_wh(frame, 240, 100)
        assert frame.shape[0] == 378, frame.shape[1] == 800
        frame = cv2.resize(frame, SimpleNetActor.CENTER_SZ_WH)
        return frame
    
    def __init__(self):
        super().__init__(
            SimpleNetActor.CENTER_SZ_WH,
            SimpleNetActor.wasd_discretizer.n_actions,
            SimpleNetActor.x_discretizer.n_actions,
            SimpleNetActor.y_discretizer.n_actions
        )

    def act(self, frames):
        assert len(frames) == 1

        frame_centers = np.array([SimpleNetActor.get_center(f.copy()) for f in frames])
        frame_centers = SimpleNetActor.preprocess(frame_centers)
        index_wasd, index_x, index_y = super(SimpleNetActor, self).act(frame_centers)
        index_wasd, index_x, index_y = int(index_wasd[0]), int(index_x[0]), int(index_y[0])
        wasd = SimpleNetActor.wasd_discretizer.index_to_action_(index_wasd)
        x = SimpleNetActor.x_discretizer.index_to_action_(index_x)
        y = SimpleNetActor.y_discretizer.index_to_action_(index_y)
        return wasd, np.array([x, y])
        
        

class LSTMNet(nn.Module):
    def __init__(self, input_sz_wh, wasd_n_actions, x_n_actions, y_n_actions):
        super(LSTMNet, self).__init__()
        self.input_frame_shape = (3,) + tuple(reversed(input_sz_wh))
        lprint(self, f"input frame shape: {self.input_frame_shape}, if it's changed, errors may occur")
    

        base_model = efficientnet_b0(weights='IMAGENET1K_V1')
        # feature_extractor = nn.Sequential(*list(base_model.features.children())[:161])
        # summary(feature_extractor, input_size=self.input_frame_shape)

        features = list(base_model.features.children())[:6]
        self.features = nn.Sequential(*features)
        t, h, w = 112, 12, 25
        from .conv_lstm import ConvLSTM
        self.conv_lstm = ConvLSTM(
            input_dim=t,
            hidden_dim=t,
            kernel_size=(3, 3),
            num_layers=1,
            batch_first=True
        )
        # self.conv_lstm = nn.ConvLSTM2d(
        #     in_channels=t,
        #     out_channels=t,
        #     kernel_size=(3, 3),
        #     padding=1,
        #     batch_first=True
        # )


        self.wasd_fc = nn.Linear(t * h * w, wasd_n_actions)
        self.x_fc = nn.Linear(t * h * w, x_n_actions)
        self.y_fc = nn.Linear(t * h * w, y_n_actions)

        self.hs = None

    def forward(self, x, train=True):
        seq_len, channels, height, width = x.size()

        x = self.features(x)
        x = x.unsqueeze(1)
        x, hs = self.conv_lstm.forward(x, hidden_state=None if train else self.hs)
        if not train: self.hs = hs
        x = x[0].squeeze(0)
        # print(x.shape)
        x = x.view(seq_len, -1)
        # print(x.shape)

        logit_wasd = self.wasd_fc(x)
        logit_x = self.x_fc(x)
        logit_y = self.y_fc(x)

        return logit_wasd, logit_x, logit_y
    
    @torch.no_grad
    def act(self, x):
        seq_len, channels, height, width = x.size()
        assert seq_len == 1
        logit_wasd, logit_x, logit_y = self.forward(x, train=False)
        return torch.argmax(logit_wasd, dim=-1), torch.argmax(logit_x, dim=-1), torch.argmax(logit_y, dim=-1)
    
    def reset(self): self.hs = None


class NetActor(nn.Module):
    x_box = [450, 350, 250, 150, 100, 50, 25, 10, 5]; x_box = np.array(x_box + [0] + (-1 * np.array(list(reversed(copy.copy(x_box))))).tolist(), dtype=np.float32)
    y_box = [50, 15, 5]; y_box = np.array(y_box + [0] + (-1 * np.array(list(reversed(copy.copy(y_box))))).tolist(), dtype=np.float32)

    x_discretizer = SimpleDiscretizer(x_box)
    y_discretizer = SimpleDiscretizer(y_box)
    wasd_discretizer = wasd_Discretizer()

    CENTER_SZ_WH = (400, 189,)

    @staticmethod
    def preprocess(im: Union[np.ndarray, List[np.ndarray]]) -> torch.Tensor: # to('cuda')
        assert len(im[0].shape) == 3 and im[0].shape[-1] == 3, "im shape should be (n, h, w, 3)"
        
        im = im[..., ::-1].transpose((0, 3, 1, 2))  # BGR to RGB, BHWC to BCHW, (n, 3, h, w)
        # im = im.transpose((0, 3, 1, 2))  # BHWC to BCHW, (n, 3, h, w)
        im = np.ascontiguousarray(im)  # contiguous
        im = torch.from_numpy(im)

        im = im.to('cuda')
        im = im.float()  # uint8 to fp16/32
        im /= 255  # 0 - 255 to 0.0 - 1.0
        return im

    @staticmethod
    def get_center(frame):
        if not iterable_eq(frame.shape, (578, 1280, 3)):
            print(f"[SimpleNetActor] Warning: input shape is {frame.shape}, use resize")
            frame = cv2.resize(frame, (1280, 578))
        assert frame.shape[0] == 578, frame.shape[1] == 1280
        frame = crop_wh(frame, 240, 100)
        assert frame.shape[0] == 378, frame.shape[1] == 800
        frame = cv2.resize(frame, NetActor.CENTER_SZ_WH)
        return frame
    
    def __init__(self, Net):
        super(NetActor, self).__init__()
        self.net = Net(
            NetActor.CENTER_SZ_WH,
            NetActor.wasd_discretizer.n_actions,
            NetActor.x_discretizer.n_actions,
            NetActor.y_discretizer.n_actions
        )

    def act(self, frames):
        assert len(frames) == 1

        frame_centers = np.array([NetActor.get_center(f.copy()) for f in frames])
        frame_centers = NetActor.preprocess(frame_centers)
        index_wasd, index_x, index_y = self.net.act(frame_centers)
        index_wasd, index_x, index_y = int(index_wasd[0]), int(index_x[0]), int(index_y[0])
        wasd = NetActor.wasd_discretizer.index_to_action_(index_wasd)
        x = NetActor.x_discretizer.index_to_action_(index_x)
        y = NetActor.y_discretizer.index_to_action_(index_y)
        return wasd, np.array([x, y])
    
    def load_model(self, path):
        self.net = load_model(self.net, path, device='cuda')