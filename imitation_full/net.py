import torch, copy, cv2, os, time
import numpy as np
import torch.nn as nn
import torch.nn.functional as F
from typing import Union, List
from torchvision.transforms import Compose

from torchsummary import summary
from torchvision.models import efficientnet_b0, efficientnet_b5
from siri.utils.logger import lprint
from siri.vision.preprocess import crop_wh, pre_transform_crop, crop
from imitation.discretizer import SimpleDiscretizer, wasd_Discretizer
from imitation.utils import iterable_eq
from UTIL.colorful import *


x_box = [225, 195, 155, 130, 105, 85, 65, 45, 32.5, 20, 12.5, 5]; x_box = np.array(x_box + [0] + (-1 * np.array(list(reversed(copy.copy(x_box))))).tolist(), dtype=np.float32)
y_box = [50, 25, 12.5, 5]; y_box = np.array(y_box + [0] + (-1 * np.array(list(reversed(copy.copy(y_box))))).tolist(), dtype=np.float32)

Y_MAX=300
Y_D_MAX=200

from imitation.transform import center_transform_train, center_transform_test

def load_model(m, pt_path, device='cuda'):
    if not os.path.exists(pt_path): 
        assert False, "file does not exists"

    cpt = torch.load(pt_path, map_location=device)
    m.load_state_dict(cpt['policy'], strict=True)
    print绿(f'loaded model {pt_path}')
    return m

class NetActor(nn.Module):
    _showed = False

    x_discretizer = SimpleDiscretizer(x_box)
    y_discretizer = SimpleDiscretizer(y_box, MAX=Y_MAX, D_MAX=Y_D_MAX)
    wasd_discretizer = wasd_Discretizer()

    CENTER_SZ_WH = (400, 189,)

    @classmethod
    def preprocess(cls, im: Union[np.ndarray, List[np.ndarray]], train=True) -> torch.Tensor: # to('cuda')
        assert len(im[0].shape) == 3 and im[0].shape[-1] == 3, "im shape should be (n, h, w, 3)"

        def trans(image):
            image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            transform = center_transform_train if train else center_transform_test
            image = transform(image)
            assert image.shape[0] == 3
            return image
        im = [trans(img) for img in im]
        im = torch.stack(im).to('cuda').float()
        if not cls._showed:
            for i in range(5):
                im0 = ((im[i].cpu().permute(1, 2, 0).numpy() + 1)/2 * 255).astype(np.uint8)[..., ::-1]
                print(im0.shape)
                cv2.imshow('raw', im0)
                cv2.waitKey(0)
                cv2.destroyAllWindows()
            cls._showed = True
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
    
    def __init__(self):
        super(NetActor, self).__init__()


    def act(self, frames):
        assert len(frames) == 1

        frame_centers = np.array([self.get_center(f.copy()) for f in frames])
        frame_centers = self.preprocess(frame_centers, train=False)
        index = self._act(frame_centers)
        index = (int(x[0]) for x in index)
        index_wasd, index_x, index_y = index[0], index[1], index[2]
        wasd = self.wasd_discretizer.index_to_action_(index_wasd)
        x = self.x_discretizer.index_to_action_(index_x)
        y = self.y_discretizer.index_to_action_(index_y)
        return wasd, np.array([x, y])
    
    def load_model(self, path):
        self.net = load_model(self.net, path, device='cuda')




class LSTMNet(NetActor):
    def __init__(self):
        super(LSTMNet, self).__init__()
        input_sz_wh = self.CENTER_SZ_WH
        wasd_n_actions = self.wasd_discretizer.n_actions
        x_n_actions = self.x_discretizer.n_actions
        y_n_actions = self.y_discretizer.n_actions


        self.input_frame_shape = (3,) + tuple(reversed(input_sz_wh))
        lprint(self, f"input frame shape: {self.input_frame_shape}, if it's changed, errors may occur")
    

        base_model = efficientnet_b5(weights='IMAGENET1K_V1')

        features = list(base_model.features.children())[:6]
        self.features = nn.Sequential(*features)
        t, h, w = 176, 12, 25
        from imitation.conv_lstm import ConvLSTM
        self.conv_lstm = ConvLSTM(
            input_dim=t,
            hidden_dim=t,
            kernel_size=(3, 3),
            num_layers=1,
            batch_first=True
        )

        self.wasd_fc = nn.Linear(t * h * w, wasd_n_actions)
        self.x_fc = nn.Linear(t * h * w, x_n_actions)
        self.y_fc = nn.Linear(t * h * w, y_n_actions)
        self.jump_fc = nn.Linear(t * h * w, 1)
        self.crouch_fc = nn.Linear(t * h * w, 1)
        self.reload_fc = nn.Linear(t * h * w, 1)

        self.r_fc = nn.Linear(t * h * w, 1)
        self.l_fc = nn.Linear(t * h * w, 1)

        self.hs = None

    def forward(self, x, train=True):
        seq_len, channels, height, width = x.size()

        x = self.features(x)
        x = x.unsqueeze(0)
        x, hs = self.conv_lstm.forward(x, hidden_state=None if train else self.hs)
        if not train: self.hs = hs
        x = x[0].squeeze(0)
        # print(x.shape)
        x = x.view(seq_len, -1)
        # print(x.shape)

        logit_wasd = self.wasd_fc(x)
        logit_x = self.x_fc(x)
        logit_y = self.y_fc(x)

        logit_jump = self.jump_fc(x)
        logit_crouch = self.crouch_fc(x)
        logit_reload = self.reload_fc(x)

        logit_r = self.r_fc(x)
        logit_l = self.l_fc(x)

        return (
            logit_wasd,
            logit_x,
            logit_y,
            logit_jump,
            logit_crouch,
            logit_reload,
            logit_r,
            logit_l
        )
    
    @torch.no_grad
    def _act(self, x):
        seq_len, channels, height, width = x.size()
        assert seq_len == 1
        logit = self.forward(x, train=False)
        (
            logit_wasd,
            logit_x,
            logit_y,
            logit_jump,
            logit_crouch,
            logit_reload,
            logit_r,
            logit_l
        ) = logit

        return (
            torch.argmax(logit_wasd, dim=-1),
            torch.argmax(logit_x, dim=-1),
            torch.argmax(logit_y, dim=-1),

            torch.sigmoid(logit_jump) > 0.5,
            torch.sigmoid(logit_crouch) > 0.5,
            torch.sigmoid(logit_reload) > 0.5,
            torch.sigmoid(logit_r) > 0.5,
            torch.sigmoid(logit_l) > 0.5,
        )
    
    def reset(self): self.hs = None





