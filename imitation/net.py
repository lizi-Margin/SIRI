import torch, copy, cv2
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
    
    def __init__(self, wasd_n_actions, x_n_actions, y_n_actions):
        super().__init__(SimpleNetActor.CENTER_SZ_WH, wasd_n_actions, x_n_actions, y_n_actions)

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
    def __init__(self):
        super(LSTMNet, self).__init__()
        # 加载预训练的EfficientNetB0
        base_model = efficientnet_b0(pretrained=True)
        # 仅使用前六个残差阶段
        features = list(base_model.features.children())[:6]
        self.features = nn.Sequential(*features)

        # 添加ConvLSTM层
        self.conv_lstm = nn.ConvLSTM2d(
            in_channels=112,
            out_channels=112,
            kernel_size=(3, 3),
            padding=1,
            batch_first=True
        )

        # 全连接层
        self.fc = nn.Linear(112 * 18 * 10, 输出类别数)

    def forward(self, x):
        # x的形状应该是(batch_size, sequence_length, channels, height, width)
        batch_size, sequence_length, channels, height, width = x.size()
        # 将序列长度和批量大小合并
        x = x.view(-1, channels, height, width)

        # 通过EfficientNetB0的前六个残差阶段
        x = self.features(x)

        # 恢复原始的批量大小和序列长度
        x = x.view(batch_size, sequence_length, 112, 18, 10)

        # 通过ConvLSTM层
        _, (x, _) = self.conv_lstm(x)
        x = x.squeeze(0)

        # 展平特征图
        x = x.view(x.size(0), -1)

        # 通过全连接层
        x = self.fc(x)

        return x