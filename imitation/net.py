import torch, copy, cv2, os, time
import numpy as np
import torch.nn as nn
import torch.nn.functional as F
from typing import Union, List
from torchvision.transforms import Compose

from torchsummary import summary
from torchvision.models import efficientnet_b0
from siri.utils.logger import lprint
from siri.vision.preprocess import crop_wh, pre_transform_crop, crop
from imitation.discretizer import SimpleDiscretizer, wasd_Discretizer
from imitation.utils import iterable_eq
from UTIL.colorful import *


x_box = [450, 380, 310, 260, 210, 170, 130, 90, 65, 40, 25, 10]; x_box = np.array(x_box + [0] + (-1 * np.array(list(reversed(copy.copy(x_box))))).tolist(), dtype=np.float32)
y_box = [100, 50, 25, 10]; y_box = np.array(y_box + [0] + (-1 * np.array(list(reversed(copy.copy(y_box))))).tolist(), dtype=np.float32)

Y_MAX=300
Y_D_MAX=200

from .transform import center_transform_train, center_transform_test

def load_model(m, pt_path, device='cuda'):
    if not os.path.exists(pt_path): 
        assert False, "file does not exists"

    cpt = torch.load(pt_path, map_location=device)
    m.load_state_dict(cpt['policy'], strict=True)
    print绿(f'loaded model {pt_path}')
    return m


# class SimpleNet(nn.Module):
#     def __init__(self, input_sz_wh, wasd_n_actions, x_n_actions, y_n_actions):
#         super(SimpleNet, self).__init__()
#         self.input_frame_shape = (3,) + tuple(reversed(input_sz_wh))
#         lprint(self, f"input frame shape: {self.input_frame_shape}, if it's changed, errors may occur")
    

#         base_model = efficientnet_b0(weights='IMAGENET1K_V1')
#         # feature_extractor = nn.Sequential(*list(base_model.features.children())[:161])
#         # summary(feature_extractor, input_size=self.input_frame_shape)

#         features = list(base_model.features.children())[:6]
#         self.features = nn.Sequential(*features)
#         t, h, w = 112, 12, 25
#         self.wasd_fc = nn.Linear(t * h * w, wasd_n_actions)
#         self.x_fc = nn.Linear(t * h * w, x_n_actions)
#         self.y_fc = nn.Linear(t * h * w, y_n_actions)

#     def forward(self, x):
#         # x的形状应该是 (batch_size, channels, height, width)
#         batch_size, channels, height, width = x.size()

#         x = self.features(x)
#         # print(x.shape)
#         x = x.view(batch_size, -1)
#         # print(x.shape)

#         logit_wasd = self.wasd_fc(x)
#         logit_x = self.x_fc(x)
#         logit_y = self.y_fc(x)

#         return logit_wasd, logit_x, logit_y
    
#     @torch.no_grad
#     def act(self, x):
#         batch_size, channels, height, width = x.size()
#         assert batch_size == 1
#         logit_wasd, logit_x, logit_y = self.forward(x)
#         return torch.argmax(logit_wasd, dim=-1), torch.argmax(logit_x, dim=-1), torch.argmax(logit_y, dim=-1)

# class SimpleNetActor(SimpleNet):
    

#     x_discretizer = SimpleDiscretizer(x_box)
#     y_discretizer = SimpleDiscretizer(y_box)
#     wasd_discretizer = wasd_Discretizer()

#     CENTER_SZ_WH = (400, 189,)

#     @staticmethod
#     def preprocess(im: Union[np.ndarray, List[np.ndarray]]) -> torch.Tensor: # to('cuda')
#         assert len(im[0].shape) == 3 and im[0].shape[-1] == 3, "im shape should be (n, h, w, 3)"
        
#         im = im[..., ::-1].transpose((0, 3, 1, 2))  # BGR to RGB, BHWC to BCHW, (n, 3, h, w)
#         # im = im.transpose((0, 3, 1, 2))  # BHWC to BCHW, (n, 3, h, w)
#         im = np.ascontiguousarray(im)  # contiguous
#         im = torch.from_numpy(im)

#         im = im.to('cuda')
#         im = im.float()  # uint8 to fp16/32
#         im /= 255  # 0 - 255 to 0.0 - 1.0
#         return im

#     @staticmethod
#     def get_center(frame):
#         if not iterable_eq(frame.shape, (578, 1280, 3)):
#             print(f"[SimpleNetActor] Warning: input shape is {frame.shape}, use resize")
#             frame = cv2.resize(frame, (1280, 578))
#         assert frame.shape[0] == 578, frame.shape[1] == 1280
#         frame = crop_wh(frame, 240, 100)
#         assert frame.shape[0] == 378, frame.shape[1] == 800
#         frame = cv2.resize(frame, SimpleNetActor.CENTER_SZ_WH)
#         return frame
    
#     def __init__(self):
#         super().__init__(
#             SimpleNetActor.CENTER_SZ_WH,
#             SimpleNetActor.wasd_discretizer.n_actions,
#             SimpleNetActor.x_discretizer.n_actions,
#             SimpleNetActor.y_discretizer.n_actions
#         )

#     def act(self, frames):
#         assert len(frames) == 1

#         frame_centers = np.array([SimpleNetActor.get_center(f.copy()) for f in frames])
#         frame_centers = SimpleNetActor.preprocess(frame_centers)
#         index_wasd, index_x, index_y = super(SimpleNetActor, self).act(frame_centers)
#         index_wasd, index_x, index_y = int(index_wasd[0]), int(index_x[0]), int(index_y[0])
#         wasd = SimpleNetActor.wasd_discretizer.index_to_action_(index_wasd)
#         x = SimpleNetActor.x_discretizer.index_to_action_(index_x)
#         y = SimpleNetActor.y_discretizer.index_to_action_(index_y)
#         return wasd, np.array([x, y])
        
        

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
    _showed = True

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
        frame_centers = NetActor.preprocess(frame_centers, train=False)
        index_wasd, index_x, index_y = self.net.act(frame_centers)
        index_wasd, index_x, index_y = int(index_wasd[0]), int(index_x[0]), int(index_y[0])
        wasd = NetActor.wasd_discretizer.index_to_action_(index_wasd)
        x = NetActor.x_discretizer.index_to_action_(index_x)
        y = NetActor.y_discretizer.index_to_action_(index_y)
        return wasd, np.array([x, y])
    
    def load_model(self, path):
        self.net = load_model(self.net, path, device='cuda')





class DVNetBase(nn.Module):
    _showed = False


    x_discretizer = SimpleDiscretizer(x_box)
    y_discretizer = SimpleDiscretizer(y_box, MAX=Y_MAX, D_MAX=Y_D_MAX)
    wasd_discretizer = wasd_Discretizer()

    CENTER_SZ_WH = (400, 189,)

    @staticmethod
    def get_center(frame):
        if not iterable_eq(frame.shape, (578, 1280, 3)):
            print(f"[SimpleNetActor] Warning: input shape is {frame.shape}, use resize")
            frame = cv2.resize(frame, (1280, 578))
        assert frame.shape[0] == 578, frame.shape[1] == 1280
        frame = crop_wh(frame, 240, 100)
        assert frame.shape[0] == 378, frame.shape[1] == 800
        frame = cv2.resize(frame, DVNet.CENTER_SZ_WH)
        return frame

    from midas.transforms import Resize, NormalizeImage, PrepareForNet
    midas_transform = Compose(
        [
            lambda img: {"image": img / 255.0},
            Resize(
                256,
                256,
                resize_target=None,
                keep_aspect_ratio=False,
                ensure_multiple_of=32,
                resize_method="minimal",
                image_interpolation_method=cv2.INTER_CUBIC,
            ),
            NormalizeImage(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5]),
            PrepareForNet(),
            lambda sample: torch.from_numpy(sample["image"]),
        ]
    )


    @classmethod
    def preprocess(cls, im: Union[np.ndarray, List[np.ndarray]], train=True) -> torch.Tensor: # to('cuda')
        if isinstance(im, np.ndarray):
            assert len(im.shape) == 4 and im.shape[-1] == 3, "im shape should be (n, h, w, 3)"
        elif isinstance(im, list):
            assert len(im[0].shape) == 3 and im[0].shape[-1] == 3, "im shape should be (n, h, w, 3)"
            im = np.stack(im)

        
        def midas_trans(image):
            image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            image = crop(image)
            image = cls.midas_transform(image)
            assert image.shape[0] == 3
            return image
        midas = torch.stack([midas_trans(oneimg) for oneimg in im]) 
        midas.to('cuda').float()
    

        def trans(image):
            image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            transform = center_transform_train if train else center_transform_test
            image = transform(image)
            assert image.shape[0] == 3
            return image
        im = torch.stack([trans(img) for img in im])
        im = im.to('cuda').float()

        if not cls._showed:
            for i in range(5):
                im0 = ((im[i].cpu().permute(1, 2, 0).numpy() + 1)/2 * 255).astype(np.uint8)[..., ::-1]
                midasim0 = ((midas[i].cpu().permute(1, 2, 0).numpy() + 1)/2 * 255).astype(np.uint8)[..., ::-1]
                print(im0.shape)
                cv2.imshow('raw', im0)
                cv2.imshow('midas', midasim0)
                cv2.waitKey(0)
                cv2.destroyAllWindows()
            cls._showed = True
        return im, midas

    
    @torch.no_grad
    def act_(self, x):
        seq_len, channels, height, width = x[0].size()
        assert seq_len == 1
        logit_wasd, logit_x, logit_y = self.forward(x, train=False)
        return torch.argmax(logit_wasd, dim=-1), torch.argmax(logit_x, dim=-1), torch.argmax(logit_y, dim=-1)
    
    def reset(self): self.hs = None

    def act(self, frames):
        assert len(frames) == 1

        frame_centers = np.array([self.get_center(f.copy()) for f in frames])
        frame_centers = self.preprocess(frame_centers,train=False)
        index_wasd, index_x, index_y = self.act_(frame_centers)
        index_wasd, index_x, index_y = int(index_wasd[0]), int(index_x[0]), int(index_y[0])
        wasd = self.wasd_discretizer.index_to_action_(index_wasd)
        x = self.x_discretizer.index_to_action_(index_x)
        y = self.y_discretizer.index_to_action_(index_y)
        return wasd, np.array([x, y])
    
    def load_model(self, path):
        self.net = load_model(self.net, path, device='cuda')


class DVNet(DVNetBase):
    def __init__(self, input_sz_wh, wasd_n_actions, x_n_actions, y_n_actions):
        super(DVNet, self).__init__()
        self.input_frame_shape = (3,) + tuple(reversed(input_sz_wh))
        lprint(self, f"input frame shape: {self.input_frame_shape}, if it's changed, errors may occur")
    
        from midas.midas_net_custom import MidasNet_small
        local_model_path = './midas_v21_small_256.pt'
        assert os.path.exists(local_model_path)
        midas_model = MidasNet_small(path=local_model_path, features=64, backbone="efficientnet_lite3", exportable=True, non_negative=True, blocks={'expand': True})
        midas_model.eval()
        self.midas_pretrained = midas_model.pretrained
        self.midas_scratch = midas_model.scratch
        

        base_model = efficientnet_b0(weights='IMAGENET1K_V1')
        features = list(base_model.features.children())[:6]
        self.features = nn.Sequential(*features)



        t, h, w = 112, 12, 25
        self.features_hw =(h, w)

        self.depth_conv = nn.Conv2d(in_channels=64, out_channels=t, kernel_size=3, stride=1, padding=1)
        
        

        from .conv_lstm import ConvLSTM
        self.conv_lstm = ConvLSTM(
            input_dim=2*t,
            hidden_dim=2*t,
            kernel_size=(3, 3),
            num_layers=1,
            batch_first=True
        )

        self.wasd_fc = nn.Linear(2*t * h * w, wasd_n_actions)
        self.x_fc = nn.Linear(2*t * h * w, x_n_actions)
        self.y_fc = nn.Linear(2*t * h * w, y_n_actions)

        self.hs = None

    def forward(self, x, train=True):
        x, midas_in = x
        seq_len, channels, height, width = x.size()
        
        with torch.no_grad():
            layer_1 = self.midas_pretrained.layer1(midas_in)
            layer_2 = self.midas_pretrained.layer2(layer_1)
            layer_3 = self.midas_pretrained.layer3(layer_2)
            layer_4 = self.midas_pretrained.layer4(layer_3)
            
            layer_1_rn = self.midas_scratch.layer1_rn(layer_1)
            layer_2_rn = self.midas_scratch.layer2_rn(layer_2)
            layer_3_rn = self.midas_scratch.layer3_rn(layer_3)
            layer_4_rn = self.midas_scratch.layer4_rn(layer_4)

            path_4 = self.midas_scratch.refinenet4(layer_4_rn)
            path_3 = self.midas_scratch.refinenet3(path_4, layer_3_rn)
            path_2 = self.midas_scratch.refinenet2(path_3, layer_2_rn)
            path_1 = self.midas_scratch.refinenet1(path_2, layer_1_rn)

        features = self.features(x) 

        d = self.depth_conv(path_1)
        d = F.adaptive_avg_pool2d(d, output_size=self.features_hw)

        # print(f"features shape: {features.shape}")  # (batch_size, 112, 12, 25)
        # print(f"path_1 shape: {d.shape}")      # 可能是 (batch_size, C, H, W)

        combined = torch.cat((features, d), dim=1).unsqueeze(1)

        o, hs = self.conv_lstm(combined, hidden_state=None if train else self.hs)
        if not train: self.hs = hs
        o = o[0].squeeze(0)
        print(o.shape)
        assert (o.shape[0] == seq_len)
        o = o.reshape(o.shape[0], -1)

        logit_wasd = self.wasd_fc(o)
        logit_x = self.x_fc(o)
        logit_y = self.y_fc(o)

        return logit_wasd, logit_x, logit_y

class DVNet2(DVNetBase):
    def __init__(self, input_sz_wh, wasd_n_actions, x_n_actions, y_n_actions):
        super(DVNet2, self).__init__()
        self.input_frame_shape = (3,) + tuple(reversed(input_sz_wh))
        lprint(self, f"input frame shape: {self.input_frame_shape}, if it's changed, errors may occur")
    
        from midas.midas_net_custom import MidasNet_small
        local_model_path = './midas_v21_small_256.pt'
        assert os.path.exists(local_model_path)
        midas_model = MidasNet_small(path=local_model_path, features=64, backbone="efficientnet_lite3", exportable=True, non_negative=True, blocks={'expand': True})
        midas_model.eval()
        self.midas_pretrained = midas_model.pretrained
        self.midas_scratch = midas_model.scratch
        

        base_model = efficientnet_b0(weights='IMAGENET1K_V1')
        features = list(base_model.features.children())[:6]
        self.features = nn.Sequential(*features)



        t, h, w = 112, 12, 25
        self.features_hw =(h, w)
        self.depth_t = 2

        self.depth_conv = nn.Conv2d(in_channels=64, out_channels=self.depth_t, kernel_size=3, stride=1, padding=1)
        
        

        lstm_t = t + self.depth_t
        from .conv_lstm import ConvLSTM
        self.conv_lstm = ConvLSTM(
            input_dim=lstm_t,
            hidden_dim=lstm_t,
            kernel_size=(3, 3),
            num_layers=1,
            batch_first=True
        )

        self.wasd_fc = nn.Linear(lstm_t * h * w, wasd_n_actions)
        self.x_fc = nn.Linear(lstm_t * h * w, x_n_actions)
        self.y_fc = nn.Linear(lstm_t * h * w, y_n_actions)

        self.hs = None

    def forward(self, x, train=True):
        x, midas_in = x
        seq_len, channels, height, width = x.size()
        
        with torch.no_grad():
            layer_1 = self.midas_pretrained.layer1(midas_in)
            layer_2 = self.midas_pretrained.layer2(layer_1)
            layer_3 = self.midas_pretrained.layer3(layer_2)
            layer_4 = self.midas_pretrained.layer4(layer_3)
            
            layer_1_rn = self.midas_scratch.layer1_rn(layer_1)
            layer_2_rn = self.midas_scratch.layer2_rn(layer_2)
            layer_3_rn = self.midas_scratch.layer3_rn(layer_3)
            layer_4_rn = self.midas_scratch.layer4_rn(layer_4)

            path_4 = self.midas_scratch.refinenet4(layer_4_rn)
            path_3 = self.midas_scratch.refinenet3(path_4, layer_3_rn)
            path_2 = self.midas_scratch.refinenet2(path_3, layer_2_rn)
            path_1 = self.midas_scratch.refinenet1(path_2, layer_1_rn)

        features = self.features(x) 

        d = self.depth_conv(path_1)
        d = F.adaptive_avg_pool2d(d, output_size=self.features_hw)

        # print(f"features shape: {features.shape}")  # (batch_size, 112, 12, 25)
        # print(f"path_1 shape: {d.shape}")      # 可能是 (batch_size, C, H, W)

        combined = torch.cat((features, d), dim=1).unsqueeze(1)

        o, hs = self.conv_lstm(combined, hidden_state=None if train else self.hs)
        if not train: self.hs = hs
        o = o[0].squeeze(0)
        print(o.shape)
        assert (o.shape[0] == seq_len)
        o = o.reshape(o.shape[0], -1)

        logit_wasd = self.wasd_fc(o)
        logit_x = self.x_fc(o)
        logit_y = self.y_fc(o)

        return logit_wasd, logit_x, logit_y


class DVNet3(DVNetBase):
    def __init__(self, input_sz_wh, wasd_n_actions, x_n_actions, y_n_actions):
        super(DVNet3, self).__init__()
        self.input_frame_shape = (3,) + tuple(reversed(input_sz_wh))
        lprint(self, f"input frame shape: {self.input_frame_shape}, if it's changed, errors may occur")
    
        from midas.midas_net_custom import MidasNet_small
        local_model_path = './midas_v21_small_256.pt'
        assert os.path.exists(local_model_path)
        midas_model = MidasNet_small(path=local_model_path, features=64, backbone="efficientnet_lite3", exportable=True, non_negative=True, blocks={'expand': True})
        midas_model.eval()
        self.midas_pretrained = midas_model.pretrained
        self.midas_scratch = midas_model.scratch
        
        base_model1 = efficientnet_b0(weights='IMAGENET1K_V1')
        base_model2 = efficientnet_b0(weights='IMAGENET1K_V1')
        features1 = list(base_model1.features.children())[:6]
        features2 = list(base_model2.features.children())[:6]
        self.raw_features = nn.Sequential(*features1)
        self.deep_features = nn.Sequential(*features2)

        raw_feat_t, raw_feat_h, raw_feat_w = 112, 12, 25
        deep_feat_t, deep_feat_h, deep_feat_w = 112, 12, 25

        from .conv_lstm import ConvLSTM
        self.raw_conv_lstm = ConvLSTM(
            input_dim=raw_feat_t,
            hidden_dim=raw_feat_t,
            kernel_size=(3, 3),
            num_layers=1,
            batch_first=True
        )
        self.deep_conv_lstm = ConvLSTM(
            input_dim=deep_feat_t,
            hidden_dim=deep_feat_t,
            kernel_size=(3, 3),
            num_layers=1,
            batch_first=True
        )
        raw_ = (raw_feat_t * raw_feat_h * raw_feat_w)
        deep_ = (deep_feat_t * deep_feat_h * deep_feat_w)
        all_ = raw_ + deep_
        self.wasd_fc = nn.Linear(all_ , wasd_n_actions)
        self.x_fc = nn.Linear(all_, x_n_actions)
        self.y_fc = nn.Linear(all_, y_n_actions)

        self.hs = None

    def forward(self, x, train=True):
        x, midas_in = x
        seq_len, channels, height, width = x.size()
        
        with torch.no_grad():
            layer_1 = self.midas_pretrained.layer1(midas_in)
            layer_2 = self.midas_pretrained.layer2(layer_1)
            layer_3 = self.midas_pretrained.layer3(layer_2)
            layer_4 = self.midas_pretrained.layer4(layer_3)
            
            layer_1_rn = self.midas_scratch.layer1_rn(layer_1)
            layer_2_rn = self.midas_scratch.layer2_rn(layer_2)
            layer_3_rn = self.midas_scratch.layer3_rn(layer_3)
            layer_4_rn = self.midas_scratch.layer4_rn(layer_4)

            path_4 = self.midas_scratch.refinenet4(layer_4_rn)
            path_3 = self.midas_scratch.refinenet3(path_4, layer_3_rn)
            path_2 = self.midas_scratch.refinenet2(path_3, layer_2_rn)
            path_1 = self.midas_scratch.refinenet1(path_2, layer_1_rn)
            midas_out = self.midas_scratch.output_conv(path_1)
            midas_out = F.interpolate(midas_out, size=(height, width), mode="bilinear", align_corners=False)
            midas_out = midas_out.expand(-1, 3, -1, -1)  # 仅在需要时使用
            midas_min = midas_out.min()
            midas_max = midas_out.max()
            midas_out = 2 * (midas_out - midas_min) / (midas_max - midas_min) - 1

        # print(x.shape)
        # print(midas_out.shape)
        raw_feat = self.raw_features(x).unsqueeze(1)
        deep_feat = self.deep_features(midas_out).unsqueeze(1)

        raw_o, raw_hs = self.raw_conv_lstm(raw_feat, hidden_state=None if train else self.hs[0])
        deep_o, deep_hs = self.deep_conv_lstm(deep_feat, hidden_state=None if train else self.hs[1])
        if not train: self.hs = (raw_hs, deep_hs,)
        raw_o = raw_o[0].squeeze(0)
        deep_o = deep_o[0].squeeze(0)
        raw_o = raw_o.reshape(raw_o.shape[0], -1)
        deep_o = deep_o.reshape(deep_o.shape[0], -1)

        combined = torch.cat((raw_o, deep_o), dim=1).unsqueeze(1)

        

        logit_wasd = self.wasd_fc(combined)
        logit_x = self.x_fc(combined)
        logit_y = self.y_fc(combined)

        return logit_wasd, logit_x, logit_y


class DVNet4(DVNetBase):
    def __init__(self, input_sz_wh, wasd_n_actions, x_n_actions, y_n_actions):
        super(DVNet4, self).__init__()
        self.input_frame_shape = (3,) + tuple(reversed(input_sz_wh))
        lprint(self, f"input frame shape: {self.input_frame_shape}, if it's changed, errors may occur")
    
        from midas.midas_net_custom import MidasNet_small
        local_model_path = './midas_v21_small_256.pt'
        assert os.path.exists(local_model_path)
        midas_model = MidasNet_small(path=local_model_path, features=64, backbone="efficientnet_lite3", exportable=True, non_negative=True, blocks={'expand': True})
        midas_model.eval()
        self.midas_pretrained = midas_model.pretrained
        self.midas_scratch = midas_model.scratch
        

        base_model2 = efficientnet_b0(weights='IMAGENET1K_V1')
        features2 = list(base_model2.features.children())[:6]
        self.deep_features = nn.Sequential(*features2)

        deep_feat_t, deep_feat_h, deep_feat_w = 112, 12, 25

        from .conv_lstm import ConvLSTM
        self.deep_conv_lstm = ConvLSTM(
            input_dim=deep_feat_t,
            hidden_dim=deep_feat_t,
            kernel_size=(3, 3),
            num_layers=1,
            batch_first=True
        )
        deep_ = (deep_feat_t * deep_feat_h * deep_feat_w)
        all_ = deep_
        self.wasd_fc = nn.Linear(all_ , wasd_n_actions)
        self.x_fc = nn.Linear(all_, x_n_actions)
        self.y_fc = nn.Linear(all_, y_n_actions)

        self.hs = None

    def forward(self, x, train=True):
        x, midas_in = x
        seq_len, channels, height, width = x.size()
        
        with torch.no_grad():
            layer_1 = self.midas_pretrained.layer1(midas_in)
            layer_2 = self.midas_pretrained.layer2(layer_1)
            layer_3 = self.midas_pretrained.layer3(layer_2)
            layer_4 = self.midas_pretrained.layer4(layer_3)
            
            layer_1_rn = self.midas_scratch.layer1_rn(layer_1)
            layer_2_rn = self.midas_scratch.layer2_rn(layer_2)
            layer_3_rn = self.midas_scratch.layer3_rn(layer_3)
            layer_4_rn = self.midas_scratch.layer4_rn(layer_4)

            path_4 = self.midas_scratch.refinenet4(layer_4_rn)
            path_3 = self.midas_scratch.refinenet3(path_4, layer_3_rn)
            path_2 = self.midas_scratch.refinenet2(path_3, layer_2_rn)
            path_1 = self.midas_scratch.refinenet1(path_2, layer_1_rn)
            midas_out = self.midas_scratch.output_conv(path_1)
            midas_out = F.interpolate(midas_out, size=(height, width), mode="bilinear", align_corners=False)
            midas_out = midas_out.expand(-1, 3, -1, -1)  # 仅在需要时使用


        # print(x.shape)
        # print(midas_out.shape)
        deep_feat = self.deep_features(midas_out).unsqueeze(1)

        deep_o, deep_hs = self.deep_conv_lstm(deep_feat, hidden_state=None if train else self.hs)
        if not train: self.hs = deep_hs
        deep_o = deep_o[0].squeeze(0)
        deep_o = deep_o.reshape(deep_o.shape[0], -1)

        combined = deep_o.unsqueeze(1)

        logit_wasd = self.wasd_fc(combined)
        logit_x = self.x_fc(combined)
        logit_y = self.y_fc(combined)

        return logit_wasd, logit_x, logit_y