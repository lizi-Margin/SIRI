import torch, copy, cv2, os, time
import numpy as np
import torch.nn as nn
import torch.nn.functional as F
from typing import Union, List
from torchvision.transforms import Compose

from imitation.conv_lstm import ConvLSTM
from torchsummary import summary
from torchvision.models import efficientnet_b0, efficientnet_b5
from siri.utils.logger import lprint
from siri.vision.preprocess import crop_wh, pre_transform_crop, crop
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

class NetActor(nn.Module):
    _showed = not AlgorithmConfig.show_preprocessed_preview

    x_discretizer = SimpleDiscretizer(x_box)
    y_discretizer = SimpleDiscretizer(y_box, MAX=Y_MAX, D_MAX=Y_D_MAX)
    wasd_discretizer = wasd_Discretizer()

    CENTER_SZ_WH = (400, 189,)

    # @classmethod
    # def preprocess(cls, im: Union[np.ndarray, List[np.ndarray]], train=True) -> torch.Tensor: # to('cuda')
    #     assert len(im[0].shape) == 3 and im[0].shape[-1] == 3, "im shape should be (n, h, w, 3)"
    #     def trans(image):
    #         image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    #         transform = center_transform_train if train else center_transform_test
    #         image = transform(image)
    #         assert image.shape[0] == 3
    #         return image
    #     im = [trans(img) for img in im]
    #     # if train:
    #     #     im = torch.stack(im).to('cpu').float()
    #     # else:
    #     #     im = torch.stack(im).to('cuda').float()
    #     im = torch.stack(im).to('cuda').float()
    #     if not cls._showed:
    #         for i in range(5):
    #             im0 = ((im[i].cpu().permute(1, 2, 0).numpy() + 1)/2 * 255).astype(np.uint8)[..., ::-1]
    #             print(im0.shape)
    #             cv2.imshow('raw', im0)
    #             cv2.waitKey(0)
    #             cv2.destroyAllWindows()
    #         cls._showed = True
    #     return im

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
            image = cv2.resize(image, (256, 256))
            # image = crop(image)
            image = cls.midas_transform(image)
            assert image.shape[0] == 3
            return image
        midas = torch.stack([midas_trans(oneimg) for oneimg in im]) 
        midas = midas.to('cuda').float()
    

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
        input_sz_wh = self.CENTER_SZ_WH
        self.input_frame_shape = (3,) + tuple(reversed(input_sz_wh))
        lprint(self, f"input frame shape: {self.input_frame_shape}, if it's changed, errors may occur")
    
        self.hs = None

    def reset(self): self.hs = None




    def act(self, frames):
        assert len(frames) == 1

        frame_centers = np.array([self.get_center(f.copy()) for f in frames])
        frame_centers = self.preprocess(frame_centers, train=False)
        index = self._act(frame_centers)
        index = tuple(int(x[0]) for x in index)
        index_wasd, index_x, index_y = index[0], index[1], index[2]
        wasd = self.wasd_discretizer.index_to_action_(index_wasd)
        x = self.x_discretizer.index_to_action_(index_x)
        y = self.y_discretizer.index_to_action_(index_y)
        return wasd, np.array([x, y])
    
    def load_model(self, path):
        load_model(self, path, device='cuda')

    @torch.no_grad()
    def _act(self, x):
        seq_len, channels, height, width = x[0].size()
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
    

    def init_act_head(self, thw):
        wasd_n_actions = self.wasd_discretizer.n_actions
        x_n_actions = self.x_discretizer.n_actions
        y_n_actions = self.y_discretizer.n_actions
        self.fc_layers = nn.ModuleDict({
            "wasd": nn.Linear(thw, wasd_n_actions),
            "x": nn.Linear(thw, x_n_actions),
            "y": nn.Linear(thw, y_n_actions),
            "jump": nn.Linear(thw, 1),
            "crouch": nn.Linear(thw, 1),
            "reload": nn.Linear(thw, 1),
            "r": nn.Linear(thw, 1),
            "l": nn.Linear(thw, 1),
        })
    
    @staticmethod
    def get_efficientnet_b5():
        base_model = efficientnet_b5(weights='IMAGENET1K_V1')
        features = list(base_model.features.children())[:6]
        features = nn.Sequential(*features)
        t, h, w = 176, 12, 25
        return features, t, h, w

class LSTMB5(NetActor):
    def __init__(self):
        super(LSTMB5, self).__init__()
        self.features, t, h, w = self.get_efficientnet_b5() 

        self.conv_lstm = ConvLSTM(
            input_dim=t,
            hidden_dim=t,
            kernel_size=(3, 3),
            num_layers=1,
            batch_first=True
        )

        self.init_act_head(t * h * w)

        

    def forward(self, x, train=True):
        x = x[0]
        seq_len, channels, height, width = x.size()

        x = self.features(x)
        x = x.unsqueeze(0)
        x, hs = self.conv_lstm.forward(x, hidden_state=None if train else self.hs)
        if not train: self.hs = hs
        x = x[0].squeeze(0)
        # print(x.shape)
        x = x.view(seq_len, -1)
        # print(x.shape)

        return tuple(self.fc_layers[name](x) for name in self.fc_layers)

    
class DVNetBase(NetActor):
    def init_midas(self):
        from midas.midas_net_custom import MidasNet_small
        local_model_path = './midas_v21_small_256.pt'
        assert os.path.exists(local_model_path)
        midas_model = MidasNet_small(path=local_model_path, features=64, backbone="efficientnet_lite3", exportable=True, non_negative=True, blocks={'expand': True})
        midas_model.eval()
        self.midas_pretrained = midas_model.pretrained
        self.midas_scratch = midas_model.scratch

    def _process_midas(self, midas_in):
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
        return path_1

class DVNet2(DVNetBase):
    def __init__(self):
        super(LSTMB5, self).__init__()
        self.init_midas()
        self.features, t, h, w = self.get_efficientnet_b5() 

        self.features_hw =(h, w)
        self.depth_t = 32
        lstm_t = t + self.depth_t
        self.depth_conv = nn.Conv2d(in_channels=64, out_channels=self.depth_t, kernel_size=3, stride=1, padding=1)


        self.conv_lstm = ConvLSTM(
            input_dim=lstm_t,
            hidden_dim=lstm_t,
            kernel_size=(3, 3),
            num_layers=1,
            batch_first=True
        )


        self.init_act_head(lstm_t * h * w)


    def forward(self, x, train=True):
        x, midas_in = x
        seq_len, channels, height, width = x.size()
        
        path_1 = self._process_midas(midas_in)        

        features = self.features(x) 

        d = self.depth_conv(path_1)
        d = F.adaptive_avg_pool2d(d, output_size=self.features_hw)
        combined = torch.cat((features, d), dim=1).unsqueeze(1)

        o, hs = self.conv_lstm(combined, hidden_state=None if train else self.hs)
        if not train: self.hs = hs
        o = o[0].squeeze(0)
        print(o.shape)
        assert (o.shape[0] == seq_len)
        o = o.reshape(o.shape[0], -1)

        return tuple(self.fc_layers[name](o) for name in self.fc_layers)

class DVNet_SAF(DVNetBase):
    def __init__(self):
        super(DVNet_SAF, self).__init__()
        self.init_midas()
        self.features, t, h, w = self.get_efficientnet_b5() 

        self.features_hw = (h, w)
        self.depth_t = 32
        lstm_t = t + self.depth_t
        self.depth_conv = nn.Conv2d(in_channels=64, out_channels=self.depth_t, kernel_size=3, stride=1, padding=1)

        # 添加空间注意力模块
        self.spatial_attention = nn.Sequential(
            nn.Conv2d(self.depth_t, 1, kernel_size=1),
            nn.Sigmoid()
        )

        
        self.conv_lstm = ConvLSTM(input_dim=lstm_t, hidden_dim=lstm_t, kernel_size=(3, 3), num_layers=1, batch_first=True)

        self.init_act_head(lstm_t * h * w)

    def forward(self, x, train=True):
        x, midas_in = x

        path_1 = self._process_midas(midas_in)

        features = self.features(x)

        d = self.depth_conv(path_1)
        d = F.adaptive_avg_pool2d(d, output_size=self.features_hw)

        # 空间注意力加权
        attention_map = self.spatial_attention(d)
        d = d * attention_map  # 加权深度特征

        combined = torch.cat((features, d), dim=1).unsqueeze(1)
        o, hs = self.conv_lstm(combined, hidden_state=None if train else self.hs)
        if not train: self.hs = hs
        o = o[0].squeeze(0)
        o = o.reshape(o.shape[0], -1)

        return tuple(self.fc_layers[name](o) for name in self.fc_layers)



class DVNet_MSFF(DVNet_SAF):
    def __init__(self):
        super(DVNet_MSFF, self).__init__()
        
        # 替换 self.depth_conv 为多尺度融合
        self.depth_conv = None
        self.msff_conv1 = nn.Conv2d(in_channels=64, out_channels=16, kernel_size=1)
        self.msff_conv3 = nn.Conv2d(in_channels=64, out_channels=16, kernel_size=3, padding=1)
        self.msff_conv5 = nn.Conv2d(in_channels=64, out_channels=16, kernel_size=5, padding=2)
        self.msff_pool = nn.AdaptiveAvgPool2d((12, 25))

    def forward(self, x, train=True):
        x, midas_in = x
        seq_len, channels, height, width = x.size()

        path_1 = self._process_midas(midas_in)

        features = self.features(x)

        d1 = self.msff_conv1(path_1)
        d3 = self.msff_conv3(path_1)
        d5 = self.msff_conv5(path_1)
        dp = self.msff_pool(path_1)

        d = torch.cat([d1, d3, d5, dp], dim=1)

        combined = torch.cat((features, d), dim=1).unsqueeze(1)
        o, hs = self.conv_lstm(combined, hidden_state=None if train else self.hs)
        if not train: self.hs = hs
        o = o[0].squeeze(0)
        o = o.reshape(o.shape[0], -1)

        return tuple(self.fc_layers[name](o) for name in self.fc_layers)



# class DVNet_Transformer(DVNetBase):
#     def __init__(self):
#         super(DVNet_Transformer, self).__init__()
#         self.init_midas()
#         self.features, t, h, w = self.get_efficientnet_b5() 

#         self.features_hw = (h, w)  # (12, 25)
#         self.depth_t = 32
#         self.depth_conv = nn.Conv2d(in_channels=64, out_channels=self.depth_t, kernel_size=3, stride=1, padding=1)
#         fused_t = t + self.depth_t
#         token_dim = fused_t * h * w

#         from torch.nn import TransformerEncoder, TransformerEncoderLayer
#         encoder_layer = TransformerEncoderLayer(d_model=token_dim, nhead=1, batch_first=True)
#         self.transformer = TransformerEncoder(encoder_layer, num_layers=1)

        
#         self.init_act_head(token_dim)

#     def forward(self, x, train=True):
#         x, midas_in = x  # x: (seq_len, C, H, W)
#         seq_len, channels, height, width = x.size()  # batch_size = seq_len

#         path_1 = self._process_midas(midas_in)  # (seq_len, 64, H, W)

#         features = self.features(x)  # (seq_len, t, h, w)
#         d = self.depth_conv(path_1)  # (seq_len, 32, H, W)
#         d = F.interpolate(d, size=self.features_hw, mode='bilinear', align_corners=False)  # (seq_len, 32, h, 2)

#         combined = torch.cat((features, d), dim=1).view(seq_len, -1)  # (seq_len, token_dim)
#         combined = combined.unsqueeze(0)  # (1, seq_len, token_dim)

#         if train:
#             fused = self.transformer(combined)  
#         else:
#             if self.hs is None:
#                 self.hs = torch.zeros((1, 4, combined.shape[1]), device=combined.device)

#             memory = torch.cat((self.hs, combined), dim=1) 
#             fused = self.transformer(memory)  # (1, 5, token_dim)

#             self.hs = memory[:, -4:, :]  # 更新历史状态，保留最近 4 帧
#             fused = fused[:, -1:]  # 1 帧

#         fused = fused.squeeze(0)  # (seq_len, token_dim)

#         return tuple(self.fc_layers[name](fused) for name in self.fc_layers)



class DVNet_CA(DVNetBase):
    def __init__(self):
        super(DVNet_CA, self).__init__()
        self.init_midas()
        self.features, t, h, w = self.get_efficientnet_b5() 

        self.features_hw = (h, w)  # (12, 25)
        self.depth_t = 64
        fused_t = t 
        from .fusion import CrossAttention, SpatialAttention
        self.cross_attention = CrossAttention(t, self.depth_t)
        # self.spatial_attention = SpatialAttention(t, self.depth_t, fused_t)

        self.conv_lstm = ConvLSTM(
            input_dim=fused_t,
            hidden_dim=fused_t,
            kernel_size=(3, 3),
            num_layers=1,
            batch_first=True
        )
        
        self.init_act_head(fused_t * h * w)

    def forward(self, x, train=True):
        x, midas_in = x  # x: (seq_len, C, H, W)
        seq_len, channels, height, width = x.size()  # batch_size = seq_len

        path_1 = self._process_midas(midas_in)  # (seq_len, 64, H, W)

        features = self.features(x)  # (seq_len, t, h, w)
        d = F.interpolate(path_1, size=self.features_hw, mode='bilinear', align_corners=False)  # (seq_len, 64, h, w)

        combined = self.cross_attention(features.unsqueeze(0), d.unsqueeze(0))  # (1, seq_len, t, h, w)

        o, hs = self.conv_lstm(combined, hidden_state=None if train else self.hs)
        if not train: self.hs = hs
        o = o[0].squeeze(0)
        assert (o.shape[0] == seq_len)
        o = o.reshape(o.shape[0], -1)

        return tuple(self.fc_layers[name](o) for name in self.fc_layers)


class DVNetDualBase(DVNetBase):
    def _process_midas(self, midas_in, height, width):
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
        return midas_out


class DVNetDual_CA(DVNetDualBase):
    def __init__(self):
        super().__init__()
        self.init_midas()
        self.raw_features, t, h, w = self.get_efficientnet_b5() 
        self.deep_features, t, h, w = self.get_efficientnet_b5() 

        from .fusion import FeatureFusionModule, HWC_SpatialAttention
        self.fusion_module = FeatureFusionModule(t)
        self.sa1 = HWC_SpatialAttention(t, t, t)
        self.sa2 = HWC_SpatialAttention(t, t, t)

        fused_t = t
        self.conv_lstm = ConvLSTM(
            input_dim=fused_t,
            hidden_dim=fused_t,
            kernel_size=(3, 3),
            num_layers=1,
            batch_first=True
        )
        
        self.init_act_head(fused_t * h * w)
    
    
    def forward(self, x, train=True):
        x, midas_in = x  # x: (seq_len, C, H, W)
        seq_len, channels, height, width = x.size()  # batch_size = seq_len

        midas_out = self._process_midas(midas_in, height, width)
        raw_feat = self.raw_features(x).unsqueeze(0)
        deep_feat = self.deep_features(midas_out).unsqueeze(0)

        raw_feat = self.sa1(raw_feat, deep_feat)
        deep_feat = self.sa2(deep_feat, raw_feat)
        combined = self.fusion_module(raw_feat, deep_feat)
        
        o, hs = self.conv_lstm(combined, hidden_state=None if train else self.hs)
        if not train: self.hs = hs
        o = o[0].squeeze(0)
        assert (o.shape[0] == seq_len)
        o = o.reshape(o.shape[0], -1)


        return tuple(self.fc_layers[name](o) for name in self.fc_layers)