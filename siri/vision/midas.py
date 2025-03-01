import os, cv2, torch
from midas.dpt_depth import DPTDepthModel
from torchvision.transforms import Compose
from midas.transforms import Resize, NormalizeImage, PrepareForNet
from midas.midas_net import MidasNet
from midas.midas_net_custom import MidasNet_small


def swin_256():
    local_model_path = './dpt_swin2_tiny_256.pt'
    assert os.path.exists(local_model_path)
    model = DPTDepthModel(
        path=local_model_path,
        backbone="swin2t16_256",
        non_negative=True,
    )
    transform = Compose(
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
            lambda sample: torch.from_numpy(sample["image"]).unsqueeze(0),
        ]
    )
    model_sz = (256, 256,)

    print(f"loaded {local_model_path}")
    return model, transform, model_sz

def swin_l_384():
    local_model_path = './dpt_swin_large_384.pt'
    assert os.path.exists(local_model_path)
    model = DPTDepthModel(
        path=local_model_path,
        backbone="swinl12_384",
        non_negative=True,
    )
    transform = Compose(
        [
            lambda img: {"image": img / 255.0},
            Resize(
                384,
                384,
                resize_target=None,
                keep_aspect_ratio=False,
                ensure_multiple_of=32,
                resize_method="minimal",
                image_interpolation_method=cv2.INTER_CUBIC,
            ),
            NormalizeImage(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5]),
            PrepareForNet(),
            lambda sample: torch.from_numpy(sample["image"]).unsqueeze(0),
        ]
    )

    model_sz = (384, 384,)

    print(f"loaded {local_model_path}")
    return model, transform, model_sz

def dpt_hybrid_384():
    local_model_path = './dpt_hybrid_384.pt'
    assert os.path.exists(local_model_path)
    model = DPTDepthModel(
        path=local_model_path,
        backbone="vitb_rn50_384",
        non_negative=True,
    )
    transform = Compose(
        [
            lambda img: {"image": img / 255.0},
            Resize(
                384,
                384,
                resize_target=None,
                keep_aspect_ratio=True,
                ensure_multiple_of=32,
                resize_method="minimal",
                image_interpolation_method=cv2.INTER_CUBIC,
            ),
            NormalizeImage(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5]),
            PrepareForNet(),
            lambda sample: torch.from_numpy(sample["image"]).unsqueeze(0),
        ]
    )
    model_sz = (384, 384,)

    print(f"loaded {local_model_path}")
    return model, transform, model_sz



def midas_small_256():
    local_model_path = './midas_v21_small_256.pt'
    assert os.path.exists(local_model_path)
    model = MidasNet_small(path=local_model_path, features=64, backbone="efficientnet_lite3", exportable=True, non_negative=True, blocks={'expand': True})
    transform = Compose(
        [
            lambda img: {"image": img / 255.0},
            Resize(
                256,
                256,
                resize_target=None,
                keep_aspect_ratio=True,
                ensure_multiple_of=32,
                resize_method="upper_bound",
                image_interpolation_method=cv2.INTER_CUBIC,
            ),
            NormalizeImage(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
            PrepareForNet(),
            lambda sample: torch.from_numpy(sample["image"]).unsqueeze(0),
        ]
    )
    model_sz = (256, 256,)

    print(f"loaded {local_model_path}")
    return model, transform, model_sz