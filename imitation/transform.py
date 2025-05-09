import torch
import numpy as np
from torchvision.transforms import Compose
import torchvision.transforms as transforms

class Normalize(object):
    def __init__(self, mean, std):
        self.__mean = np.array(mean, dtype=np.float32)
        self.__std = np.array(std, dtype=np.float32)

    def __call__(self, sample):
        sample = (sample - self.__mean) / self.__std
        return sample

class PrepareForNet(object):
    def __call__(self, sample):
        image = np.transpose(sample, (2, 0, 1))
        sample = np.ascontiguousarray(image).astype(np.float32)
        return sample


center_transform_test = Compose(
    [
        lambda img: (img / 255.0),
        PrepareForNet(),
        lambda sample: torch.from_numpy(sample),
        transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5]),
    ]
)

def debug(x):
    print("X")
    return x

def center_transform_train_f():
    return Compose(
        [
            lambda img: (img / 255.0),
            PrepareForNet(),
            lambda sample: torch.from_numpy(sample),
            # transforms.RandomRotation(degrees=(-0.5, -0.5), fill=(0, 0, 0)),
            transforms.RandomErasing(p=0.8, scale=(0.003, 0.003), ratio=(0.3, 0.3)),
            transforms.RandomErasing(p=0.8, scale=(0.003, 0.003), ratio=(0.3, 0.3)),
            transforms.RandomErasing(p=0.8, scale=(0.003, 0.003), ratio=(0.3, 0.3)),
            transforms.ColorJitter(brightness=0.05, contrast=0.05, saturation=0.05, hue=0.01),
            transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5]),
        ]
    )

center_transform_train = center_transform_train_f()

# center_transform_train = center_transform_test