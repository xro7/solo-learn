import albumentations as A
from albumentations.pytorch import ToTensorV2
from PIL import Image
import numpy as np
from typing import List
import torch


def get_albumentations(image_size=224):
    augmentations = {'visualization': A.Compose([

        A.OneOf([
            A.Sequential([
                A.LongestMaxSize(max_size=image_size),
                A.PadIfNeeded(min_height=image_size, min_width=image_size, border_mode=0, value=0),
            ], p=0.3),
            A.RandomResizedCrop(image_size, image_size, scale=(0.85, 1.0), ratio=(1.0, 3), p=0.45),
            A.Resize(image_size, image_size, p=0.25)
        ], p=1.0),

        A.OneOf([
            A.ShiftScaleRotate(shift_limit=0.0625, scale_limit=0.0, rotate_limit=45, border_mode=0, p=1.0),
            A.Perspective(scale=(0.05, 0.15), p=1.0),
            A.OpticalDistortion(p=1.0),
        ], p=0.3),

        A.OneOf([
            A.ChannelShuffle(p=1.0),
            A.ColorJitter(brightness=0.4, contrast=0.4, saturation=0.2, p=1.0),
            A.FancyPCA(p=0.1),
        ], p=0.4),

        A.OneOf([
            A.GaussianBlur(blur_limit=5, p=1.0),
            A.InvertImg(p=1.0),
            A.ToGray(p=1.0),
        ], p=0.3),

    ]), 'train': A.Compose([

        A.OneOf([
            A.Sequential([
                A.LongestMaxSize(max_size=image_size),
                A.PadIfNeeded(min_height=image_size, min_width=image_size, border_mode=0, value=0),
            ], p=0.3),
            A.RandomResizedCrop(image_size, image_size, scale=(0.85, 1.0), ratio=(1.0, 3), p=0.45),
            A.Resize(image_size, image_size, p=0.25)
        ], p=1.0),

        A.OneOf([
            A.ShiftScaleRotate(shift_limit=0.0625, scale_limit=0.0, rotate_limit=45, border_mode=0, p=1.0),
            A.Perspective(scale=(0.05, 0.15), p=1.0),
            A.OpticalDistortion(p=1.0),
        ], p=0.3),

        A.OneOf([
            A.ChannelShuffle(p=1.0),
            A.ColorJitter(brightness=0.4, contrast=0.4, saturation=0.2, p=1.0),
            A.FancyPCA(p=0.1),
        ], p=0.4),

        A.OneOf([
            A.GaussianBlur(blur_limit=5, p=1.0),
            A.InvertImg(p=1.0),
            A.ToGray(p=1.0),
        ], p=0.3),

        A.Normalize(
            mean=[0.485, 0.456, 0.406],
            std=[0.229, 0.224, 0.225],
        ),
        ToTensorV2()

    ]), 'val': A.Compose([

        A.LongestMaxSize(max_size=image_size),
        A.PadIfNeeded(min_height=image_size, min_width=image_size, border_mode=0, value=0),
        A.Normalize(
            mean=[0.485, 0.456, 0.406],
            std=[0.229, 0.224, 0.225],
        ),
        ToTensorV2()
    ])

    }

    return augmentations


def get_albumentations_v2(image_size=224):
    augmentations = {'visualization': A.Compose([

        A.OneOf([
            A.Sequential([
                A.LongestMaxSize(max_size=image_size),
                A.OneOf([
                    A.PadIfNeeded(min_height=image_size, min_width=image_size, border_mode=0, value=0),
                    A.PadIfNeeded(min_height=image_size, min_width=image_size, border_mode=0, value=[255, 255, 255]),
                ], p=1.0),
            ], p=0.3),
        ], p=1.0),

        A.OneOf([
            A.ShiftScaleRotate(shift_limit=0.0625, scale_limit=0.0, rotate_limit=45, border_mode=0, p=1.0),
            A.Perspective(scale=(0.05, 0.15), p=1.0),
            A.OpticalDistortion(p=1.0),
        ], p=0.3),

        A.OneOf([
            A.ChannelShuffle(p=1.0),
            A.ColorJitter(brightness=0.4, contrast=0.4, saturation=0.2, p=1.0),
            A.FancyPCA(p=0.1),
        ], p=0.4),

        A.OneOf([
            A.GaussianBlur(blur_limit=5, p=1.0),
            A.InvertImg(p=1.0),
            A.ToGray(p=1.0),
        ], p=0.3),

        # A.OneOf([
        #     A.RandomToneCurve(),
        #     #A.HorizontalFlip(),
        #     A.Sharpen(),
        # ], p=0.1)

    ]), 'train': A.Compose([

        A.OneOf([
            A.Sequential([
                A.LongestMaxSize(max_size=image_size),
                A.OneOf([
                    A.PadIfNeeded(min_height=image_size, min_width=image_size, border_mode=0, value=0),
                    A.PadIfNeeded(min_height=image_size, min_width=image_size, border_mode=0, value=[255, 255, 255]),
                ], p=1.0),
            ], p=0.3),
        ], p=1.0),

        A.OneOf([
            A.ShiftScaleRotate(shift_limit=0.0625, scale_limit=0.0, rotate_limit=45, border_mode=0, p=1.0),
            A.Perspective(scale=(0.05, 0.15), p=1.0),
            A.OpticalDistortion(p=1.0),
        ], p=0.3),

        A.OneOf([
            A.ChannelShuffle(p=1.0),
            A.ColorJitter(brightness=0.4, contrast=0.4, saturation=0.2, p=1.0),
            A.FancyPCA(p=0.1),
        ], p=0.4),

        A.OneOf([
            A.GaussianBlur(blur_limit=5, p=1.0),
            A.InvertImg(p=1.0),
            A.ToGray(p=1.0),
        ], p=0.3),

        A.OneOf([
            A.RandomToneCurve(),
            A.HorizontalFlip(),
            A.Sharpen(),
        ], p=0.3),

        A.Normalize(
            mean=[0.485, 0.456, 0.406],
            std=[0.229, 0.224, 0.225],
        ),
        ToTensorV2()

    ]), 'val': A.Compose([

        A.LongestMaxSize(max_size=image_size),
        A.PadIfNeeded(min_height=image_size, min_width=image_size, border_mode=0, value=0),
        A.Normalize(
            mean=[0.485, 0.456, 0.406],
            std=[0.229, 0.224, 0.225],
        ),
        ToTensorV2()
    ])

    }

    return augmentations


class AugApplier:

    def __init__(self, augmentations, mode='visualization', number_of_augs=2) -> None:
        self.augmentations = augmentations
        self.mode = mode
        self.number_of_augs = number_of_augs

    def __call__(self, x: np.ndarray) -> List[torch.Tensor]:
        """Applies transforms n times to generate n crops.

        Args:
            x (Image): an image in the PIL.Image format.

        Returns:
            List[torch.Tensor]: an image in the tensor format.
        """
        if isinstance(x, Image.Image):
            x = np.array(x)
        out = []
        for i in range(self.number_of_augs):
            out.append(self.augmentations[self.mode](image=x)['image'])
        return out
