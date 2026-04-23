"""Image transforms for DINOv3 preprocessing."""

import torch
from torchvision import transforms
from PIL import Image


# DINOv3 uses ImageNet normalization
IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD = (0.229, 0.224, 0.225)


def get_train_transform(input_size: int = 224) -> transforms.Compose:
    """Training transform with augmentation."""
    return transforms.Compose([
        transforms.RandomResizedCrop(input_size, scale=(0.6, 1.0)),
        transforms.RandomHorizontalFlip(),
        transforms.ColorJitter(
            brightness=0.2, contrast=0.2, saturation=0.1, hue=0.05
        ),
        transforms.ToTensor(),
        transforms.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),
    ])


def get_eval_transform(input_size: int = 224) -> transforms.Compose:
    """Evaluation / embedding extraction transform."""
    return transforms.Compose([
        transforms.Resize(int(input_size * 256 / 224)),  # 256 for 224 crop
        transforms.CenterCrop(input_size),
        transforms.ToTensor(),
        transforms.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),
    ])
