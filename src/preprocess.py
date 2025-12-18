"""Data loading and preprocessing for CIFAR-10 and ImageNet-100."""

from pathlib import Path
from typing import Tuple, Callable, Optional

import torch
from torch.utils.data import Dataset
import torchvision.transforms as transforms
import torchvision.datasets as datasets
from omegaconf import DictConfig


def get_transforms(cfg: DictConfig) -> Tuple[transforms.Compose, transforms.Compose]:
    """Get train and validation transforms.

    Args:
        cfg: Full configuration with both dataset and model sections
    """

    # Mean and std for normalization
    mean = cfg.dataset.preprocessing.mean
    std = cfg.dataset.preprocessing.std

    normalize = transforms.Normalize(mean=mean, std=std)

    # Training transforms (with augmentation)
    train_transforms = [transforms.ToTensor()]

    if cfg.dataset.preprocessing.augmentation:
        if "random_crop" in cfg.dataset.preprocessing.augmentation_types:
            train_transforms.insert(0, transforms.RandomCrop(
                cfg.model.input_size,
                padding=4
            ))
        if "random_horizontal_flip" in cfg.dataset.preprocessing.augmentation_types:
            train_transforms.insert(1, transforms.RandomHorizontalFlip())

    train_transforms.append(normalize)
    train_transform = transforms.Compose(train_transforms)
    
    # Validation transforms (no augmentation)
    val_transform = transforms.Compose([
        transforms.ToTensor(),
        normalize,
    ])
    
    return train_transform, val_transform


def build_dataset(
    dataset_cfg: DictConfig,
    train: bool = True,
    transform: Optional[Callable] = None,
    cache_dir: str = ".cache",
) -> Dataset:
    """Build and return dataset."""
    
    # Create cache directory
    cache_path = Path(cache_dir)
    cache_path.mkdir(parents=True, exist_ok=True)
    
    if dataset_cfg.name == "CIFAR-10":
        dataset = datasets.CIFAR10(
            root=str(cache_path / "CIFAR-10"),
            train=train,
            download=True,
            transform=transform,
        )
    elif dataset_cfg.name == "ImageNet-100":
        # ImageNet-100: requires manual setup
        imagenet_path = cache_path / "ImageNet-100"
        if not imagenet_path.exists():
            raise FileNotFoundError(
                f"ImageNet-100 data not found at {imagenet_path}. "
                "Please manually download ImageNet and prepare 100-class subset."
            )
        
        from torchvision.datasets import ImageFolder
        split_dir = "train" if train else "val"
        dataset = ImageFolder(
            root=str(imagenet_path / split_dir),
            transform=transform,
        )
    else:
        raise ValueError(f"Unknown dataset: {dataset_cfg.name}")
    
    return dataset