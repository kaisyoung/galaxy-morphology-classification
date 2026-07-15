from astroNN.datasets import load_galaxy10
import numpy as np
from torch.utils.data import DataLoader

from src.preprocessing.dataset import Galaxy10Dataset
from src.preprocessing.split import split_dataset
from src.preprocessing.transforms import get_transform


def build_dataloaders(
    batch_size=32,
    train_ratio=0.8,
    val_ratio=0.1,
    test_ratio=0.1,
    random_state=42,
    num_workers=0,
    train_transform=None,
    val_transform=None,
    test_transform=None,
):
    images, labels = load_galaxy10()
    labels = labels.astype(np.int64)

    (
        train_images,
        train_labels,
        val_images,
        val_labels,
        test_images,
        test_labels,
    ) = split_dataset(
        images,
        labels,
        train_ratio=train_ratio,
        val_ratio=val_ratio,
        test_ratio=test_ratio,
        random_state=random_state,
    )

    if train_transform is None:
        train_transform = get_transform()

    if val_transform is None:
        val_transform = get_transform()

    if test_transform is None:
        test_transform = get_transform()

    train_dataset = Galaxy10Dataset(train_images, train_labels, transform=train_transform)
    val_dataset = Galaxy10Dataset(val_images, val_labels, transform=val_transform)
    test_dataset = Galaxy10Dataset(test_images, test_labels, transform=test_transform)

    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
    )
    test_loader = DataLoader(
        test_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
    )

    return train_loader, val_loader, test_loader
