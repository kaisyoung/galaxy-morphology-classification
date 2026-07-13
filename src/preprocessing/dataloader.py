from astroNN.datasets import load_galaxy10
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
):
    images, labels = load_galaxy10()

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

    transform = get_transform()

    train_dataset = Galaxy10Dataset(train_images, train_labels, transform=transform)
    val_dataset = Galaxy10Dataset(val_images, val_labels, transform=transform)
    test_dataset = Galaxy10Dataset(test_images, test_labels, transform=transform)

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
