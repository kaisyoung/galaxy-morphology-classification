from sklearn.model_selection import train_test_split


def split_dataset(
    images,
    labels,
    train_ratio=0.8,
    val_ratio=0.1,
    test_ratio=0.1,
    random_state=42,
    stratify=True,
):
    total_ratio = train_ratio + val_ratio + test_ratio
    if abs(total_ratio - 1.0) > 1e-8:
        raise ValueError("train_ratio, val_ratio, and test_ratio must sum to 1.0")

    validation_test_ratio = val_ratio + test_ratio
    stratify_labels = labels if stratify else None

    train_images, remaining_images, train_labels, remaining_labels = train_test_split(
        images,
        labels,
        test_size=validation_test_ratio,
        random_state=random_state,
        stratify=stratify_labels,
    )

    test_fraction = test_ratio / validation_test_ratio
    remaining_stratify_labels = remaining_labels if stratify else None

    val_images, test_images, val_labels, test_labels = train_test_split(
        remaining_images,
        remaining_labels,
        test_size=test_fraction,
        random_state=random_state,
        stratify=remaining_stratify_labels,
    )

    return (
        train_images,
        train_labels,
        val_images,
        val_labels,
        test_images,
        test_labels,
    )
