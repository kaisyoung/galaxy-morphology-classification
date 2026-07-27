from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from sklearn.metrics import ConfusionMatrixDisplay

from src.evaluation.metrics import CLASS_NAMES


def normalize_confusion_matrix(confusion):
    """Normalize each row of a confusion matrix."""

    confusion = np.asarray(
        confusion,
        dtype=float,
    )

    row_sums = confusion.sum(
        axis=1,
        keepdims=True,
    )

    return np.divide(
        confusion,
        row_sums,
        out=np.zeros_like(
            confusion,
            dtype=float,
        ),
        where=row_sums != 0,
    )


def plot_confusion_matrix(
    confusion,
    title,
    normalize=False,
    save_path: Path | str | None = None,
):
    """Plot a raw or row-normalized confusion matrix."""

    confusion = np.asarray(confusion)

    if normalize:
        displayed_confusion = (
            normalize_confusion_matrix(confusion)
        )
        value_format = ".2f"
    else:
        displayed_confusion = confusion
        value_format = "d"

    figure, axis = plt.subplots(
        figsize=(11, 9),
    )

    display = ConfusionMatrixDisplay(
        confusion_matrix=displayed_confusion,
        display_labels=CLASS_NAMES,
    )

    display.plot(
        ax=axis,
        xticks_rotation=45,
        values_format=value_format,
        colorbar=True,
    )

    axis.set_title(title)
    figure.tight_layout()

    if save_path is not None:
        save_path = Path(save_path)
        save_path.parent.mkdir(
            parents=True,
            exist_ok=True,
        )

        figure.savefig(
            save_path,
            dpi=300,
            bbox_inches="tight",
        )

    return figure, axis