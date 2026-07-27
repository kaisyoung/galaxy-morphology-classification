from __future__ import annotations

import argparse
from pathlib import Path
from typing import Callable

import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from src.baseline.train_baseline import (
    BATCH_SIZE,
    DEVICE,
    NUM_WORKERS,
    SEED,
    build_model,
    build_transforms,
)
from src.evaluation.metrics import CLASS_NAMES
from src.models.convnext_nano import build_convnext_nano
from src.models.convnext_nano_imagenet import build_convnext_nano_imagenet
from src.models.densenet121 import build_densenet121
from src.models.resnet50 import build_resnet50
from src.models.resnet50_imagenet import build_resnet50_imagenet
from src.preprocessing.dataloader import build_dataloaders


PROJECT_ROOT = Path(__file__).resolve().parents[2]
CHECKPOINT_DIR = PROJECT_ROOT / "outputs" / "checkpoints"
GRADCAM_DIR = PROJECT_ROOT / "outputs" / "figures" / "gradcam"

MODEL_BUILDERS: dict[str, Callable[[], nn.Module]] = {
    "simplecnn": lambda: build_model("simplecnn"),
    "vgg16": lambda: build_model("vgg16"),
    "densenet121": build_densenet121,
    "resnet50_imagenet": build_resnet50_imagenet,
    "resnet50_zoobot": build_resnet50,
    "convnext_nano_imagenet": build_convnext_nano_imagenet,
    "convnext_nano_zoobot": build_convnext_nano,
}

DISPLAY_NAMES = {
    "simplecnn": "SimpleCNN",
    "vgg16": "VGG16",
    "densenet121": "DenseNet121",
    "resnet50_imagenet": "ResNet50 ImageNet",
    "resnet50_zoobot": "ResNet50 Zoobot",
    "convnext_nano_imagenet": "ConvNeXT-Nano ImageNet",
    "convnext_nano_zoobot": "ConvNeXT-Nano Zoobot",
}


class GradCAM:
    """Minimal Grad-CAM implementation for image classifiers."""

    def __init__(
        self,
        model: nn.Module,
        target_layer: nn.Module,
    ) -> None:
        self.model = model
        self.activations: torch.Tensor | None = None
        self.gradients: torch.Tensor | None = None

        self._forward_handle = target_layer.register_forward_hook(
            self._capture_activations
        )

    def _capture_activations(
        self,
        _module: nn.Module,
        _inputs: tuple[torch.Tensor, ...],
        output: torch.Tensor,
    ) -> None:
        if not isinstance(output, torch.Tensor):
            raise TypeError(
                "The selected Grad-CAM layer must return a tensor."
            )

        self.activations = output
        output.register_hook(self._capture_gradients)

    def _capture_gradients(
        self,
        gradients: torch.Tensor,
    ) -> None:
        self.gradients = gradients.detach()

    def generate(
        self,
        input_tensor: torch.Tensor,
        target_class: int | None = None,
    ) -> tuple[np.ndarray, int, int, torch.Tensor]:
        """Return heatmap, predicted class, target class and probabilities."""

        self.activations = None
        self.gradients = None
        self.model.zero_grad(set_to_none=True)

        logits = self.model(input_tensor)

        if not isinstance(logits, torch.Tensor) or logits.ndim != 2:
            raise ValueError(
                "The model must return logits with shape [batch, classes]."
            )

        predicted_class = int(logits.argmax(dim=1).item())

        if target_class is None:
            target_class = predicted_class

        if not 0 <= target_class < logits.shape[1]:
            raise ValueError(
                f"target_class must be between 0 and {logits.shape[1] - 1}."
            )

        logits[0, target_class].backward()

        if self.activations is None or self.gradients is None:
            raise RuntimeError(
                "Grad-CAM could not capture activations or gradients."
            )

        if self.activations.ndim != 4:
            raise ValueError(
                "The selected target layer must produce [B, C, H, W] features."
            )

        weights = self.gradients.mean(dim=(2, 3), keepdim=True)
        cam = (weights * self.activations.detach()).sum(
            dim=1,
            keepdim=True,
        )
        cam = F.relu(cam)
        cam = F.interpolate(
            cam,
            size=input_tensor.shape[-2:],
            mode="bilinear",
            align_corners=False,
        )

        heatmap = cam[0, 0].cpu().numpy()
        minimum = float(heatmap.min())
        maximum = float(heatmap.max())

        if maximum > minimum:
            heatmap = (heatmap - minimum) / (maximum - minimum)
        else:
            heatmap = np.zeros_like(heatmap, dtype=np.float32)

        probabilities = torch.softmax(logits.detach(), dim=1)

        return (
            heatmap,
            predicted_class,
            target_class,
            probabilities,
        )

    def close(self) -> None:
        self._forward_handle.remove()


def build_project_model(model_name: str) -> nn.Module:
    """Build one of the seven project model configurations."""

    if model_name not in MODEL_BUILDERS:
        supported = ", ".join(MODEL_BUILDERS)
        raise ValueError(
            f"Unsupported model '{model_name}'. Supported models: {supported}"
        )

    return MODEL_BUILDERS[model_name]()


def load_trained_model(
    model_name: str,
    device: torch.device = DEVICE,
) -> nn.Module:
    """Build a model and load the checkpoint saved by train_one()."""

    checkpoint_path = CHECKPOINT_DIR / f"{model_name}.pt"

    if not checkpoint_path.exists():
        raise FileNotFoundError(
            f"Checkpoint not found: {checkpoint_path}"
        )

    model = build_project_model(model_name)

    try:
        checkpoint = torch.load(
            checkpoint_path,
            map_location=device,
            weights_only=True,
        )
    except TypeError:
        # Compatibility with older PyTorch versions.
        checkpoint = torch.load(
            checkpoint_path,
            map_location=device,
        )

    if (
        isinstance(checkpoint, dict)
        and "model_state_dict" in checkpoint
    ):
        state_dict = checkpoint["model_state_dict"]
    else:
        state_dict = checkpoint

    if not isinstance(state_dict, dict):
        raise TypeError(
            "Checkpoint must be a state_dict or contain 'model_state_dict'."
        )

    model.load_state_dict(state_dict)
    model = model.to(device)
    model.eval()

    return model


def find_last_convolution(model: nn.Module) -> nn.Conv2d:
    """Select the last Conv2d layer, which remains spatial for all models."""

    convolution_layers = [
        module
        for module in model.modules()
        if isinstance(module, nn.Conv2d)
    ]

    if not convolution_layers:
        raise ValueError(
            "No Conv2d layer was found in the selected model."
        )

    return convolution_layers[-1]


def load_test_sample(
    sample_index: int,
    device: torch.device = DEVICE,
) -> tuple[torch.Tensor, int]:
    """Load one image from the same fixed test split used during training."""

    _, evaluation_transform = build_transforms()

    _, _, test_loader = build_dataloaders(
        batch_size=BATCH_SIZE,
        train_ratio=0.8,
        val_ratio=0.1,
        test_ratio=0.1,
        random_state=SEED,
        num_workers=NUM_WORKERS,
        train_transform=evaluation_transform,
        val_transform=evaluation_transform,
        test_transform=evaluation_transform,
    )

    test_dataset = test_loader.dataset

    if not 0 <= sample_index < len(test_dataset):
        raise IndexError(
            f"sample_index must be between 0 and {len(test_dataset) - 1}."
        )

    image, true_class = test_dataset[sample_index]

    if not isinstance(image, torch.Tensor) or image.ndim != 3:
        raise TypeError(
            "The test dataset must return an image tensor with shape [C, H, W]."
        )

    return image.unsqueeze(0).to(device), int(true_class)


def tensor_to_image(input_tensor: torch.Tensor) -> np.ndarray:
    """Reverse the ImageNet normalisation used by build_transforms()."""

    mean = torch.tensor(
        [0.485, 0.456, 0.406],
        dtype=input_tensor.dtype,
        device=input_tensor.device,
    ).view(1, 3, 1, 1)

    std = torch.tensor(
        [0.229, 0.224, 0.225],
        dtype=input_tensor.dtype,
        device=input_tensor.device,
    ).view(1, 3, 1, 1)

    image = input_tensor * std + mean

    return (
        image[0]
        .detach()
        .cpu()
        .clamp(0, 1)
        .permute(1, 2, 0)
        .numpy()
    )


def create_gradcam_figure(
    image: np.ndarray,
    heatmap: np.ndarray,
    model_name: str,
    sample_index: int,
    true_class: int,
    predicted_class: int,
    predicted_confidence: float,
) -> plt.Figure:
    """Create and return a Grad-CAM figure."""

    figure, axes = plt.subplots(
        1,
        3,
        figsize=(15.5, 5.0),
    )

    axes[0].imshow(
        image
    )

    axes[0].set_title(
        f"Input image\n"
        f"True: {CLASS_NAMES[true_class]}",
        fontsize=11,
        fontweight="normal",
        color="#30343B",
    )

    heatmap_artist = axes[1].imshow(
        heatmap,
        cmap="turbo",
        vmin=0,
        vmax=1,
    )

    axes[1].set_title(
        "Grad-CAM heatmap",
        fontsize=11,
        fontweight="normal",
        color="#30343B",
    )

    axes[2].imshow(
        image
    )

    axes[2].imshow(
        heatmap,
        cmap="turbo",
        alpha=0.35,
        vmin=0,
        vmax=1,
    )

    status = (
        "Correct"
        if predicted_class == true_class
        else "Incorrect"
    )

    axes[2].set_title(
        f"{status} prediction\n"
        f"Predicted: "
        f"{CLASS_NAMES[predicted_class]}\n"
        f"Confidence: "
        f"{predicted_confidence:.3f}",
        fontsize=11,
        fontweight="normal",
        color="#30343B",
    )

    for axis in axes:
        axis.axis(
            "off"
        )

    colorbar = figure.colorbar(
        heatmap_artist,
        ax=axes,
        fraction=0.022,
        pad=0.05,
    )

    colorbar.set_label(
        "Relative importance",
        rotation=270,
        labelpad=16,
    )

    figure.suptitle(
        f"{DISPLAY_NAMES.get(model_name, model_name)} "
        f"Grad-CAM — test sample {sample_index}",
        x=0.5,
        y=0.98,
        fontsize=13,
        fontweight="normal",
        color="#30343B",
    )

    figure.subplots_adjust(
        top=0.80,
        wspace=0.16,
        right=0.86,
    )

    return figure


def generate_gradcam(
    model_name: str,
    sample_index: int,
    target_class: int | None = None,
    save: bool = True,
    return_figure: bool = False,
) -> dict[str, object]:
    """Generate one Grad-CAM result for any project model."""

    model = load_trained_model(
        model_name
    )

    target_layer = find_last_convolution(
        model
    )

    input_tensor, true_class = load_test_sample(
        sample_index
    )

    gradcam = GradCAM(
        model,
        target_layer,
    )

    try:
        (
            heatmap,
            predicted_class,
            used_target_class,
            probabilities,
        ) = gradcam.generate(
            input_tensor,
            target_class=target_class,
        )
    finally:
        gradcam.close()

    predicted_confidence = float(
        probabilities[
            0,
            predicted_class,
        ].item()
    )

    output_path = (
        GRADCAM_DIR
        / f"{model_name}_sample_{sample_index}.png"
    )

    figure = create_gradcam_figure(
        image=tensor_to_image(
            input_tensor
        ),
        heatmap=heatmap,
        model_name=model_name,
        sample_index=sample_index,
        true_class=true_class,
        predicted_class=predicted_class,
        predicted_confidence=(
            predicted_confidence
        ),
    )

    if save:
        output_path.parent.mkdir(
            parents=True,
            exist_ok=True,
        )

        figure.savefig(
            output_path,
            dpi=300,
            bbox_inches="tight",
        )

    result: dict[str, object] = {
        "model_name": model_name,
        "sample_index": sample_index,
        "true_class": (
            CLASS_NAMES[true_class]
        ),
        "predicted_class": (
            CLASS_NAMES[predicted_class]
        ),
        "predicted_confidence": (
            predicted_confidence
        ),
        "target_class": (
            CLASS_NAMES[used_target_class]
        ),
        "correct": (
            predicted_class
            == true_class
        ),
        "output_path": (
            output_path
            if save
            else None
        ),
    }

    if return_figure:
        result["figure"] = figure
    else:
        plt.close(
            figure
        )

    return result


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Generate Grad-CAM for a trained project model."
    )
    parser.add_argument(
        "--model",
        required=True,
        choices=list(MODEL_BUILDERS),
    )
    parser.add_argument(
        "--sample-index",
        type=int,
        default=0,
    )
    parser.add_argument(
        "--target-class",
        type=int,
        default=None,
    )
    parser.add_argument(
        "--no-save",
        action="store_true",
    )

    arguments = parser.parse_args()

    result = generate_gradcam(
        model_name=arguments.model,
        sample_index=arguments.sample_index,
        target_class=arguments.target_class,
        save=not arguments.no_save,
    )

    print("\n================ GRAD-CAM ================")
    for key, value in result.items():
        print(f"{key}: {value}")


if __name__ == "__main__":
    main()
