from __future__ import annotations

import json
from pathlib import Path

import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_RESULT_DIR = PROJECT_ROOT / "outputs" / "results"


MODEL_DISPLAY_NAMES = {
    "simplecnn": "SimpleCNN",
    "vgg16": "VGG16",
    "densenet121": "DenseNet121",
    "resnet50_imagenet": "ResNet50 ImageNet",
    "resnet50_zoobot": "ResNet50 Zoobot",
    "convnext_nano_imagenet": "ConvNeXT-Nano ImageNet",
    "convnext_nano_zoobot": "ConvNeXT-Nano Zoobot",
}


PRETRAINING_TYPES = {
    "simplecnn": "None",
    "vgg16": "ImageNet",
    "densenet121": "ImageNet",
    "resnet50_imagenet": "ImageNet",
    "resnet50_zoobot": "Zoobot",
    "convnext_nano_imagenet": "ImageNet",
    "convnext_nano_zoobot": "Zoobot",
}


def load_model_results(
    result_dir: Path | str = DEFAULT_RESULT_DIR,
):
    """Load all model JSON results from the result directory."""

    result_dir = Path(result_dir)
    results = {}

    for result_path in sorted(result_dir.glob("*.json")):
        with result_path.open(
            "r",
            encoding="utf-8",
        ) as result_file:
            result = json.load(result_file)

        model_name = result["model_name"]
        results[model_name] = result

    return results


def build_summary_dataframe(results):
    """Build one-row-per-model comparison table."""

    rows = []

    for model_name, result in results.items():
        test_metrics = result["test_metrics"]

        test_sample_count = int(
            result.get("test_sample_count", 0)
        )

        inference_time_seconds = float(
            result["inference_time_seconds"]
        )

        if test_sample_count > 0 and inference_time_seconds > 0:
            inference_per_image_ms = (
                inference_time_seconds
                / test_sample_count
                * 1000
            )

            throughput_images_per_second = (
                test_sample_count
                / inference_time_seconds
            )
        else:
            inference_per_image_ms = float("nan")
            throughput_images_per_second = float("nan")

        rows.append(
            {
                "Model Key": model_name,
                "Model": MODEL_DISPLAY_NAMES.get(
                    model_name,
                    model_name,
                ),
                "Pretraining": PRETRAINING_TYPES.get(
                    model_name,
                    "Unknown",
                ),
                "Accuracy": test_metrics["accuracy"],
                "Macro-F1": test_metrics["macro_f1"],
                "Weighted-F1": test_metrics["weighted_f1"],
                "Parameters": result["parameter_count"],
                "Parameters (M)": (
                    result["parameter_count"]
                    / 1_000_000
                ),
                "Best Epoch": result["best_epoch"],
                "Training Time (min)": (
                    result["training_time_seconds"]
                    / 60
                ),
                "Inference Time (s)": (
                    inference_time_seconds
                ),
                "Test Samples": test_sample_count,
                "Inference per Image (ms)": (
                    inference_per_image_ms
                ),
                "Throughput (images/s)": (
                    throughput_images_per_second
                ),
            }
        )

    summary_df = pd.DataFrame(rows)

    if not summary_df.empty:
        summary_df = summary_df.sort_values(
            by="Macro-F1",
            ascending=False,
        ).reset_index(drop=True)

    return summary_df


def build_per_class_dataframe(results):
    """Build long-format per-class metric table."""

    rows = []

    for model_name, result in results.items():
        display_name = MODEL_DISPLAY_NAMES.get(
            model_name,
            model_name,
        )

        for class_name, metrics in (
            result["per_class_metrics"].items()
        ):
            rows.append(
                {
                    "Model Key": model_name,
                    "Model": display_name,
                    "Class": class_name,
                    "Precision": metrics["precision"],
                    "Recall": metrics["recall"],
                    "F1": metrics["f1"],
                    "Support": metrics["support"],
                }
            )

    return pd.DataFrame(rows)


def build_pretraining_ablation_dataframe(results):
    """Compare ImageNet and Zoobot for matching architectures."""

    model_pairs = {
        "ResNet50": (
            "resnet50_imagenet",
            "resnet50_zoobot",
        ),
        "ConvNeXT-Nano": (
            "convnext_nano_imagenet",
            "convnext_nano_zoobot",
        ),
    }

    rows = []

    for architecture, (
        imagenet_key,
        zoobot_key,
    ) in model_pairs.items():
        if (
            imagenet_key not in results
            or zoobot_key not in results
        ):
            continue

        imagenet_metrics = (
            results[imagenet_key]["test_metrics"]
        )
        zoobot_metrics = (
            results[zoobot_key]["test_metrics"]
        )

        rows.append(
            {
                "Architecture": architecture,
                "ImageNet Accuracy": (
                    imagenet_metrics["accuracy"]
                ),
                "Zoobot Accuracy": (
                    zoobot_metrics["accuracy"]
                ),
                "Accuracy Difference": (
                    zoobot_metrics["accuracy"]
                    - imagenet_metrics["accuracy"]
                ),
                "ImageNet Macro-F1": (
                    imagenet_metrics["macro_f1"]
                ),
                "Zoobot Macro-F1": (
                    zoobot_metrics["macro_f1"]
                ),
                "Macro-F1 Difference": (
                    zoobot_metrics["macro_f1"]
                    - imagenet_metrics["macro_f1"]
                ),
            }
        )

    return pd.DataFrame(rows)


def get_best_model(summary_df):
    """Return the row with the highest Macro-F1."""

    if summary_df.empty:
        raise ValueError(
            "The summary dataframe is empty."
        )

    return summary_df.loc[
        summary_df["Macro-F1"].idxmax()
    ]