"""Evaluation metrics for the classification head."""

import json
import os
from typing import Dict, List

import numpy as np
import torch
from torch.utils.data import DataLoader


def evaluate(backbone, classifier, dataloader, device: str = "cpu") -> Dict:
    """Run evaluation and compute classification metrics.

    Returns dict with accuracy, macro_f1, per-class metrics.
    """
    classifier.eval()
    all_preds = []
    all_labels = []

    with torch.no_grad():
        for images, labels in dataloader:
            # Only evaluate labeled samples
            mask = labels != -1
            if mask.sum() == 0:
                continue

            images = images[mask]
            labels = labels[mask]

            embeddings = backbone.extract(images)
            logits = classifier(embeddings.to(device))
            preds = logits.argmax(dim=1).cpu()

            all_preds.extend(preds.tolist())
            all_labels.extend(labels.tolist())

    if not all_labels:
        return {"accuracy": 0.0, "macro_f1": 0.0, "num_samples": 0}

    from sklearn.metrics import accuracy_score, f1_score, classification_report

    accuracy = accuracy_score(all_labels, all_preds)
    macro_f1 = f1_score(all_labels, all_preds, average="macro", zero_division=0)

    report = classification_report(
        all_labels, all_preds, zero_division=0, output_dict=True,
    )

    return {
        "accuracy": accuracy,
        "macro_f1": macro_f1,
        "num_samples": len(all_labels),
        "classification_report": report,
    }


def save_confusion_matrix(
    all_labels: List[int],
    all_preds: List[int],
    class_names: List[str],
    output_dir: str,
):
    """Save confusion matrix as JSON and PNG."""
    from sklearn.metrics import confusion_matrix

    cm = confusion_matrix(all_labels, all_preds)
    os.makedirs(output_dir, exist_ok=True)

    # Save as JSON
    cm_path = os.path.join(output_dir, "confusion_matrix.json")
    with open(cm_path, "w") as f:
        json.dump({
            "matrix": cm.tolist(),
            "class_names": class_names,
        }, f, indent=2)

    # Save as PNG
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt

        fig, ax = plt.subplots(figsize=(8, 6))
        im = ax.imshow(cm, interpolation="nearest", cmap=plt.cm.Blues)
        ax.set_title("Confusion Matrix")
        fig.colorbar(im)

        tick_marks = list(range(len(class_names)))
        ax.set_xticks(tick_marks)
        ax.set_xticklabels(class_names, rotation=45, ha="right")
        ax.set_yticks(tick_marks)
        ax.set_yticklabels(class_names)

        ax.set_ylabel("True Label")
        ax.set_xlabel("Predicted Label")
        fig.tight_layout()

        png_path = os.path.join(output_dir, "confusion_matrix.png")
        fig.savefig(png_path, dpi=150)
        plt.close(fig)
        print(f"Confusion matrix saved to {png_path}")
    except Exception as e:
        print(f"Could not save confusion matrix plot: {e}")
