"""Main training loop for the classification head on DINOv3 embeddings."""

import json
import os
from datetime import datetime

import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
from tqdm import tqdm

from ml.data.dataset import AquaSiteDataset, build_dataset
from ml.data.transforms import get_eval_transform, get_train_transform
from ml.models.backbone import DINOv3Backbone
from ml.models.classifier import build_classifier, save_checkpoint
from ml.training.evaluate import evaluate


def train(config):
    """Run the full training pipeline.

    1. Load dataset (local or Supabase)
    2. Initialize frozen DINOv3 backbone
    3. Train linear classifier head
    4. Log metrics and save checkpoints
    """
    device = config.training.device
    print(f"Device: {device}")

    # ── Build dataset ──────────────────────────────────────────────
    dataset = build_dataset(config)

    if not dataset.has_labels:
        print("\n" + "=" * 60)
        print("WARNING: No labeled samples found!")
        print("The training loop requires labels to compute loss.")
        print("Options:")
        print("  1. Label images via the Django annotation UI")
        print("  2. Provide a labels_csv in config.yaml")
        print("  3. Run 'embed' instead to extract embeddings for clustering")
        print("=" * 60 + "\n")
        print("Running embedding extraction instead...")
        from ml.training.embed import extract_embeddings
        extract_embeddings(config)
        return

    print(f"Classes ({dataset.num_classes}): {dataset.class_names}")

    # ── Split into train / val ─────────────────────────────────────
    train_subset, val_subset = dataset.split(train_frac=0.8)

    # Apply different transforms: training augmentation vs eval
    # We wrap the dataset so subsets get the right transform
    train_dataset = _TransformOverride(
        train_subset, get_train_transform(config.model.input_size)
    )
    val_dataset = _TransformOverride(
        val_subset, get_eval_transform(config.model.input_size)
    )

    train_loader = DataLoader(
        train_dataset,
        batch_size=config.training.batch_size,
        shuffle=True,
        num_workers=config.training.num_workers,
        pin_memory=(device == "cuda"),
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=config.training.batch_size,
        shuffle=False,
        num_workers=config.training.num_workers,
        pin_memory=(device == "cuda"),
    )

    # ── Initialize models ──────────────────────────────────────────
    backbone = DINOv3Backbone(
        model_name=config.model.name,
        device=device,
        hf_token=config.hf_token,
    )

    classifier = build_classifier(config, num_classes=dataset.num_classes).to(device)
    print(f"Classifier: {classifier.__class__.__name__}")
    trainable_params = sum(p.numel() for p in classifier.parameters() if p.requires_grad)
    print(f"Trainable parameters: {trainable_params:,}")

    optimizer = torch.optim.AdamW(
        classifier.parameters(),
        lr=config.training.learning_rate,
    )

    # ── Training loop ──────────────────────────────────────────────
    os.makedirs(config.training.checkpoint_dir, exist_ok=True)
    os.makedirs(config.training.log_dir, exist_ok=True)

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    log_path = os.path.join(config.training.log_dir, f"train_{timestamp}.jsonl")
    best_val_f1 = 0.0

    print(f"\nStarting training for {config.training.epochs} epochs...")
    print(f"Train: {len(train_subset)} samples | Val: {len(val_subset)} samples\n")

    for epoch in range(1, config.training.epochs + 1):
        classifier.train()
        epoch_loss = 0.0
        num_labeled_batches = 0

        for images, labels in tqdm(train_loader, desc=f"Epoch {epoch}/{config.training.epochs}"):
            # Skip unlabeled samples
            mask = labels != -1
            if mask.sum() == 0:
                continue

            images_labeled = images[mask]
            labels_labeled = labels[mask].to(device)

            # Forward: backbone (frozen) -> classifier (trainable)
            embeddings = backbone.extract(images_labeled)
            logits = classifier(embeddings)
            loss = F.cross_entropy(logits, labels_labeled)

            # Backward
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            epoch_loss += loss.item()
            num_labeled_batches += 1

        avg_loss = epoch_loss / max(num_labeled_batches, 1)

        # ── Validation ──────────────────────────────────────────
        val_metrics = evaluate(backbone, classifier, val_loader, device=device)

        # ── Log ─────────────────────────────────────────────────
        log_entry = {
            "epoch": epoch,
            "train_loss": round(avg_loss, 4),
            "val_accuracy": round(val_metrics["accuracy"], 4),
            "val_macro_f1": round(val_metrics["macro_f1"], 4),
            "val_samples": val_metrics["num_samples"],
        }
        with open(log_path, "a") as f:
            f.write(json.dumps(log_entry) + "\n")

        print(f"  loss={avg_loss:.4f}  val_acc={val_metrics['accuracy']:.4f}  "
              f"val_f1={val_metrics['macro_f1']:.4f}")

        # ── Checkpoint ──────────────────────────────────────────
        ckpt_path = os.path.join(
            config.training.checkpoint_dir,
            f"classifier_epoch_{epoch:03d}.pt",
        )
        save_checkpoint(
            classifier, optimizer, epoch, config,
            dataset.class_names, best_val_f1, ckpt_path,
        )

        if val_metrics["macro_f1"] > best_val_f1:
            best_val_f1 = val_metrics["macro_f1"]
            best_path = os.path.join(
                config.training.checkpoint_dir, "classifier_best.pt"
            )
            save_checkpoint(
                classifier, optimizer, epoch, config,
                dataset.class_names, best_val_f1, best_path,
            )
            print(f"  New best model saved (F1={best_val_f1:.4f})")

    print(f"\nTraining complete. Best val F1: {best_val_f1:.4f}")
    print(f"Logs: {log_path}")
    print(f"Checkpoints: {config.training.checkpoint_dir}")


class _TransformOverride:
    """Applies a different transform to a Subset than its parent dataset uses.

    Works for any dataset that implements `load_raw(idx) -> PIL.Image` and
    `get_label_idx(idx) -> int`. That covers both AquaSiteDataset (full frame)
    and AquaSiteBoxDataset (per-box crop) — the subclass-specific crop/load
    logic lives on the dataset, not here.
    """

    def __init__(self, subset, transform):
        self.subset = subset
        self.transform = transform

    def __len__(self):
        return len(self.subset)

    def __getitem__(self, idx):
        real_idx = self.subset.indices[idx]
        dataset = self.subset.dataset
        img = dataset.load_raw(real_idx)
        img = self.transform(img)
        return img, dataset.get_label_idx(real_idx)
