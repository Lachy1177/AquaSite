import json
import os
from datetime import datetime

import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
from tqdm import tqdm

from ml.data.dataset import build_dataset
from ml.data.transforms import get_eval_transform, get_train_transform
from ml.models.backbone import DINOv3Backbone
from ml.models.classifier import build_classifier, load_checkpoint, save_checkpoint
from ml.training.evaluate import evaluate


def train(config) -> None:
    """Run the full training pipeline: split → train → pick best → test.

    The function is one long sequence of steps rather than smaller helpers
    deliberately — it's the canonical training script and you can read it
    top-to-bottom to see the whole pipeline at a glance.
    """
    device = config.training.device
    print(f"Device: {device}")

    # Build the dataset
    # reads BoundingBox rows from the Postgres DB via Django
    # ORM, downloads each parent frame from Supabase (cached locally), applies
    # the min_box_fraction area filter, and returns an AquaSiteBoxDataset
    # where each sample is one per-box crop with one taxonomy label.
    dataset = build_dataset(config)

    # If no labels exist yet we can't compute a loss. Fall through to
    # embedding extraction so the user has at least something useful.
    if not dataset.has_labels:
        _warn_no_labels_and_run_embed(config)
        return

    print(f"Classes ({dataset.num_classes}): {dataset.class_names}")

    # The classifier MLP
    if config.classifier.type != "mlp":
        raise ValueError(
            f"This training loop targets MLP only. "
            f"Got classifier.type={config.classifier.type!r}. "
            f"Set classifier.type: 'mlp' in ml/config.yaml."
        )

    # ─tratified train / val / test split 
    # Three sets, three different jobs:
    #   - train: the optimiser sees these every epoch and updates weights.
    #   - val:   evaluated *after* each epoch to decide which checkpoint
    #            is "best". The model never trains on val, but val IS used
    #            for model selection — so it can leak slightly into the
    #            final choice of which weights ship.
    #   - test:  held out entirely until training finishes. Used exactly
    #            ONCE at the end. The score on this set is the honest
    #            measure of generalisation, because no decision during
    #            training depended on it.
    train_subset, val_subset, test_subset = dataset.stratified_split(
        val_frac=config.training.val_frac,
        test_frac=config.training.test_frac,
    )
    print(
        f"Split: train={len(train_subset)}  "
        f"val={len(val_subset)}  "
        f"test={len(test_subset)}  "
        f"(val_frac={config.training.val_frac}, test_frac={config.training.test_frac})"
    )

    # Apply per-split transforms 
    # Train uses augmentations (flips, colour jitter) to help the head
    # generalise.
    train_dataset = _TransformOverride(
        train_subset, get_train_transform(config.model.input_size)
    )
    val_dataset = _TransformOverride(
        val_subset, get_eval_transform(config.model.input_size)
    )
    test_dataset = _TransformOverride(
        test_subset, get_eval_transform(config.model.input_size)
    )

    # DataLoaders 
    # `shuffle=True` only on train — for val/test you want a stable order
    # so the same sample is always sample N (useful for debugging).
    # pin_memory only helps on CUDA; on CPU it's a no-op.
    loader_kwargs = dict(
        batch_size=config.training.batch_size,
        num_workers=config.training.num_workers,
        pin_memory=(device == "cuda"),
    )
    train_loader = DataLoader(train_dataset, shuffle=True, **loader_kwargs)
    val_loader = DataLoader(val_dataset, shuffle=False, **loader_kwargs)
    test_loader = DataLoader(test_dataset, shuffle=False, **loader_kwargs)

    # Build the models 
    # Backbone: DINOv3 ViT-B/16. Constructor sets requires_grad=False on
    # every parameter and calls .eval() — i.e. the backbone is frozen.
    backbone = DINOv3Backbone(
        model_name=config.model.name,
        device=device,
        hf_token=config.hf_token,
    )
    # Head: tiny MLP that maps the backbone's 768-d output to num_classes
    # logits. This is the *only* thing that learns.
    classifier = build_classifier(config, num_classes=dataset.num_classes).to(device)
    print(f"Classifier: {classifier.__class__.__name__}")
    trainable_params = sum(p.numel() for p in classifier.parameters() if p.requires_grad)
    print(f"Trainable parameters: {trainable_params:,}")

    # Optimiser 
    # classifier.parameters() — NOT the backbone's — so only the
    # MLP head's weights ever get updated.
    optimizer = torch.optim.AdamW(
        classifier.parameters(),
        lr=config.training.learning_rate,
    )

    # ── 8. Set up output paths ────────────────────────────────────────────
    os.makedirs(config.training.checkpoint_dir, exist_ok=True)
    os.makedirs(config.training.log_dir, exist_ok=True)
    run_timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    log_path = os.path.join(config.training.log_dir, f"train_{run_timestamp}.jsonl")
    best_checkpoint_path = os.path.join(
        config.training.checkpoint_dir, "classifier_best.pt"
    )

    # ── 9. Training loop ──────────────────────────────────────────────────
    print(f"\nStarting training for {config.training.epochs} epochs...")
    best_val_macro_f1 = 0.0  # tracks the best score seen so far

    for epoch in range(1, config.training.epochs + 1):
        #One epoch of training 
        # .train() enables training-mode behaviour — in our MLP that means
        # dropout is active (drops random activations) so the model can't
        # rely on any specific neuron.
        classifier.train()
        epoch_loss_sum = 0.0
        labelled_batches_seen = 0

        for images, labels in tqdm(train_loader, desc=f"Epoch {epoch}/{config.training.epochs}"):
            label_mask = labels != -1
            if label_mask.sum() == 0:
                continue
            images_labelled = images[label_mask]
            labels_labelled = labels[label_mask].to(device)

            # Forward pass step 1: pixels → 768-d embeddings.
            embeddings = backbone.extract(images_labelled)
            # Forward pass embeddings → logits over the classes.
            # The classifier IS in autograd-tracked mode, so we can backprop.
            logits = classifier(embeddings)
            # Cross-entropy combines softmax + negative log likelihood then returns the mean loss over the batch.
            loss = F.cross_entropy(logits, labels_labelled)

            # Standard PyTorch step: zero old gradients, compute new ones
            # via backprop, take an optimiser step. Only the classifier's
            # params have requires_grad=True so only they update.
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            epoch_loss_sum += loss.item()
            labelled_batches_seen += 1

        # Average loss across all batches that had at least one labelled sample.
        # max(..., 1) just guards against divide-by-zero on an empty epoch.
        avg_train_loss = epoch_loss_sum / max(labelled_batches_seen, 1)

        #Validation pass
        # Returns accuracy and
        # macro-F1. 
        val_metrics = evaluate(backbone, classifier, val_loader, device=device)

        # Log this epoch 
        # One JSON object per line so the file is easy to load with
        # `pandas.read_json(path, lines=True)` later.
        log_entry = {
            "epoch": epoch,
            "train_loss": round(avg_train_loss, 4),
            "val_accuracy": round(val_metrics["accuracy"], 4),
            "val_macro_f1": round(val_metrics["macro_f1"], 4),
            "val_samples": val_metrics["num_samples"],
        }
        with open(log_path, "a") as f:
            f.write(json.dumps(log_entry) + "\n")
        print(
            f"  loss={avg_train_loss:.4f}  "
            f"val_acc={val_metrics['accuracy']:.4f}  "
            f"val_f1={val_metrics['macro_f1']:.4f}"
        )

        # Per-epoch checkpoint + best-tracker
        # Always write a per-epoch checkpoint so we can recover any specific
        # epoch later (e.g. for ablations or sanity checks).
        epoch_ckpt_path = os.path.join(
            config.training.checkpoint_dir,
            f"classifier_epoch_{epoch:03d}.pt",
        )
        save_checkpoint(
            classifier, optimizer, epoch, config,
            dataset.class_names, best_val_macro_f1, epoch_ckpt_path,
        )
        # If this epoch beat the best val score so far, ALSO overwrite
        # `classifier_best.pt`. This is the file the inference bridge
        # (ml/bridge.py) loads at serve time.
        if val_metrics["macro_f1"] > best_val_macro_f1:
            best_val_macro_f1 = val_metrics["macro_f1"]
            save_checkpoint(
                classifier, optimizer, epoch, config,
                dataset.class_names, best_val_macro_f1, best_checkpoint_path,
            )
            print(f"  → new best (val F1={best_val_macro_f1:.4f}) saved to classifier_best.pt")

    print(f"\nTraining complete. Best val macro-F1: {best_val_macro_f1:.4f}")

    # Final held-out test evaluation 
    # Reload the best checkpoint (the one with the highest val F1) and run
    # it ONCE on the held-out test set. This is the only honest score we
    # report — every other number above was used in some way to make a
    # decision (which epoch is best), so it has a tiny optimistic bias.
    print("\nEvaluating best checkpoint on held-out TEST set...")
    best_classifier, _ = load_checkpoint(best_checkpoint_path, device=device)
    test_metrics = evaluate(backbone, best_classifier, test_loader, device=device)
    print(
        f"  test_acc={test_metrics['accuracy']:.4f}  "
        f"test_f1={test_metrics['macro_f1']:.4f}  "
        f"(n={test_metrics['num_samples']})"
    )

    # Append a final TEST line to the log so the score lives next to the
    # per-epoch history.
    with open(log_path, "a") as f:
        f.write(json.dumps({
            "epoch": "TEST",
            "test_accuracy": round(test_metrics["accuracy"], 4),
            "test_macro_f1": round(test_metrics["macro_f1"], 4),
            "test_samples": test_metrics["num_samples"],
            "best_val_macro_f1": round(best_val_macro_f1, 4),
        }) + "\n")

    print(f"\nLogs:        {log_path}")
    print(f"Checkpoints: {config.training.checkpoint_dir}")


def _warn_no_labels_and_run_embed(config) -> None:
    """Print the no-labels warning and fall through to embedding extraction.

    Triggered when `build_dataset` returns a dataset with zero labels — for
    example a fresh database with images uploaded but no boxes drawn yet.
    Embedding extraction still works without labels and is useful for the
    DINOv3 clustering step downstream.
    """
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


class _TransformOverride:
    """Lets a torch Subset use a different transform than its parent dataset.

    Why this exists
    ---------------
    `torch.utils.data.Subset` is just a thin wrapper around (parent, indices)
    — it forwards __getitem__ straight to the parent, which means the parent's
    transform is what runs. But we want train to use augmentations and
    val/test to use a deterministic eval transform, even though they share
    the same underlying records list. This adapter intercepts __getitem__,
    calls the parent's raw-load + label-lookup methods, then applies its
    OWN transform.

    The wrapped dataset must implement:
        - load_raw(idx)      -> PIL.Image  (untransformed image)
        - get_label_idx(idx) -> int        (or -1 for unlabeled)
    Both AquaSiteDataset and AquaSiteBoxDataset satisfy this interface.
    """

    def __init__(self, subset, transform):
        self.subset = subset
        self.transform = transform

    def __len__(self):
        return len(self.subset)

    def __getitem__(self, idx):
        # Map subset-local idx -> parent-dataset idx via the indices list
        # Subset stored at construction time.
        parent_idx = self.subset.indices[idx]
        parent = self.subset.dataset
        image = parent.load_raw(parent_idx)
        image = self.transform(image)
        return image, parent.get_label_idx(parent_idx)
