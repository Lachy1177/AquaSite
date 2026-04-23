"""Classification heads for DINOv3 embeddings.

Two heads are supported:
  - LinearClassifier: LayerNorm -> Linear(embedding_dim, num_classes).
    Strong baseline; canonical linear probe on frozen backbones.
  - MLPClassifier: LayerNorm -> [Linear -> GELU -> Dropout] x N -> Linear.
    Use when the linear probe plateaus and you have enough labels to
    support a few hundred-k more parameters without overfitting.

Which one `build_classifier` returns is driven by config.classifier.type.
The checkpoint format stores enough metadata to reconstruct either
architecture from a .pt file, so ml/bridge.py does not need to know
which one it's loading.
"""

from typing import List, Tuple

import torch
import torch.nn as nn


class LinearClassifier(nn.Module):
    """LayerNorm + single Linear on top of frozen DINOv3 embeddings."""

    def __init__(self, embedding_dim: int = 768, num_classes: int = 5):
        super().__init__()
        self.head = nn.Sequential(
            nn.LayerNorm(embedding_dim),
            nn.Linear(embedding_dim, num_classes),
        )
        self.num_classes = num_classes

    def forward(self, embeddings: torch.Tensor) -> torch.Tensor:
        return self.head(embeddings)


class MLPClassifier(nn.Module):
    """LayerNorm + MLP on top of frozen DINOv3 embeddings.

    Architecture:
        LayerNorm(embedding_dim)
        -> [Linear(in, h) -> GELU -> Dropout(p)]  for each h in hidden_dims
        -> Linear(last_hidden, num_classes)

    If hidden_dims is empty, reduces to the LinearClassifier structure
    (LayerNorm -> Linear). The factory uses that degenerate case as the
    bridge between "linear probe" and "MLP head".
    """

    def __init__(
        self,
        embedding_dim: int = 768,
        hidden_dims: List[int] = (512, 256),
        num_classes: int = 5,
        dropout: float = 0.3,
    ):
        super().__init__()
        self.embedding_dim = embedding_dim
        self.hidden_dims = list(hidden_dims)
        self.num_classes = num_classes
        self.dropout = dropout

        layers: List[nn.Module] = [nn.LayerNorm(embedding_dim)]
        in_dim = embedding_dim
        for h in self.hidden_dims:
            layers.append(nn.Linear(in_dim, h))
            layers.append(nn.GELU())
            if dropout > 0:
                layers.append(nn.Dropout(dropout))
            in_dim = h
        layers.append(nn.Linear(in_dim, num_classes))

        self.head = nn.Sequential(*layers)

    def forward(self, embeddings: torch.Tensor) -> torch.Tensor:
        return self.head(embeddings)


def build_classifier(config, num_classes: int) -> nn.Module:
    """Construct the classifier head described by config.classifier.

    Reads:
        config.model.embedding_dim
        config.classifier.type            # "linear" or "mlp"
        config.classifier.hidden_dims     # list[int], MLP only
        config.classifier.dropout         # float, MLP only
    """
    head_type = getattr(config.classifier, "type", "linear").lower()
    embedding_dim = config.model.embedding_dim

    if head_type == "linear":
        return LinearClassifier(
            embedding_dim=embedding_dim,
            num_classes=num_classes,
        )
    if head_type == "mlp":
        return MLPClassifier(
            embedding_dim=embedding_dim,
            hidden_dims=config.classifier.hidden_dims,
            num_classes=num_classes,
            dropout=config.classifier.dropout,
        )
    raise ValueError(
        f"Unknown classifier.type={head_type!r}. Expected 'linear' or 'mlp'."
    )


def save_checkpoint(
    classifier: nn.Module,
    optimizer,
    epoch: int,
    config,
    class_names: list,
    best_val_f1: float,
    path: str,
):
    """Save classifier checkpoint with enough metadata to rebuild the head."""
    if isinstance(classifier, MLPClassifier):
        head_type = "mlp"
        hidden_dims = classifier.hidden_dims
        dropout = classifier.dropout
    elif isinstance(classifier, LinearClassifier):
        head_type = "linear"
        hidden_dims = []
        dropout = 0.0
    else:
        raise TypeError(
            f"Unknown classifier class: {type(classifier).__name__}. "
            "Extend save_checkpoint to persist its architecture."
        )

    torch.save({
        "model_state_dict": classifier.state_dict(),
        "optimizer_state_dict": optimizer.state_dict(),
        "epoch": epoch,
        "num_classes": classifier.num_classes,
        "class_names": class_names,
        "best_val_f1": best_val_f1,
        "model_name": config.model.name,
        "embedding_dim": config.model.embedding_dim,
        "classifier_type": head_type,
        "hidden_dims": hidden_dims,
        "dropout": dropout,
    }, path)


def load_checkpoint(path: str, device: str = "cpu") -> Tuple[nn.Module, dict]:
    """Load a classifier checkpoint, reconstructing the architecture it was saved with.

    Falls back to LinearClassifier when 'classifier_type' is missing, so
    older checkpoints written before the MLP head landed still load.
    """
    ckpt = torch.load(path, map_location=device, weights_only=False)

    head_type = ckpt.get("classifier_type", "linear")
    embedding_dim = ckpt["embedding_dim"]
    num_classes = ckpt["num_classes"]

    if head_type == "linear":
        classifier: nn.Module = LinearClassifier(
            embedding_dim=embedding_dim,
            num_classes=num_classes,
        )
    elif head_type == "mlp":
        classifier = MLPClassifier(
            embedding_dim=embedding_dim,
            hidden_dims=ckpt.get("hidden_dims", []),
            num_classes=num_classes,
            dropout=ckpt.get("dropout", 0.0),
        )
    else:
        raise ValueError(f"Unknown classifier_type in checkpoint: {head_type!r}")

    classifier.load_state_dict(ckpt["model_state_dict"])
    classifier.to(device)
    classifier.eval()
    return classifier, ckpt
