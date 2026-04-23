"""CLI entry point for the AquaSite ML pipeline.

Usage:
    python -m ml.run download                    # Download DINOv3 weights
    python -m ml.run embed                       # Extract embeddings
    python -m ml.run embed --data_source=local   # Override data source
    python -m ml.run train                       # Train classifier
    python -m ml.run train --epochs=50           # Override epochs
    python -m ml.run eval --checkpoint path.pt   # Evaluate a checkpoint
"""

import argparse
import os
import sys


def setup_django():
    """Initialize Django so we can access ORM models."""
    # Add project root to path
    project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    if project_root not in sys.path:
        sys.path.insert(0, project_root)

    os.environ.setdefault("DJANGO_SETTINGS_MODULE", "AquaSightWeb.settings")

    import django
    django.setup()


def cmd_download(args, config):
    """Download DINOv3 model weights to local cache."""
    from ml.models.backbone import DINOv3Backbone
    DINOv3Backbone.download(config.model.name, hf_token=config.hf_token)


def cmd_embed(args, config):
    """Extract embeddings for all images."""
    from ml.training.embed import extract_embeddings
    extract_embeddings(config)


def cmd_train(args, config):
    """Train the classification head."""
    from ml.training.train import train
    train(config)


def cmd_cluster(args, config):
    """Cluster embeddings into groups of similar images."""
    from ml.training.cluster import cluster_embeddings
    cluster_embeddings(config, n_clusters=args.n_clusters)


def cmd_eval(args, config):
    """Evaluate a classifier checkpoint."""
    from ml.data.dataset import build_dataset
    from ml.data.transforms import get_eval_transform
    from ml.models.backbone import DINOv3Backbone
    from ml.models.classifier import load_checkpoint
    from ml.training.evaluate import evaluate

    import torch
    from torch.utils.data import DataLoader

    if not args.checkpoint:
        # Try the best checkpoint
        ckpt_path = os.path.join(config.training.checkpoint_dir, "classifier_best.pt")
        if not os.path.exists(ckpt_path):
            print("No checkpoint found. Run 'train' first.")
            return
    else:
        ckpt_path = args.checkpoint

    print(f"Loading checkpoint: {ckpt_path}")
    classifier, meta = load_checkpoint(ckpt_path, device=config.training.device)
    print(f"Classes: {meta.get('class_names', [])}")

    backbone = DINOv3Backbone(
        model_name=config.model.name,
        device=config.training.device,
        hf_token=config.hf_token,
    )

    dataset = build_dataset(config)
    # Override with eval transform
    from ml.data.transforms import get_eval_transform
    dataset.transform = get_eval_transform(config.model.input_size)

    loader = DataLoader(
        dataset,
        batch_size=config.training.batch_size,
        shuffle=False,
        num_workers=config.training.num_workers,
    )

    metrics = evaluate(backbone, classifier, loader, device=config.training.device)

    print(f"\nEvaluation Results:")
    print(f"  Accuracy:  {metrics['accuracy']:.4f}")
    print(f"  Macro F1:  {metrics['macro_f1']:.4f}")
    print(f"  Samples:   {metrics['num_samples']}")

    if metrics.get("classification_report"):
        from sklearn.metrics import classification_report
        # Print a formatted report
        import json
        print(f"\n{json.dumps(metrics['classification_report'], indent=2)}")


def main():
    parser = argparse.ArgumentParser(
        description="AquaSite ML Pipeline",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    sub = parser.add_subparsers(dest="command", required=True)

    # download
    sub.add_parser("download", help="Download DINOv3 model weights")

    # embed
    p_embed = sub.add_parser("embed", help="Extract embeddings for all images")
    p_embed.add_argument("--data_source", choices=["local", "supabase"])

    # train
    p_train = sub.add_parser("train", help="Train classification head")
    p_train.add_argument("--data_source", choices=["local", "supabase"])
    p_train.add_argument("--epochs", type=int)
    p_train.add_argument("--lr", type=float)
    p_train.add_argument("--batch_size", type=int)

    # cluster
    p_cluster = sub.add_parser("cluster", help="Cluster embeddings into groups")
    p_cluster.add_argument("--n_clusters", type=int, default=20, help="Number of clusters")

    # eval
    p_eval = sub.add_parser("eval", help="Evaluate a checkpoint")
    p_eval.add_argument("--checkpoint", type=str, default=None)
    p_eval.add_argument("--data_source", choices=["local", "supabase"])

    # Common
    for p in [p_embed, p_train, p_eval, p_cluster]:
        p.add_argument("--config", type=str, default=None, help="Path to config.yaml")

    args = parser.parse_args()

    # Setup Django for ORM access
    setup_django()

    # Load config with CLI overrides
    from ml.data.config import load_config
    config_path = getattr(args, "config", None)
    config = load_config(config_path)

    # Apply CLI overrides
    if hasattr(args, "data_source") and args.data_source:
        config.data_source = args.data_source
    if hasattr(args, "epochs") and args.epochs:
        config.training.epochs = args.epochs
    if hasattr(args, "lr") and args.lr:
        config.training.learning_rate = args.lr
    if hasattr(args, "batch_size") and args.batch_size:
        config.training.batch_size = args.batch_size

    # Apply cluster overrides
    if hasattr(args, "n_clusters") and args.n_clusters:
        pass  # handled directly in cmd_cluster

    # Dispatch
    commands = {
        "download": cmd_download,
        "cluster": cmd_cluster,
        "embed": cmd_embed,
        "train": cmd_train,
        "eval": cmd_eval,
    }
    commands[args.command](args, config)


if __name__ == "__main__":
    main()
