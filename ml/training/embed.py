"""Extract DINOv3 embeddings for all images in the dataset."""

import json
import os

import numpy as np
import torch
from torch.utils.data import DataLoader
from tqdm import tqdm

from ml.data.transforms import get_eval_transform
from ml.models.backbone import DINOv3Backbone


def extract_embeddings(config):
    """Extract embeddings for every image and save to disk.

    Outputs:
        embeddings.npy  -- (N, 768) float32 array
        filenames.json  -- list of N filenames in matching order
    """
    # Load data records (no transform yet -- we apply eval transform)
    if config.data_source == "local":
        from ml.data.local_loader import load_local_images
        records = load_local_images(
            image_dirs=config.local.image_dirs,
            metadata_csv=config.local.metadata_csv,
            labels_csv=config.local.labels_csv,
        )
    else:
        from ml.data.supabase_loader import load_supabase_images
        records = load_supabase_images(
            use_django_media=config.supabase.use_django_media,
            download_to=config.supabase.download_to,
            supabase_url=config.supabase_url,
            supabase_key=config.supabase_key,
        )

    if not records:
        print("No images found. Check your config.yaml paths.")
        return

    # Build a simple dataset with eval transforms
    from ml.data.dataset import AquaSiteDataset
    transform = get_eval_transform(config.model.input_size)
    dataset = AquaSiteDataset(records, transform=transform)

    loader = DataLoader(
        dataset,
        batch_size=config.embedding.batch_size,
        shuffle=False,
        num_workers=config.training.num_workers,
        pin_memory=(config.training.device == "cuda"),
    )

    # Load backbone
    backbone = DINOv3Backbone(
        model_name=config.model.name,
        device=config.training.device,
        hf_token=config.hf_token,
    )

    # Extract
    all_embeddings = []
    filenames = []

    print(f"Extracting embeddings for {len(dataset)} images...")
    for images, _ in tqdm(loader, desc="Embedding"):
        emb = backbone.extract(images)
        all_embeddings.append(emb.cpu().numpy())

    # Collect filenames in order
    for rec in records:
        filenames.append(rec.filename)

    embeddings_array = np.concatenate(all_embeddings, axis=0)
    print(f"Embeddings shape: {embeddings_array.shape}")

    # Save
    os.makedirs(config.embedding.output_dir, exist_ok=True)
    emb_path = os.path.join(config.embedding.output_dir, "embeddings.npy")
    names_path = os.path.join(config.embedding.output_dir, "filenames.json")

    np.save(emb_path, embeddings_array)
    with open(names_path, "w") as f:
        json.dump(filenames, f)

    print(f"Saved embeddings to {emb_path}")
    print(f"Saved filenames to {names_path}")
