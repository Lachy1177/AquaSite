"""Cluster DINOv3 embeddings to group similar images.

Uses K-means to assign each image to a cluster based on its
768-dim embedding vector. Similar underwater scenes end up together.
"""

import json
import os

import numpy as np
from sklearn.cluster import KMeans
from sklearn.preprocessing import normalize


def cluster_embeddings(config, n_clusters: int = 20):
    """Load embeddings and run K-means clustering.

    Args:
        config: MLConfig instance.
        n_clusters: Number of clusters to create.

    Outputs:
        clusters.json -- {filename: cluster_id} mapping
        cluster_summary.json -- per-cluster stats
    """
    emb_path = os.path.join(config.embedding.output_dir, "embeddings.npy")
    names_path = os.path.join(config.embedding.output_dir, "filenames.json")

    if not os.path.exists(emb_path):
        print(f"Embeddings not found at {emb_path}")
        print("Run 'python -m ml.run embed' first.")
        return None

    # Load
    embeddings = np.load(emb_path)
    with open(names_path, "r") as f:
        filenames = json.load(f)

    print(f"Loaded {len(filenames)} embeddings, shape {embeddings.shape}")

    # L2-normalize for better cosine-like clustering
    embeddings_norm = normalize(embeddings, norm="l2")

    # K-means
    print(f"Running K-means with {n_clusters} clusters...")
    kmeans = KMeans(n_clusters=n_clusters, random_state=42, n_init=10)
    labels = kmeans.fit_predict(embeddings_norm)

    # Build mapping
    cluster_map = {}
    for fname, label in zip(filenames, labels):
        cluster_map[fname] = int(label)

    # Cluster summary
    summary = {}
    for i in range(n_clusters):
        members = [f for f, c in cluster_map.items() if c == i]
        # Find the image closest to the centroid (representative image)
        mask = labels == i
        if mask.sum() > 0:
            cluster_embs = embeddings_norm[mask]
            centroid = kmeans.cluster_centers_[i]
            dists = np.linalg.norm(cluster_embs - centroid, axis=1)
            closest_idx = np.where(mask)[0][np.argmin(dists)]
            representative = filenames[closest_idx]
        else:
            representative = None

        summary[str(i)] = {
            "count": len(members),
            "representative": representative,
        }

    # Save
    clusters_path = os.path.join(config.embedding.output_dir, "clusters.json")
    summary_path = os.path.join(config.embedding.output_dir, "cluster_summary.json")

    with open(clusters_path, "w") as f:
        json.dump(cluster_map, f)
    with open(summary_path, "w") as f:
        json.dump(summary, f, indent=2)

    print(f"\nCluster distribution:")
    for i in range(n_clusters):
        count = summary[str(i)]["count"]
        bar = "#" * (count // 20)
        print(f"  Cluster {i:2d}: {count:5d} images  {bar}")

    print(f"\nSaved cluster assignments to {clusters_path}")
    return cluster_map
