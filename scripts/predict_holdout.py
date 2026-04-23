#!/usr/bin/env python3
"""
Diagnose a trained classifier head.

  1. Loads classifier_best.pt and prints its metadata.
  2. Runs per-box inference on every labeled BoundingBox in the DB and
     prints a confusion matrix + per-class metrics.
     (Caveat: without a seeded train/val split, we cannot reproduce the
     specific val subset the training loop used. So this matrix is over
     ALL labeled boxes — a mix of train + val. Treat as "did the model
     learn the labels at all", not generalization.)
  3. Runs inference on the held-out unlabeled frames in extracted_frames/
     (the ones NOT selected by dino_pipeline.py's K-means curation) and
     writes a CSV ranked by model uncertainty (entropy). Use it as the
     next active-learning annotation queue.
     (Caveat: this is whole-frame inference on a model trained on crops.
     Numbers are directional, not calibrated.)

Usage:
  python scripts/predict_holdout.py
  python scripts/predict_holdout.py --frames-dir extracted_frames --top-k 30
"""

import argparse
import csv
import math
import os
import sys
from pathlib import Path

SCRIPT_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = SCRIPT_DIR.parent
sys.path.insert(0, str(PROJECT_ROOT))

os.environ.setdefault("DJANGO_SETTINGS_MODULE", "AquaSightWeb.settings")
import django
django.setup()

from django.conf import settings  # noqa: E402

parser = argparse.ArgumentParser(
    description="Evaluate the trained classifier on labeled boxes + held-out frames",
    formatter_class=argparse.RawDescriptionHelpFormatter,
)
parser.add_argument(
    "--checkpoint",
    type=str,
    default=str(PROJECT_ROOT / "ml" / "outputs" / "checkpoints" / "classifier_best.pt"),
)
parser.add_argument(
    "--frames-dir",
    type=str,
    default=str(PROJECT_ROOT / "extracted_frames"),
    help="Directory with unlabeled held-out JPGs",
)
parser.add_argument(
    "--top-k",
    type=int,
    default=30,
    help="Print the top-K most uncertain held-out frames (by entropy)",
)
parser.add_argument(
    "--out-csv",
    type=str,
    default=str(PROJECT_ROOT / "ml" / "outputs" / "holdout_predictions.csv"),
)
args = parser.parse_args()


print("Loading dependencies...")
import numpy as np  # noqa: E402
import torch  # noqa: E402
import torch.nn.functional as F  # noqa: E402
from PIL import Image  # noqa: E402

from ml.data.config import load_config  # noqa: E402
from ml.data.dataset import AquaSiteBoxDataset, load_box_records  # noqa: E402
from ml.data.transforms import get_eval_transform  # noqa: E402
from ml.models.backbone import DINOv3Backbone  # noqa: E402
from ml.models.classifier import load_checkpoint  # noqa: E402


config = load_config()
device = config.training.device
print(f"  Device: {device}")


# ─── 1. Checkpoint metadata ────────────────────────────────────────────────

print()
print("=" * 60)
print("1. CHECKPOINT METADATA")
print("=" * 60)
if not Path(args.checkpoint).exists():
    print(f"  ERROR: {args.checkpoint} not found. Run `python -m ml.run train` first.")
    sys.exit(1)

classifier, meta = load_checkpoint(args.checkpoint, device=device)
classifier.eval()
print(f"  Path            : {args.checkpoint}")
print(f"  Class           : {classifier.__class__.__name__}")
print(f"  Hidden dims     : {meta.get('hidden_dims')}")
print(f"  Dropout         : {meta.get('dropout')}")
print(f"  Best val F1     : {meta.get('best_val_f1'):.4f}")
print(f"  Best epoch      : {meta.get('epoch')}")
print(f"  num_classes     : {meta.get('num_classes')}")
print(f"  class_names     : {meta.get('class_names')}")

class_names = meta["class_names"]
class_to_idx = {name: i for i, name in enumerate(class_names)}

# Load backbone (frozen, 768-d CLS)
print()
print("Loading DINOv3 backbone...")
backbone = DINOv3Backbone(
    model_name=config.model.name,
    device=device,
    hf_token=config.hf_token,
)
transform = get_eval_transform(config.model.input_size)


# ─── 2. Confusion matrix on labeled box crops ──────────────────────────────

print()
print("=" * 60)
print("2. PREDICTIONS ON LABELED BOX CROPS (training + val mixed)")
print("=" * 60)
print("  (includes training data; shows what the model has 'memorised' —")
print("   not a clean generalisation estimate)")
print()

records = load_box_records(config)
ds = AquaSiteBoxDataset(records, transform=transform)

# Class vocabulary sanity: classifier was trained with meta['class_names'],
# but the DB may have gained new labels since then. Warn if mismatched.
current = set(ds.class_names)
saved = set(class_names)
if current - saved:
    print(f"  WARNING: DB has new labels since training: {sorted(current - saved)}")
    print(f"           These boxes will be dropped from this diagnostic.")
if saved - current:
    print(f"  NOTE: classifier knows labels not in current DB: {sorted(saved - current)}")

y_true, y_pred, y_conf = [], [], []
with torch.no_grad():
    for i in range(len(ds)):
        true_label = ds.records[i].label
        if true_label not in class_to_idx:
            continue  # DB-side new label not in checkpoint vocab
        tensor, _ = ds[i]
        tensor = tensor.unsqueeze(0)
        emb = backbone.extract(tensor)
        logits = classifier(emb.to(device))
        probs = F.softmax(logits, dim=1)[0].cpu().numpy()
        pred_idx = int(probs.argmax())
        y_true.append(class_to_idx[true_label])
        y_pred.append(pred_idx)
        y_conf.append(float(probs[pred_idx]))

print(f"  Evaluated on {len(y_true)} labeled boxes")
if y_true:
    from sklearn.metrics import (
        accuracy_score, f1_score, confusion_matrix, classification_report,
    )
    acc = accuracy_score(y_true, y_pred)
    macro_f1 = f1_score(y_true, y_pred, average="macro", zero_division=0)
    print(f"  Accuracy        : {acc:.4f}  ({sum(t == p for t, p in zip(y_true, y_pred))}/{len(y_true)})")
    print(f"  Macro F1        : {macro_f1:.4f}")
    print(f"  Avg confidence  : {sum(y_conf)/len(y_conf):.4f}  (higher = more confident)")

    print()
    print("  Per-class performance:")
    report = classification_report(
        y_true, y_pred, zero_division=0, output_dict=True,
        labels=list(range(len(class_names))),
        target_names=class_names,
    )
    print(f"    {'class':<20s} {'precision':>9s} {'recall':>9s} {'f1':>6s} {'support':>7s}")
    for name in class_names:
        r = report.get(name, {})
        sup = int(r.get('support', 0))
        if sup == 0:
            continue
        print(f"    {name:<20s} {r['precision']:>9.3f} {r['recall']:>9.3f} {r['f1-score']:>6.3f} {sup:>7d}")

    # Compact confusion matrix — print only classes that actually appear
    present = sorted(set(y_true) | set(y_pred))
    cm = confusion_matrix(y_true, y_pred, labels=present)
    short_names = [class_names[i][:10] for i in present]
    print()
    print("  Confusion matrix (rows=true, cols=pred):")
    header = "              " + " ".join(f"{n:>10s}" for n in short_names)
    print(header)
    for i, row in enumerate(cm):
        print(f"    {short_names[i]:>10s}  " + " ".join(f"{v:>10d}" for v in row))


# ─── 3. Inference on held-out unlabeled frames ─────────────────────────────

frames_dir = Path(args.frames_dir)
print()
print("=" * 60)
print("3. INFERENCE ON HELD-OUT UNLABELED FRAMES")
print("=" * 60)
print("  (whole-frame inference with a crop-trained model — directional only)")
if not frames_dir.exists() or not any(frames_dir.glob("*.jpg")):
    print(f"  No frames found in {frames_dir}. Skipping.")
    sys.exit(0)

# Skip frames that were curated into the DB (we'd be re-testing training-adjacent data)
from annotator.models import AnalysisImage  # noqa: E402
curated_names = set(AnalysisImage.objects.values_list("original_filename", flat=True))

jpgs = sorted(str(p) for p in frames_dir.glob("*.jpg") if p.name not in curated_names)
print(f"  Found {len(jpgs)} held-out (uncurated) frames")

rows = []
with torch.no_grad():
    for i, path in enumerate(jpgs):
        try:
            img = Image.open(path).convert("RGB")
        except Exception as e:
            print(f"  WARNING: could not open {path}: {e}")
            continue
        tensor = transform(img).unsqueeze(0)
        emb = backbone.extract(tensor)
        logits = classifier(emb.to(device))
        probs = F.softmax(logits, dim=1)[0].cpu().numpy()
        top_idx = int(probs.argmax())
        # Shannon entropy — high = uncertain across all classes, low = peaked on one
        entropy = float(-sum(p * math.log(p + 1e-12) for p in probs))
        rows.append({
            "filename": Path(path).name,
            "top1_label": class_names[top_idx],
            "top1_confidence": round(float(probs[top_idx]), 4),
            "entropy": round(entropy, 4),
            **{f"p_{n}": round(float(probs[i]), 4) for i, n in enumerate(class_names)},
        })
        if (i + 1) % 100 == 0:
            print(f"  {i+1}/{len(jpgs)}")

print(f"  Done — {len(rows)} predictions")

# CSV dump
out = Path(args.out_csv)
out.parent.mkdir(parents=True, exist_ok=True)
with open(out, "w", newline="", encoding="utf-8") as f:
    w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
    w.writeheader()
    w.writerows(rows)
print(f"  Full CSV         : {out}")

# Summary stats
from collections import Counter
top1_counts = Counter(r["top1_label"] for r in rows)
print()
print("  Top-1 prediction distribution over held-out frames:")
for label, n in top1_counts.most_common():
    print(f"    {label:<20s}  {n:>4}  ({n/len(rows)*100:.1f}%)")

conf_vals = [r["top1_confidence"] for r in rows]
conf_vals.sort()
print()
print("  Confidence distribution (percentiles):")
for pct in [10, 25, 50, 75, 90]:
    idx = int(len(conf_vals) * pct / 100)
    print(f"    p{pct:>2}: {conf_vals[idx]:.3f}")

# Active learning queue: most uncertain frames (highest entropy)
rows_sorted = sorted(rows, key=lambda r: -r["entropy"])
print()
print(f"  Top-{args.top_k} most uncertain frames (candidate annotation queue):")
print(f"    {'filename':<45s} {'entropy':>7s} {'top1':>18s} {'conf':>6s}")
for r in rows_sorted[:args.top_k]:
    print(f"    {r['filename']:<45s} {r['entropy']:>7.3f} {r['top1_label']:>18s} {r['top1_confidence']:>6.3f}")
