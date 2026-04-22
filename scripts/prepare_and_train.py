"""
prepare_and_train.py
Splits knife_openimages into train/val, remaps class IDs to match the
multi-class model, then fine-tunes models/best.pt on your RTX 5070.

Run: python scripts/prepare_and_train.py
"""

import os, shutil, random, yaml
from pathlib import Path
from ultralytics import YOLO

# ── Paths ─────────────────────────────────────────────────────────────────────
BASE        = Path(__file__).resolve().parent.parent
SRC_IMGS    = BASE / "data" / "knife_openimages" / "images" / "val"
SRC_LBLS    = BASE / "data" / "knife_openimages" / "labels" / "val"
TRAIN_DIR   = BASE / "dataset" / "train"
VAL_DIR     = BASE / "dataset" / "valid"
DATA_YAML   = BASE / "dataset" / "data.yaml"
MODEL_PATH  = BASE / "models" / "best.pt"

# Class mapping: knife is class 4 in our multi-class model
# knife_openimages has class 0 → remap to 4
CLASS_REMAP = {0: 4}

# Multi-class dataset definition
CLASSES = ["fire", "other", "smoke", "gun", "knife"]

TRAIN_SPLIT = 0.85   # 85% train, 15% val
SEED        = 42

# ── Split images ──────────────────────────────────────────────────────────────
def remap_label(src: Path, dst: Path):
    lines = src.read_text().strip().splitlines()
    remapped = []
    for line in lines:
        parts = line.split()
        cls = int(parts[0])
        cls = CLASS_REMAP.get(cls, cls)
        remapped.append(f"{cls} {' '.join(parts[1:])}")
    dst.write_text("\n".join(remapped))

def prepare():
    images = sorted(SRC_IMGS.glob("*.jpg")) + sorted(SRC_IMGS.glob("*.png"))
    random.seed(SEED)
    random.shuffle(images)
    split = int(len(images) * TRAIN_SPLIT)
    splits = {"train": images[:split], "valid": images[split:]}

    for split_name, imgs in splits.items():
        img_out = BASE / "dataset" / split_name / "images"
        lbl_out = BASE / "dataset" / split_name / "labels"
        img_out.mkdir(parents=True, exist_ok=True)
        lbl_out.mkdir(parents=True, exist_ok=True)

        for img_path in imgs:
            lbl_path = SRC_LBLS / (img_path.stem + ".txt")
            shutil.copy2(img_path, img_out / img_path.name)
            if lbl_path.exists():
                remap_label(lbl_path, lbl_out / lbl_path.name)

        print(f"  {split_name}: {len(imgs)} images")

    # Write data.yaml
    data = {
        "path": str(BASE / "dataset"),
        "train": "train/images",
        "val":   "valid/images",
        "nc":    len(CLASSES),
        "names": CLASSES,
    }
    DATA_YAML.write_text(yaml.dump(data, default_flow_style=False))
    print(f"  data.yaml written → {DATA_YAML}")

# ── Train ─────────────────────────────────────────────────────────────────────
def train():
    base_model = str(MODEL_PATH) if MODEL_PATH.exists() else "yolo11s.pt"
    print(f"\n[Train] base model: {base_model}")

    model = YOLO(base_model)
    model.train(
        data      = str(DATA_YAML),
        epochs    = 80,
        imgsz     = 640,
        batch     = 16,
        device    = 0,          # RTX 5070
        workers   = 4,
        optimizer = "AdamW",
        lr0       = 0.001,
        patience  = 15,         # early stop if no improvement
        freeze    = 10,         # freeze first 10 layers → keep fire/smoke/gun knowledge
        name      = "blackeyes_v2",
        project   = str(BASE / "runs" / "detect"),
        exist_ok  = True,
        cache     = True,
        amp       = True,       # mixed precision — faster on RTX 5070
    )

    best = BASE / "runs" / "detect" / "blackeyes_v2" / "weights" / "best.pt"
    if best.exists():
        dest = BASE / "models" / "best.pt"
        shutil.copy2(best, dest)
        print(f"\n✓ New model saved → {dest}")
    else:
        print("\n⚠ Training finished but best.pt not found — check runs/detect/blackeyes_v2/")

# ── Entry ─────────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    print("── Preparing dataset ─────────────────────────────────")
    prepare()
    print("\n── Starting training on RTX 5070 ─────────────────────")
    train()
