#!/usr/bin/env python3
"""
Download YOLOv8n + SAM2 Hiera-L weights into graduate_pro vision_ai models/ (paths match enhanced_detection_config.json).

Usage (repo root, conda env g1_env):
    python scripts/download_vision_weights.py
"""
from __future__ import annotations

import os
import sys
import urllib.request
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
MODELS = (
    REPO_ROOT
    / "graduate_pro"
    / "src"
    / "vision_ai"
    / "vision_ai"
    / "models"
)
YOLO_DIR = MODELS / "yolo"
SAM2_DIR = MODELS / "sam2"

YOLO_URL = "https://github.com/ultralytics/assets/releases/download/v8.3.0/yolov8n.pt"
SAM2_URL = "https://dl.fbaipublicfiles.com/segment_anything_2/072824/sam2_hiera_large.pt"


def download(url: str, dest: Path, desc: str) -> None:
    dest.parent.mkdir(parents=True, exist_ok=True)
    if dest.is_file() and dest.stat().st_size > 1_000_000:
        print(f"[OK] {desc} already exists: {dest} ({dest.stat().st_size // 1_000_000} MB)")
        return
    print(f"[DL] {desc}\n      {url}\n  -> {dest}")
    tmp = dest.with_suffix(dest.suffix + ".part")
    try:
        urllib.request.urlretrieve(url, tmp)
        tmp.replace(dest)
        print(f"[OK] Saved {dest} ({dest.stat().st_size // 1_000_000} MB)")
    except Exception as e:
        if tmp.is_file():
            tmp.unlink(missing_ok=True)
        print(f"[FAIL] {desc}: {e}")
        raise


def main() -> int:
    os.chdir(REPO_ROOT)
    print(f"Models directory: {MODELS}")
    try:
        download(YOLO_URL, YOLO_DIR / "yolov8n.pt", "YOLOv8n")
    except Exception:
        print("Tip: pip install ultralytics && python -c \"from ultralytics import YOLO; YOLO('yolov8n.pt')\"")
        return 1
    try:
        download(SAM2_URL, SAM2_DIR / "sam2_hiera_large.pt", "SAM2 Hiera-L")
    except Exception:
        print("Tip: manually download SAM2 checkpoint from https://github.com/facebookresearch/segment-anything-2")
        return 1
    print("\nNext: conda activate g1_env && pip install -r requirements-g1-env.txt")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
