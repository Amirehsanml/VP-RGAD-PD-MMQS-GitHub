#!/usr/bin/env python3
from __future__ import annotations

from pathlib import Path
import argparse
import time
import warnings

import pandas as pd
import torch
import torch.nn.functional as F
from PIL import Image
from tqdm import tqdm

try:
    import open_clip
except ImportError as e:
    raise ImportError(
        "open_clip_torch is required. Install it with:\n"
        "pip install open_clip_torch==2.23.0"
    ) from e


PROJECT_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_DATA_CSV = PROJECT_ROOT / "MMQS/Dataset/multimodal_final_updated.csv"
DEFAULT_IMAGE_ROOT = Path("/home/user2/ehsan/sajjad_version/mdcrapn_implementation/data/mmqsd")

TARGET_DIM = 768

# Correct HF-hub identifier for BiomedCLIP
BIOMEDCLIP_ID = "hf-hub:microsoft/BiomedCLIP-PubMedBERT_256-vit_base_patch16_224"


def resolve_image_path(image_root: Path, rel_path: str) -> Path:
    rel = str(rel_path).strip().replace("\\", "/").lstrip("./")
    rel = rel.replace("Multimodal_images/Multimodal_images/", "Multimodal_images/")
    return image_root / rel


def pad_or_truncate(x: torch.Tensor, target_dim: int = TARGET_DIM) -> torch.Tensor:
    x = x.flatten().float()
    n = x.numel()
    if n == target_dim:
        return x
    if n > target_dim:
        return x[:target_dim]
    return F.pad(x, (0, target_dim - n))


def ensure_dir(path: Path) -> None:
    path.mkdir(parents=True, exist_ok=True)


def load_biomedclip(device: torch.device):
    print("[STEP] Loading BiomedCLIP ...")
    print(f"[INFO] model_id: {BIOMEDCLIP_ID}")

    model, _, preprocess = open_clip.create_model_and_transforms(BIOMEDCLIP_ID)
    model.eval().to(device)
    return model, preprocess


def encode_biomedclip(model, preprocess, pil_img: Image.Image, device: torch.device) -> torch.Tensor:
    with torch.no_grad():
        img_tensor = preprocess(pil_img).unsqueeze(0).to(device)
        feats = model.encode_image(img_tensor).squeeze(0)
    return feats


def main():
    parser = argparse.ArgumentParser(description="Build BiomedCLIP embeddings for MMQS.")
    parser.add_argument("--data_csv", type=str, default=str(DEFAULT_DATA_CSV))
    parser.add_argument("--image_root", type=str, default=str(DEFAULT_IMAGE_ROOT))
    parser.add_argument("--output_root", type=str, default="embeddings")
    parser.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")
    args = parser.parse_args()

    warnings.filterwarnings("ignore", category=UserWarning)

    data_csv = Path(args.data_csv)
    image_root = Path(args.image_root)
    output_root = Path(args.output_root)
    device = torch.device(args.device)

    print("🔥 START BIOMEDCLIP EMBEDDING PIPELINE 🔥")
    print(f"[INFO] data_csv: {data_csv}")
    print(f"[INFO] image_root: {image_root}")
    print(f"[INFO] output_root: {output_root}")
    print(f"[INFO] device: {device}")

    df = pd.read_csv(data_csv)
    print(f"[INFO] dataset rows: {len(df)}")
    print(f"[INFO] unique relative paths: {df['relative_path'].nunique()}")

    image_paths = [resolve_image_path(image_root, p) for p in df["relative_path"].astype(str).tolist()]

    print("[INFO] sample path:", image_paths[0])
    print("[INFO] sample exists:", image_paths[0].exists())

    model, preprocess = load_biomedclip(device)

    out_path = output_root / "biomedclip" / "embeddings" / "biomedclip_embeddings.pt"
    manifest_path = out_path.with_suffix(".manifest.csv")
    ensure_dir(out_path.parent)

    all_embeds = []
    rows = []
    n_missing = 0
    n_failed = 0
    start = time.time()

    for i, path in enumerate(tqdm(image_paths, desc="BiomedCLIP", dynamic_ncols=True)):
        if i % 500 == 0 and i > 0:
            print(f"[BiomedCLIP] progress {i}/{len(image_paths)}")

        try:
            pil_img = Image.open(path).convert("RGB")
        except Exception as e:
            n_missing += 1
            n_failed += 1
            print(f"[WARN] failed image: {path} | {e}")
            all_embeds.append(torch.zeros(TARGET_DIM))
            rows.append({"idx": i, "path": str(path), "status": "failed_open", "error": str(e)})
            continue

        try:
            raw = encode_biomedclip(model, preprocess, pil_img, device)
            raw = torch.nan_to_num(raw, nan=0.0, posinf=0.0, neginf=0.0)
            raw = pad_or_truncate(raw, TARGET_DIM)
            all_embeds.append(raw.cpu())
            rows.append({"idx": i, "path": str(path), "status": "ok", "error": ""})
        except Exception as e:
            n_failed += 1
            print(f"[WARN] encoding failed: {path} | {e}")
            all_embeds.append(torch.zeros(TARGET_DIM))
            rows.append({"idx": i, "path": str(path), "status": "failed_encode", "error": str(e)})

    all_embeds = torch.stack(all_embeds, dim=0)
    torch.save(all_embeds, out_path)
    pd.DataFrame(rows).to_csv(manifest_path, index=False)

    elapsed = time.time() - start
    print("\n[DONE] BiomedCLIP")
    print(f"[INFO] shape: {tuple(all_embeds.shape)}")
    print(f"[INFO] missing/open failures: {n_missing}")
    print(f"[INFO] encode failures: {n_failed}")
    print(f"[INFO] time: {elapsed/60:.2f} min")
    print(f"[INFO] saved: {out_path}")
    print(f"[INFO] manifest: {manifest_path}")


if __name__ == "__main__":
    main()