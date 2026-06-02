from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd
from PIL import Image
from tqdm import tqdm

from src.models.vision_backbones import EMBEDDING_MODELS
from src.utils.io import list_images


def extract_transformers_embeddings(image_paths: list[Path], model_name: str, batch_size: int) -> np.ndarray:
    import torch
    from transformers import AutoImageProcessor, AutoModel

    device = "cuda" if torch.cuda.is_available() else "cpu"
    processor = AutoImageProcessor.from_pretrained(model_name)
    model = AutoModel.from_pretrained(model_name).to(device).eval()
    vectors = []
    for start in tqdm(range(0, len(image_paths), batch_size), desc=model_name):
        batch_paths = image_paths[start : start + batch_size]
        images = [Image.open(path).convert("RGB") for path in batch_paths]
        inputs = processor(images=images, return_tensors="pt").to(device)
        with torch.no_grad(), torch.autocast(device_type=device, enabled=device == "cuda"):
            output = model(**inputs)
            if hasattr(output, "pooler_output") and output.pooler_output is not None:
                emb = output.pooler_output
            else:
                emb = output.last_hidden_state.mean(dim=1)
        vectors.append(emb.float().cpu().numpy())
    return np.concatenate(vectors, axis=0)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-dir", type=Path, default=Path("data"))
    parser.add_argument("--out-dir", type=Path, default=Path("data/embeddings"))
    parser.add_argument("--model", choices=sorted(EMBEDDING_MODELS), default="dinov2")
    parser.add_argument("--batch-size", type=int, default=16)
    args = parser.parse_args()

    image_paths = list_images(args.data_dir)
    if not image_paths:
        raise FileNotFoundError(f"No images found under {args.data_dir}")
    args.out_dir.mkdir(parents=True, exist_ok=True)
    vectors = extract_transformers_embeddings(image_paths, EMBEDDING_MODELS[args.model], args.batch_size)
    np.save(args.out_dir / f"{args.model}.npy", vectors)
    pd.DataFrame({"path": [str(p) for p in image_paths]}).to_csv(
        args.out_dir / f"{args.model}_paths.csv", index=False
    )


if __name__ == "__main__":
    main()
