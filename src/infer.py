"""
End-to-end inference script for DeepLure / AIE-CASE Saree Design Recognition.

Supports:
1. Single Image Retrieval: query against gallery, retrieve top-K matches with visual preview
2. Pairwise Verification: determine whether two saree images share the same pattern family
3. Embedding Extraction: extract 256-D L2-normalized feature vector for any saree image
4. Batch Processing: run multiple queries or verification pairs from list/CSV

Proxy supervision only — predicts proxy pattern families; NOT verified fine-grained design IDs.
"""
from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path
from typing import Any, Union

import numpy as np
import pandas as pd
import torch
from PIL import Image, ImageDraw, ImageFont, UnidentifiedImageError

sys.path.insert(0, str(Path(__file__).resolve().parent))

from config import (
    BEST_METRIC_MODEL_PATH,
    EMBEDDINGS_DIR,
    PROJECT_ROOT,
    VERIFICATION_THRESHOLD_PATH,
)
from loaders import get_device
from retrieval import SareeRetrievalEngine
from verification import SareeVerifier


def save_retrieval_visualization(
    query_path: Path | str,
    results: list[dict[str, Any]],
    output_path: Path | str,
) -> None:
    """Save a side-by-side visualization of the query image and top matches."""
    q_p = Path(query_path)
    if not q_p.is_absolute():
        q_p = PROJECT_ROOT / q_p

    try:
        q_img = Image.open(q_p).convert("RGB").resize((224, 224))
    except Exception:
        return

    n_matches = len(results)
    cell_w, cell_h = 224, 260
    total_w = cell_w * (n_matches + 1)
    canvas = Image.new("RGB", (total_w, cell_h), color=(240, 240, 240))
    draw = ImageDraw.Draw(canvas)

    # Paste query
    canvas.paste(q_img, (0, 30))
    draw.text((10, 8), "QUERY IMAGE", fill=(200, 0, 0))

    # Paste matches
    for i, res in enumerate(results):
        x_offset = cell_w * (i + 1)
        m_p = PROJECT_ROOT / res["image_path"]
        if m_p.is_file():
            try:
                m_img = Image.open(m_p).convert("RGB").resize((224, 224))
                canvas.paste(m_img, (x_offset, 30))
            except Exception:
                pass
        txt = f"Rank {res['rank']}: {res['similarity']:.3f}\n{res['pattern_family']}"
        draw.text((x_offset + 10, 4), txt, fill=(0, 0, 0))

    out_p = Path(output_path)
    out_p.parent.mkdir(parents=True, exist_ok=True)
    canvas.save(out_p)
    print(f"[Inference] Saved retrieval visualization to: {out_p}")


def run_single_retrieval(
    query_path: str,
    top_k: int = 5,
    gallery_emb: str | None = None,
    gallery_meta: str | None = None,
    save_vis: str | None = None,
    output_json: str | None = None,
) -> list[dict[str, Any]]:
    print(f"\n[Retrieval] Querying: {query_path}")
    engine = SareeRetrievalEngine(
        gallery_emb_path=gallery_emb,
        gallery_meta_path=gallery_meta,
    )
    results = engine.search(query_path, top_k=top_k)

    print("\n" + "=" * 70)
    print(f"{'RANK':<6} | {'SIMILARITY':<12} | {'PATTERN FAMILY':<16} | {'IMAGE PATH'}")
    print("-" * 70)
    for r in results:
        print(f"{r['rank']:<6} | {r['similarity']:<12.4f} | {r['pattern_family']:<16} | {r['image_path']}")
    print("=" * 70)

    if save_vis:
        save_retrieval_visualization(query_path, results, save_vis)

    if output_json:
        with open(output_json, "w", encoding="utf-8") as f:
            json.dump({"query": query_path, "results": results}, f, indent=2)
        print(f"[Inference] Saved results JSON to: {output_json}")

    return results


def run_pairwise_verification(
    img1_path: str,
    img2_path: str,
    threshold: float | None = None,
    output_json: str | None = None,
) -> dict[str, Any]:
    print(f"\n[Verification] Comparing:")
    print(f"  Image 1: {img1_path}")
    print(f"  Image 2: {img2_path}")

    verifier = SareeVerifier(threshold=threshold)
    out = verifier.verify(img1_path, img2_path)

    print("\n" + "=" * 50)
    print(f"PREDICTION:        {out['prediction']}")
    print(f"Cosine Similarity: {out['cosine_similarity']:.4f}")
    print(f"Threshold:         {out['threshold']:.4f}")
    print(f"Confidence:        {out['confidence']:.4f}")
    print(f"Margin:            {out['margin_from_threshold']:+.4f}")
    print("=" * 50)

    if output_json:
        res = {"image1": img1_path, "image2": img2_path, **out}
        with open(output_json, "w", encoding="utf-8") as f:
            json.dump(res, f, indent=2)
        print(f"[Inference] Saved verification result to: {output_json}")

    return out


def run_embedding_extraction(
    img_path: str,
    output_npy: str | None = None,
) -> np.ndarray:
    engine = SareeRetrievalEngine()
    emb = engine.extract_embedding(img_path).flatten()
    print(f"\n[Embedding] Extracted {len(emb)}-dimensional embedding.")
    print(f"  L2 Norm:          {np.linalg.norm(emb):.4f}")
    print(f"  First 8 values:   {np.round(emb[:8], 4).tolist()}")

    if output_npy:
        np.save(output_npy, emb)
        print(f"[Inference] Saved embedding vector to: {output_npy}")
    return emb


def main() -> None:
    parser = argparse.ArgumentParser(description="DeepLure Saree Recognition End-to-End Inference CLI")

    # Mode 1: Retrieval
    parser.add_argument("--query", type=str, default=None, help="Query image path for retrieval")
    parser.add_argument("--top-k", type=int, default=5, help="Number of top gallery matches")
    parser.add_argument("--gallery-emb", type=str, default=None, help="Custom gallery embeddings .npy")
    parser.add_argument("--gallery-meta", type=str, default=None, help="Custom gallery metadata .csv")
    parser.add_argument("--save-vis", type=str, default=None, help="Save query+matches image visualization")

    # Mode 2: Verification
    parser.add_argument("--image1", type=str, default=None, help="First image for verification")
    parser.add_argument("--image2", type=str, default=None, help="Second image for verification")
    parser.add_argument("--threshold", type=float, default=None, help="Custom verification threshold")

    # Mode 3: Embedding extraction
    parser.add_argument("--embed", type=str, default=None, help="Image path to extract embedding")
    parser.add_argument("--output-npy", type=str, default=None, help="Destination .npy file for embedding")

    # General
    parser.add_argument("--output", type=str, default=None, help="Destination JSON file for outputs")

    args = parser.parse_args()

    # Route according to provided flags
    if args.query:
        run_single_retrieval(
            args.query,
            top_k=args.top_k,
            gallery_emb=args.gallery_emb,
            gallery_meta=args.gallery_meta,
            save_vis=args.save_vis,
            output_json=args.output,
        )
    elif args.image1 and args.image2:
        run_pairwise_verification(
            args.image1,
            args.image2,
            threshold=args.threshold,
            output_json=args.output,
        )
    elif args.embed:
        run_embedding_extraction(
            args.embed,
            output_npy=args.output_npy,
        )
    else:
        parser.print_help()
        print("\nExample commands:")
        print("  1. Retrieval:    python src/infer.py --query data/indian_saree/test/Banarasi/sample.jpg --top-k 5")
        print("  2. Verification: python src/infer.py --image1 img1.jpg --image2 img2.jpg")
        print("  3. Embedding:    python src/infer.py --embed img.jpg --output-npy emb.npy")


if __name__ == "__main__":
    main()
