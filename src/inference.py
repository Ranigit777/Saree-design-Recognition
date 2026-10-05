"""
Unified CLI and programmatic interface for Saree Design Recognition inference.

Supports exact flag formats specified in documentation:
- python src/inference.py --image path/to/image.jpg --top-k 5
- python src/inference.py --image-a a.jpg --image-b b.jpg
- python src/inference.py --embed path/to/image.jpg
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

from infer import (
    run_embedding_extraction,
    run_pairwise_verification,
    run_single_retrieval,
)


def main() -> None:
    parser = argparse.ArgumentParser(description="AIE-CASE Saree Design Recognition Inference")

    # Retrieval args
    parser.add_argument("--image", "--query", type=str, default=None, dest="image", help="Query saree image")
    parser.add_argument("--top-k", type=int, default=5, help="Number of retrieved matches (default: 5)")
    parser.add_argument("--save-vis", type=str, default=None, help="Path to save visualization image")

    # Verification args
    parser.add_argument("--image-a", "--image1", type=str, default=None, dest="image_a", help="First image for pair verification")
    parser.add_argument("--image-b", "--image2", type=str, default=None, dest="image_b", help="Second image for pair verification")
    parser.add_argument("--threshold", type=float, default=None, help="Custom cosine similarity threshold")

    # Embedding extraction
    parser.add_argument("--embed", type=str, default=None, help="Extract 256-D embedding vector")
    parser.add_argument("--output-npy", type=str, default=None, help="Save embedding to .npy")

    # General
    parser.add_argument("--output", type=str, default=None, help="Save results to JSON file")

    args = parser.parse_args()

    if args.image_a and args.image_b:
        run_pairwise_verification(
            img1_path=args.image_a,
            img2_path=args.image_b,
            threshold=args.threshold,
            output_json=args.output,
        )
    elif args.image:
        run_single_retrieval(
            query_path=args.image,
            top_k=args.top_k,
            save_vis=args.save_vis,
            output_json=args.output,
        )
    elif args.embed:
        run_embedding_extraction(
            img_path=args.embed,
            output_npy=args.output_npy,
        )
    else:
        parser.print_help()


if __name__ == "__main__":
    main()
