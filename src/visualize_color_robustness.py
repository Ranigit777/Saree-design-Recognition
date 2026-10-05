"""
Visualize color robustness transformations on sample saree images.
Produces a visual demonstration grid showing original and transformed variants.
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from PIL import Image

sys.path.insert(0, str(Path(__file__).resolve().parent))

from color_robustness import TRANSFORMS
from config import PLOTS_DIR, PROJECT_ROOT, TEST_MANIFEST_PATH


def generate_transformation_grid(sample_image_path: Path | str | None = None, num_samples: int = 3) -> Path:
    test_df = pd.read_csv(TEST_MANIFEST_PATH)
    PLOTS_DIR.mkdir(parents=True, exist_ok=True)
    out_path = PLOTS_DIR / "color_robustness_transforms_grid.png"

    if sample_image_path:
        sample_paths = [Path(sample_image_path)]
    else:
        # Pick one image per pattern family
        sample_paths = []
        for fam in sorted(test_df["pattern_family"].unique())[:num_samples]:
            fam_rows = test_df[test_df["pattern_family"] == fam]
            if not fam_rows.empty:
                sample_paths.append(PROJECT_ROOT / fam_rows.iloc[0]["image_path"])

    n_rows = len(sample_paths)
    n_cols = len(TRANSFORMS)

    fig, axes = plt.subplots(n_rows, n_cols, figsize=(n_cols * 2.5, n_rows * 2.8))
    if n_rows == 1:
        axes = np.expand_dims(axes, 0)

    for r_idx, img_p in enumerate(sample_paths):
        with Image.open(img_p) as raw_im:
            base_im = raw_im.convert("RGB").resize((224, 224))

        for c_idx, (name, (desc, tfm)) in enumerate(TRANSFORMS.items()):
            transformed = tfm(base_im.copy())
            ax = axes[r_idx, c_idx]
            ax.imshow(transformed)
            ax.axis("off")
            if r_idx == 0:
                ax.set_title(name.replace("_", "\n"), fontsize=9, fontweight="bold")
            if c_idx == 0:
                fam = img_p.parent.name
                ax.set_ylabel(fam, fontsize=10, rotation=0, labelpad=40, va="center")

    fig.suptitle("Controlled Synthetic Color Transformations (Robustness Suite)", fontsize=13, fontweight="bold", y=0.98)
    fig.tight_layout()
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"[Visualization] Saved color transformation grid to: {out_path}")
    return out_path


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--image", type=str, default=None, help="Specific saree image path")
    args = parser.parse_args()
    generate_transformation_grid(sample_image_path=args.image)


if __name__ == "__main__":
    main()
