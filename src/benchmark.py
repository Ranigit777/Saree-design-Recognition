"""Efficiency benchmark for metric-learning model."""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parent))

from config import BEST_METRIC_MODEL_PATH, EFFICIENCY_REPORT_PATH, TRAIN_CONFIG
from loaders import get_device
from model import SareeEmbeddingModel


def count_parameters(model: torch.nn.Module) -> tuple[int, int]:
    total = sum(p.numel() for p in model.parameters())
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    return total, trainable


def run_benchmark(batch_size: int = 32, repeats: int = 30) -> dict:
    device = get_device()
    if not BEST_METRIC_MODEL_PATH.is_file():
        raise FileNotFoundError(f"Missing {BEST_METRIC_MODEL_PATH}")
    ckpt = torch.load(BEST_METRIC_MODEL_PATH, map_location=device, weights_only=False)
    cfg = ckpt.get("config", TRAIN_CONFIG.to_dict())
    model = SareeEmbeddingModel(
        embedding_dim=int(cfg.get("embedding_dim", TRAIN_CONFIG.embedding_dim)),
        pretrained=False,
    ).to(device)
    model.load_state_dict(ckpt["model_state_dict"])
    model.eval()

    total_p, train_p = count_parameters(model)
    ckpt_mb = BEST_METRIC_MODEL_PATH.stat().st_size / (1024 * 1024)

    dummy = torch.randn(1, 3, TRAIN_CONFIG.image_size, TRAIN_CONFIG.image_size, device=device)
    with torch.no_grad():
        for _ in range(5):
            model(dummy, return_logits=False)
    times = []
    with torch.no_grad():
        for _ in range(repeats):
            t0 = time.perf_counter()
            model(dummy, return_logits=False)
            if device.type == "cuda":
                torch.cuda.synchronize()
            times.append((time.perf_counter() - t0) * 1000)

    batch = torch.randn(batch_size, 3, TRAIN_CONFIG.image_size, TRAIN_CONFIG.image_size, device=device)
    batch_times = []
    with torch.no_grad():
        for _ in range(max(10, repeats // 3)):
            t0 = time.perf_counter()
            model(batch, return_logits=False)
            if device.type == "cuda":
                torch.cuda.synchronize()
            batch_times.append((time.perf_counter() - t0) * 1000)

    report = {
        "device": str(device),
        "total_parameters": total_p,
        "trainable_parameters": train_p,
        "embedding_dimension": int(cfg.get("embedding_dim", TRAIN_CONFIG.embedding_dim)),
        "checkpoint_size_mb": round(ckpt_mb, 3),
        "single_image_latency_ms_mean": float(np.mean(times)),
        "single_image_latency_ms_std": float(np.std(times)),
        "batch_latency_ms_mean": float(np.mean(batch_times)),
        "batch_latency_ms_std": float(np.std(batch_times)),
        "batch_size": batch_size,
    }
    with open(EFFICIENCY_REPORT_PATH, "w", encoding="utf-8") as f:
        json.dump(report, f, indent=2)
    return report


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.parse_args()
    print(json.dumps(run_benchmark(), indent=2))


if __name__ == "__main__":
    main()
