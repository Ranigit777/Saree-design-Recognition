"""
Batch sampler for future metric learning at pattern-family level (proxy task).

Does not treat augmentation_group as design identity.
"""
from __future__ import annotations

import random
from collections import defaultdict
from typing import Iterator

import pandas as pd
from torch.utils.data import Sampler

from config import PATTERN_FAMILIES, SPLIT_RANDOM_SEED
from dataset import SareeDataset


class PatternFamilyBatchSampler(Sampler[list[int]]):
    """
    Yields batches with multiple images from several pattern families.

    Intended for contrastive/triplet training where positives share the same
    pattern-family label (proxy only, not fine-grained design).
    """

    def __init__(
        self,
        dataset: SareeDataset,
        batch_size: int = 32,
        samples_per_family: int = 2,
        families_per_batch: int | None = None,
        seed: int = SPLIT_RANDOM_SEED,
    ) -> None:
        if batch_size < 2:
            raise ValueError("batch_size must be >= 2")
        self.dataset = dataset
        self.batch_size = batch_size
        self.samples_per_family = max(1, samples_per_family)
        self.seed = seed

        df: pd.DataFrame = dataset.df
        self.by_family: dict[str, list[int]] = defaultdict(list)
        for idx, row in df.iterrows():
            self.by_family[row["pattern_family"]].append(int(idx))

        n_families = len(PATTERN_FAMILIES)
        if families_per_batch is None:
            families_per_batch = min(n_families, max(2, batch_size // self.samples_per_family))
        self.families_per_batch = min(n_families, max(2, families_per_batch))

    def __iter__(self) -> Iterator[list[int]]:
        rng = random.Random(self.seed)
        family_pools = {f: rng.sample(idxs, len(idxs)) if len(idxs) else [] for f, idxs in self.by_family.items()}
        pointers = {f: 0 for f in family_pools}

        def next_index(fam: str) -> int | None:
            pool = family_pools[fam]
            if not pool:
                return None
            if pointers[fam] >= len(pool):
                rng.shuffle(pool)
                pointers[fam] = 0
            idx = pool[pointers[fam]]
            pointers[fam] += 1
            return idx

        families = [f for f in PATTERN_FAMILIES if family_pools.get(f)]
        num_batches = len(self)
        for _ in range(num_batches):
            batch: list[int] = []
            chosen = rng.sample(families, k=min(self.families_per_batch, len(families)))
            for fam in chosen:
                for _ in range(self.samples_per_family):
                    if len(batch) >= self.batch_size:
                        break
                    ix = next_index(fam)
                    if ix is not None:
                        batch.append(ix)
                if len(batch) >= self.batch_size:
                    break
            while len(batch) < self.batch_size:
                fam = rng.choice(families)
                ix = next_index(fam)
                if ix is None:
                    break
                batch.append(ix)
            if len(batch) >= 2:
                yield batch[: self.batch_size]

    def __len__(self) -> int:
        return max(1, len(self.dataset) // self.batch_size)
