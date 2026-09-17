"""
Dataset — PyTorch Dataset / DataLoader utilities
=================================================
Handles spectrogram loading, on-the-fly augmentation, balanced
sampling, and PCA-based feature extraction for quantum models.
"""

import json
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader, Subset, WeightedRandomSampler
from torchvision import transforms
from pathlib import Path
from typing import Tuple, Optional, List, Dict
from sklearn.model_selection import StratifiedShuffleSplit
from sklearn.decomposition import IncrementalPCA
from collections import Counter
import re
import logging
import joblib

logger = logging.getLogger(__name__)


class SpectrogramDataset(Dataset):
    """
    Loads pre-computed .npy spectrogram images.
    Returns (image_tensor, label) pairs.
    """

    def __init__(
        self,
        metadata: List[Dict],
        transform=None,
        return_flat: bool = False,
    ):
        self.metadata = metadata
        self.transform = transform
        self.return_flat = return_flat

    def __len__(self) -> int:
        return len(self.metadata)

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, int]:
        rec = self.metadata[idx]
        img = np.load(rec["path"]).astype(np.float32) / 255.0

        # Expand to (1, H, W) grayscale
        if img.ndim == 2:
            img = img[np.newaxis, :, :]

        img = torch.from_numpy(img)

        if self.transform is not None:
            img = self.transform(img)

        label = rec["label"]

        if self.return_flat:
            return img.flatten(), label
        return img, label

    @property
    def labels(self) -> np.ndarray:
        return np.array([r["label"] for r in self.metadata])


# ── Transforms ──────────────────────────────────────────────────────────────

def get_train_transform(img_size: int = 224) -> transforms.Compose:
    return transforms.Compose([
        transforms.RandomHorizontalFlip(p=0.5),
        transforms.RandomVerticalFlip(p=0.3),
        transforms.RandomAffine(degrees=5, translate=(0.05, 0.05), scale=(0.95, 1.05)),
        transforms.RandomErasing(p=0.2, scale=(0.02, 0.08)),
        transforms.Normalize(mean=[0.5], std=[0.5]),
    ])


def get_val_transform() -> transforms.Compose:
    return transforms.Compose([
        transforms.Normalize(mean=[0.5], std=[0.5]),
    ])


# ── Splitting ───────────────────────────────────────────────────────────────

def create_splits(
    metadata: List[Dict],
    ratios: Tuple[float, float, float] = (0.6, 0.2, 0.2),
    seed: int = 42,
    save_dir: Optional[Path] = None,
) -> Tuple[List[Dict], List[Dict], List[Dict]]:
    """Group-aware stratified train/val/test split preserving class balance.

    If possible, samples extracted from the same source IQ file are kept in the
    same split to prevent leakage. Falls back to pure stratified split when no
    file-group information can be inferred from metadata paths.
    """

    def _extract_group_id(rec: Dict) -> Optional[str]:
        # Expected filename pattern from preprocessing:
        # label{label}_file{file_idx}_img{img_idx}.npy
        path = str(rec.get("path", ""))
        m = re.search(r"label(\d+)_file(\d+)_img\d+\.npy$", path)
        if m:
            return f"label{m.group(1)}_file{m.group(2)}"
        return None

    labels = np.array([r["label"] for r in metadata])
    n = len(metadata)
    groups = [_extract_group_id(r) for r in metadata]
    use_group_split = all(g is not None for g in groups)

    if use_group_split:
        # Per-class group split: preserves class balance while preventing
        # same-source leakage across train/val/test.
        rng = np.random.default_rng(seed)
        train_idx, val_idx, test_idx = [], [], []

        labels_arr = np.array(labels)
        group_arr = np.array(groups)

        for cls in np.unique(labels_arr):
            cls_idx = np.where(labels_arr == cls)[0]
            cls_groups = np.unique(group_arr[cls_idx])
            cls_groups = rng.permutation(cls_groups)

            n_groups = len(cls_groups)
            if n_groups == 1:
                train_groups = cls_groups
                val_groups = np.array([], dtype=object)
                test_groups = np.array([], dtype=object)
            else:
                n_train = max(1, int(round(n_groups * ratios[0])))
                n_val = int(round(n_groups * ratios[1]))

                if n_train >= n_groups:
                    n_train = n_groups - 1
                if n_train + n_val >= n_groups:
                    n_val = max(0, n_groups - n_train - 1)

                train_groups = cls_groups[:n_train]
                val_groups = cls_groups[n_train:n_train + n_val]
                test_groups = cls_groups[n_train + n_val:]

                if len(test_groups) == 0 and len(val_groups) > 1:
                    test_groups = val_groups[-1:]
                    val_groups = val_groups[:-1]

            grp_to_split = {
                g: "train" for g in train_groups
            }
            grp_to_split.update({g: "val" for g in val_groups})
            grp_to_split.update({g: "test" for g in test_groups})

            for i in cls_idx:
                split_name = grp_to_split.get(group_arr[i], "train")
                if split_name == "train":
                    train_idx.append(i)
                elif split_name == "val":
                    val_idx.append(i)
                else:
                    test_idx.append(i)

        train_idx = np.array(sorted(train_idx), dtype=np.int64)
        val_idx = np.array(sorted(val_idx), dtype=np.int64)
        test_idx = np.array(sorted(test_idx), dtype=np.int64)

        # If class groups are too few (e.g. one source file per class),
        # group-aware split can collapse val/test to empty. Fall back to
        # standard stratified splitting to keep all sets usable.
        if len(val_idx) == 0 or len(test_idx) == 0:
            use_group_split = False
    if not use_group_split:
        # Fallback stratified split (no group info available)
        # First split: train vs (val+test)
        sss1 = StratifiedShuffleSplit(
            n_splits=1, test_size=ratios[1] + ratios[2], random_state=seed
        )
        train_idx, rest_idx = next(sss1.split(np.zeros(n), labels))

        # Second split: val vs test
        rest_labels = labels[rest_idx]
        val_ratio = ratios[1] / (ratios[1] + ratios[2])
        sss2 = StratifiedShuffleSplit(
            n_splits=1, test_size=1.0 - val_ratio, random_state=seed
        )
        val_sub, test_sub = next(sss2.split(np.zeros(len(rest_idx)), rest_labels))
        val_idx = rest_idx[val_sub]
        test_idx = rest_idx[test_sub]

    meta_arr = np.array(metadata)
    train_meta = meta_arr[train_idx].tolist()
    val_meta = meta_arr[val_idx].tolist()
    test_meta = meta_arr[test_idx].tolist()

    # Log distribution
    for name, m in [("Train", train_meta), ("Val", val_meta), ("Test", test_meta)]:
        dist = Counter(r["label"] for r in m)
        logger.info(f"  {name}: {len(m)} samples — {dict(sorted(dist.items()))}")

    if use_group_split:
        logger.info("  Split mode: group-aware (no same-source file across splits)")
    else:
        logger.info("  Split mode: stratified (fallback, group id not detected)")

    if save_dir is not None:
        save_dir.mkdir(parents=True, exist_ok=True)
        for name, m in [("train", train_meta), ("val", val_meta), ("test", test_meta)]:
            with open(save_dir / f"{name}.json", 'w') as f:
                json.dump(m, f)

    return train_meta, val_meta, test_meta


# ── Balanced sampling ───────────────────────────────────────────────────────

def make_balanced_sampler(labels: np.ndarray) -> WeightedRandomSampler:
    """Inverse-frequency weighted sampler for class imbalance."""
    counts = Counter(labels.tolist())
    weights_per_class = {c: 1.0 / cnt for c, cnt in counts.items()}
    sample_weights = np.array([weights_per_class[l] for l in labels])
    return WeightedRandomSampler(
        weights=torch.from_numpy(sample_weights).double(),
        num_samples=len(labels),
        replacement=True,
    )


# ── DataLoader factory ──────────────────────────────────────────────────────

def build_dataloaders(
    train_meta: List[Dict],
    val_meta: List[Dict],
    test_meta: List[Dict],
    batch_size: int = 32,
    num_workers: int = 8,
    pin_memory: bool = True,
    for_quantum: bool = False,
) -> Tuple[DataLoader, DataLoader, DataLoader]:
    """Build train/val/test DataLoaders with proper transforms."""
    train_ds = SpectrogramDataset(
        train_meta,
        transform=get_train_transform(),
        return_flat=for_quantum,
    )
    val_ds = SpectrogramDataset(
        val_meta,
        transform=get_val_transform(),
        return_flat=for_quantum,
    )
    test_ds = SpectrogramDataset(
        test_meta,
        transform=get_val_transform(),
        return_flat=for_quantum,
    )

    sampler = make_balanced_sampler(train_ds.labels)

    train_loader = DataLoader(
        train_ds, batch_size=batch_size, sampler=sampler,
        num_workers=num_workers, pin_memory=pin_memory, drop_last=True,
    )
    val_loader = DataLoader(
        val_ds, batch_size=batch_size, shuffle=False,
        num_workers=num_workers, pin_memory=pin_memory,
    )
    test_loader = DataLoader(
        test_ds, batch_size=batch_size, shuffle=False,
        num_workers=num_workers, pin_memory=pin_memory,
    )
    return train_loader, val_loader, test_loader


# ── PCA feature extraction (for quantum models) ────────────────────────────

class PCAExtractor:
    """
    Incremental PCA for reducing spectrogram dimensionality
    before quantum encoding. Fits in streaming fashion for
    large datasets.
    """

    def __init__(self, n_components: int = 256, batch_size: int = 2048):
        self.pca = IncrementalPCA(n_components=n_components, batch_size=batch_size)
        self.n_components = n_components
        self.fitted = False

    def fit(self, metadata: List[Dict]) -> "PCAExtractor":
        """Fit PCA incrementally on all spectrograms."""
        logger.info(f"Fitting PCA ({self.n_components} components) on {len(metadata)} images...")
        batch = []
        for rec in tqdm(metadata, desc="PCA fitting"):
            img = np.load(rec["path"]).astype(np.float32).flatten() / 255.0
            batch.append(img)
            if len(batch) >= self.pca.batch_size:
                self.pca.partial_fit(np.array(batch))
                batch = []
        if batch:
            self.pca.partial_fit(np.array(batch))
        self.fitted = True
        ev = self.pca.explained_variance_ratio_.sum()
        logger.info(f"PCA: {self.n_components} components explain {ev:.3%} variance")
        return self

    def transform(self, img_flat: np.ndarray) -> np.ndarray:
        return self.pca.transform(img_flat.reshape(1, -1)).flatten()

    def save(self, path: Path):
        joblib.dump(self.pca, path)
        logger.info(f"PCA saved → {path}")

    def load(self, path: Path) -> "PCAExtractor":
        self.pca = joblib.load(path)
        self.fitted = True
        return self


class PCASpectrogramDataset(Dataset):
    """Dataset that returns PCA-reduced features for quantum models."""

    def __init__(
        self,
        metadata: List[Dict],
        pca: PCAExtractor,
        use_feature_engineering: bool = True,
    ):
        self.metadata = metadata
        self.pca = pca
        self.use_feature_engineering = use_feature_engineering

    @staticmethod
    def _engineer_features(features: np.ndarray) -> np.ndarray:
        """Apply robust, non-linear feature engineering while keeping same dimension."""
        x = np.asarray(features, dtype=np.float32)
        x = np.nan_to_num(x, nan=0.0, posinf=0.0, neginf=0.0)

        # Robust normalization: less sensitive to outliers than mean/std scaling.
        median = float(np.median(x))
        q1 = float(np.percentile(x, 25.0))
        q3 = float(np.percentile(x, 75.0))
        iqr = q3 - q1
        if iqr < 1e-6:
            iqr = 1.0
        z = (x - median) / iqr

        # Local trend removal to emphasize high-frequency discriminative cues.
        kernel_size = min(31, max(5, len(z) // 32))
        if kernel_size % 2 == 0:
            kernel_size += 1
        kernel = np.ones(kernel_size, dtype=np.float32) / float(kernel_size)
        trend = np.convolve(z, kernel, mode="same")
        z_hf = z - trend

        # Non-linear basis expansion with saturation control.
        z_log = np.sign(z) * np.log1p(np.abs(z))
        z_sq = np.sign(z) * (z * z) / (1.0 + np.abs(z))
        z_hf_norm = z_hf / (np.sqrt(np.mean(z_hf * z_hf)) + 1e-6)

        engineered = (
            0.35 * z
            + 0.20 * z_log
            + 0.20 * np.tanh(z_hf)
            + 0.15 * z_sq
            + 0.10 * z_hf_norm
        )
        engineered = np.clip(engineered, -4.0, 4.0)
        return engineered.astype(np.float32, copy=False)

    def __len__(self):
        return len(self.metadata)

    def __getitem__(self, idx) -> Tuple[torch.Tensor, int]:
        rec = self.metadata[idx]
        img = np.load(rec["path"]).astype(np.float32).flatten() / 255.0
        features = self.pca.transform(img)
        if self.use_feature_engineering:
            features = self._engineer_features(features)
        return torch.from_numpy(features).float(), rec["label"]


# ── Subsample for data-efficiency experiments ───────────────────────────────

def subsample_metadata(
    metadata: List[Dict],
    fraction: float,
    seed: int = 42,
) -> List[Dict]:
    """Stratified subsample preserving class ratios."""
    if fraction >= 1.0:
        return metadata
    labels = np.array([r["label"] for r in metadata])
    rng = np.random.default_rng(seed)
    selected = []
    for cls in np.unique(labels):
        cls_idx = np.where(labels == cls)[0]
        n_keep = max(1, int(len(cls_idx) * fraction))
        chosen = rng.choice(cls_idx, size=n_keep, replace=False)
        selected.extend(chosen.tolist())
    meta_arr = np.array(metadata)
    return meta_arr[sorted(selected)].tolist()


def reduce_redundancy_metadata(
    metadata: List[Dict],
    stride: int = 2,
) -> List[Dict]:
    """
    Reduce temporal redundancy by keeping every `stride`-th sample per class,
    ordered by the image index embedded in filenames like `..._img000123.npy`.
    """
    if stride <= 1:
        return metadata

    import re

    def _img_idx(path: str) -> int:
        m = re.search(r"_img(\d+)\.npy$", path)
        return int(m.group(1)) if m else 0

    by_class: Dict[int, List[Dict]] = {}
    for rec in metadata:
        by_class.setdefault(int(rec["label"]), []).append(rec)

    reduced: List[Dict] = []
    for cls, items in by_class.items():
        items_sorted = sorted(items, key=lambda r: _img_idx(r["path"]))
        picked = items_sorted[::stride]
        if not picked and items_sorted:
            picked = [items_sorted[0]]
        reduced.extend(picked)

    return reduced


# helper import for PCA fitting progress
from tqdm import tqdm
