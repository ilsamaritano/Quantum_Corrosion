#!/usr/bin/env python3
"""
Synthetic end-to-end validation — verifies the full pipeline
without requiring the real 3 GB IQ dataset.
Generates fake IQ data, runs preprocessing, trains 1 epoch,
evaluates, and generates all figures.
"""
import sys, json, logging, shutil
from pathlib import Path
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
logging.basicConfig(level=logging.INFO, format="%(asctime)s %(message)s")
log = logging.getLogger("test")

from src.config import get_config

# ── 1. Generate synthetic IQ data ──────────────────────────────────────────
def generate_synthetic_data(cfg):
    """Create small fake .iq files for each class."""
    log.info("Generating synthetic IQ data...")
    n_samples_per_file = cfg.signal.fft_size * cfg.signal.n_stacks * 3  # enough for 3 images
    for label, cls_name in enumerate(cfg.class_names):
        cls_dir = cfg.paths.data_raw / cls_name
        cls_dir.mkdir(parents=True, exist_ok=True)
        rng = np.random.default_rng(42 + label)
        # Create 2 small IQ files per class
        for fid in range(2):
            # Simulate class-dependent signal characteristics
            freq_offset = label * 0.1
            noise_level = 0.05 + label * 0.02
            t = np.arange(n_samples_per_file, dtype=np.float32)
            iq = (
                np.exp(1j * 2 * np.pi * (0.01 + freq_offset) * t) * (1.0 + label * 0.3)
                + rng.normal(0, noise_level, n_samples_per_file)
                + 1j * rng.normal(0, noise_level, n_samples_per_file)
            ).astype(np.complex64)
            fpath = cls_dir / f"synthetic_{fid}.iq"
            iq.tofile(str(fpath))
        log.info(f"  {cls_name}: 2 files, {n_samples_per_file} samples each")


# ── 2. Run preprocessing ──────────────────────────────────────────────────
def test_preprocessing(cfg):
    from src.preprocessing import preprocess_dataset
    log.info("Testing preprocessing...")
    records = preprocess_dataset(cfg, max_workers=2)
    assert len(records) > 0, "No spectrograms produced!"
    log.info(f"  ✓ Produced {len(records)} spectrograms")
    # Verify a sample
    sample = np.load(records[0]["path"])
    assert sample.shape == cfg.signal.img_size, f"Shape mismatch: {sample.shape}"
    assert sample.dtype == np.uint8
    log.info(f"  ✓ Sample shape={sample.shape}, dtype={sample.dtype}")
    return records


# ── 3. Test dataset & dataloaders ──────────────────────────────────────────
def test_dataset(cfg, records):
    from src.dataset import create_splits, build_dataloaders
    log.info("Testing dataset splits & loaders...")
    train_m, val_m, test_m = create_splits(
        records, ratios=(0.6, 0.2, 0.2), seed=42,
        save_dir=cfg.paths.data_splits,
    )
    assert len(train_m) > 0 and len(val_m) > 0 and len(test_m) > 0
    log.info(f"  ✓ Splits: train={len(train_m)}, val={len(val_m)}, test={len(test_m)}")

    import torch
    train_loader, val_loader, test_loader = build_dataloaders(
        train_m, val_m, test_m,
        batch_size=4, num_workers=0, pin_memory=False,
    )
    batch = next(iter(train_loader))
    imgs, labels = batch
    assert imgs.shape[1] == 1, f"Expected 1 channel, got {imgs.shape[1]}"
    assert imgs.shape[2] == 224 and imgs.shape[3] == 224
    log.info(f"  ✓ Batch shape: {imgs.shape}, labels: {labels.tolist()}")
    return train_m, val_m, test_m, train_loader, val_loader, test_loader


# ── 4. Test classical models ──────────────────────────────────────────────
def test_classical(cfg, train_loader, val_loader, test_loader):
    import torch
    from src.classical_models import build_classical_model, count_parameters, model_summary
    from src.training import train_model, set_seed
    from src.evaluation import evaluate_model

    results = {}
    for arch in ["resnet18", "mobilenetv2", "efficientnet_b0"]:
        log.info(f"  Testing {arch}...")
        set_seed(42)
        model = build_classical_model(arch, cfg.n_classes, 1, pretrained=False, dropout=0.1)
        params = count_parameters(model)
        log.info(f"    Params: {params['trainable']:,}")

        # Quick 2-epoch train
        rr = train_model(
            model, train_loader, val_loader,
            epochs=2, lr=1e-3, device="cpu",
            use_amp=False, patience=5, round_id=0,
            model_name=f"test_{arch}",
        )
        log.info(f"    ✓ Trained: best val acc={rr.best_val_acc:.4f}")

        er = evaluate_model(
            model, test_loader, cfg.class_names,
            device="cpu", model_name=arch, round_id=0,
            train_time=rr.train_time_total,
        )
        log.info(f"    ✓ Test F1={er.f1_score:.4f}, Acc={er.accuracy:.4f}")
        results[arch] = er

    return results


# ── 5. Test quantum model ──────────────────────────────────────────────────
def test_quantum(cfg, train_m, val_m, test_m):
    try:
        import pennylane
    except ImportError:
        log.warning("  PennyLane not installed — skipping quantum test")
        return None

    import torch
    from torch.utils.data import DataLoader
    from src.dataset import PCAExtractor, PCASpectrogramDataset
    from src.quantum_models import build_quantum_model, quantum_param_count
    from src.training import train_model, set_seed
    from src.evaluation import evaluate_model

    log.info("  Testing quantum model...")

    # Fit PCA on small data
    pca = PCAExtractor(n_components=16, batch_size=8)  # tiny for test
    pca.fit(train_m)

    train_ds = PCASpectrogramDataset(train_m, pca)
    val_ds = PCASpectrogramDataset(val_m, pca)
    test_ds = PCASpectrogramDataset(test_m, pca)
    train_ld = DataLoader(train_ds, batch_size=4, shuffle=True)
    val_ld = DataLoader(val_ds, batch_size=4)
    test_ld = DataLoader(test_ds, batch_size=4)

    set_seed(42)
    model = build_quantum_model(
        input_dim=16, n_classes=cfg.n_classes,
        n_qubits=4, n_layers=1, encoding="amplitude",
    )
    qpc = quantum_param_count(model)
    log.info(f"    Quantum params: {qpc['quantum']}, total: {qpc['total']}")

    rr = train_model(
        model, train_ld, val_ld,
        epochs=2, lr=5e-3, device="cpu",
        use_amp=False, patience=5, model_name="test_quantum",
    )
    er = evaluate_model(model, test_ld, cfg.class_names, device="cpu", model_name="quantum")
    log.info(f"    ✓ Quantum test F1={er.f1_score:.4f}")
    return er


# ── 6. Test figure generation ──────────────────────────────────────────────
def test_figures(cfg, classical_results, test_m):
    from src.evaluation import AggregatedResult
    from src.plots import generate_all_figures

    log.info("Testing figure generation...")

    # Build mock AggregatedResults
    agg_list = []
    for name, er in classical_results.items():
        agg = AggregatedResult(
            model_name=name, n_rounds=1,
            mean_accuracy=er.accuracy, std_accuracy=0.01,
            mean_f1=er.f1_score, std_f1=0.01,
            min_f1=er.f1_score - 0.02, max_f1=er.f1_score + 0.02,
            var_f1=0.0001,
            mean_precision=er.precision, mean_recall=er.recall,
            mean_inference_ms=er.inference_time_ms,
            mean_train_time=er.train_time_s,
            n_params_trainable=er.n_params_trainable,
            per_round_f1=[er.f1_score],
        )
        agg_list.append(agg)

    # Confusion matrices
    cms = {}
    for name, er in classical_results.items():
        if er.confusion_matrix is not None:
            cms[name] = (er.confusion_matrix, cfg.class_names)

    # Example spectrograms
    specs = {}
    for cls_idx, cls_name in enumerate(cfg.class_names[:3]):
        for rec in test_m:
            if rec["label"] == cls_idx:
                specs[cls_name] = np.load(rec["path"])
                break

    # Mock learning curves
    fracs = [0.2, 0.5, 1.0]
    lc_data = {
        "resnet18": [(0.3, 0.05), (0.5, 0.04), (0.7, 0.03)],
        "mobilenetv2": [(0.25, 0.06), (0.45, 0.05), (0.65, 0.04)],
    }

    generate_all_figures(
        aggregated=agg_list,
        out_dir=cfg.paths.figures,
        cfg=cfg.plot,
        example_spectrograms=specs,
        learning_curve_data=lc_data,
        learning_curve_fractions=fracs,
        confusion_matrices=cms,
    )

    figs = list(cfg.paths.figures.glob("*.pdf"))
    log.info(f"  ✓ Generated {len(figs)} PDF figures:")
    for f in sorted(figs):
        log.info(f"    {f.name}")

    return figs


# ── Main test ──────────────────────────────────────────────────────────────
def main():
    log.info("=" * 60)
    log.info("SYNTHETIC END-TO-END VALIDATION")
    log.info("=" * 60)

    cfg = get_config()
    # Clean any previous test data
    for d in [cfg.paths.data_raw, cfg.paths.data_processed, cfg.paths.data_splits,
              cfg.paths.figures, cfg.paths.models]:
        if d.exists():
            shutil.rmtree(d)
    cfg.ensure_dirs()

    # Run all stages
    generate_synthetic_data(cfg)
    records = test_preprocessing(cfg)
    train_m, val_m, test_m, train_ld, val_ld, test_ld = test_dataset(cfg, records)
    classical_results = test_classical(cfg, train_ld, val_ld, test_ld)
    quantum_result = test_quantum(cfg, train_m, val_m, test_m)
    figs = test_figures(cfg, classical_results, test_m)

    log.info("")
    log.info("=" * 60)
    log.info("ALL TESTS PASSED ✓")
    log.info(f"  Spectrograms: {len(records)}")
    log.info(f"  Classical models tested: {list(classical_results.keys())}")
    log.info(f"  Quantum tested: {'yes' if quantum_result else 'skipped (no PennyLane)'}")
    log.info(f"  Figures generated: {len(figs)}")
    log.info("=" * 60)


if __name__ == "__main__":
    main()
