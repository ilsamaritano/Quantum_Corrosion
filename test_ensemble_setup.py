#!/usr/bin/env python3
"""Quick ensemble test"""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent))

from src.config import get_config
print("Config loaded ✓")

from src.ensemble import EnsembleClassifier
print("Ensemble module loaded ✓")

cfg = get_config()
print(f"Config ready: device={cfg.train.device}")

# Load models paths
from src.classical_models import build_classical_model
classical_ckpt = Path("results/models/efficientnet_b0_round0.pt")
quantum_ckpt = Path("results/models/hybrid_vqc_amplitude_round0.pt")

if classical_ckpt.exists():
    print(f"Classical checkpoint found: {classical_ckpt}")
else:
    print(f"❌ Classical checkpoint NOT found")

if quantum_ckpt.exists():
    print(f"Quantum checkpoint found: {quantum_ckpt}")
else:
    print(f"❌ Quantum checkpoint NOT found")

print("\nAll dependencies checked ✓")
