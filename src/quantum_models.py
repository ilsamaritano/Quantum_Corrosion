"""
Quantum Models — Hybrid quantum-classical classifiers
======================================================
PennyLane-based VQC with multiple encoding strategies,
integrated with a classical feature-extraction head.
"""

import numpy as np
import torch
import torch.nn as nn
from typing import List, Optional
import logging

try:
    import pennylane as qml
    HAS_PENNYLANE = True
except ImportError:
    HAS_PENNYLANE = False

logger = logging.getLogger(__name__)


# ── Quantum circuit components ──────────────────────────────────────────────

def _resolve_diff_method(encoding: str, backend: str, diff_method: str) -> str:
    """Pick a PennyLane diff method that is compatible with the selected setup."""
    method = (diff_method or "").strip().lower()
    backend_name = (backend or "").strip().lower()

    # Adjoint is not compatible with state-prep style amplitude embedding
    # on lightning.gpu for this circuit; parameter-shift is a safe fallback.
    if encoding == "amplitude" and method == "adjoint" and backend_name == "lightning.gpu":
        logger.warning(
            "Incompatible diff_method='adjoint' for encoding='amplitude' on backend='lightning.gpu'. "
            "Falling back to diff_method='parameter-shift'."
        )
        return "parameter-shift"

    return method or "adjoint"

def _build_amplitude_circuit(
    n_qubits: int,
    n_layers: int,
    entanglement: str,
    backend: str,
    diff_method: str,
):
    """VQC with amplitude encoding: input must be 2^n_qubits dimensional."""
    dev = qml.device(backend, wires=n_qubits)

    @qml.qnode(dev, interface="torch", diff_method=diff_method)
    def circuit(inputs, weights):
        # Amplitude encoding (normalised input vector)
        qml.AmplitudeEmbedding(features=inputs, wires=range(n_qubits),
                                normalize=True, pad_with=0.0)
        # Variational layers
        for layer in range(n_layers):
            for q in range(n_qubits):
                qml.RY(weights[layer, q, 0], wires=q)
                qml.RZ(weights[layer, q, 1], wires=q)
            # Entanglement
            if entanglement == "full":
                for q1 in range(n_qubits):
                    for q2 in range(q1 + 1, n_qubits):
                        qml.CNOT(wires=[q1, q2])
            elif entanglement == "linear":
                for q in range(n_qubits - 1):
                    qml.CNOT(wires=[q, q + 1])
            elif entanglement == "circular":
                for q in range(n_qubits - 1):
                    qml.CNOT(wires=[q, q + 1])
                qml.CNOT(wires=[n_qubits - 1, 0])

        return qml.probs(wires=range(n_qubits))

    weight_shape = (n_layers, n_qubits, 2)
    return circuit, weight_shape, dev


def _build_angle_circuit(
    n_qubits: int,
    n_layers: int,
    entanglement: str,
    backend: str,
    diff_method: str,
):
    """VQC with angle encoding: one feature per qubit per re-upload layer."""
    dev = qml.device(backend, wires=n_qubits)

    @qml.qnode(dev, interface="torch", diff_method=diff_method)
    def circuit(inputs, weights):
        # Data re-uploading with angle encoding
        for layer in range(n_layers):
            for q in range(n_qubits):
                idx = layer * n_qubits + q
                if idx < len(inputs):
                    qml.RX(inputs[idx], wires=q)
                qml.RY(weights[layer, q, 0], wires=q)
                qml.RZ(weights[layer, q, 1], wires=q)
            if entanglement == "full":
                for q1 in range(n_qubits):
                    for q2 in range(q1 + 1, n_qubits):
                        qml.CNOT(wires=[q1, q2])
            elif entanglement == "linear":
                for q in range(n_qubits - 1):
                    qml.CNOT(wires=[q, q + 1])
            elif entanglement == "circular":
                for q in range(n_qubits - 1):
                    qml.CNOT(wires=[q, q + 1])
                qml.CNOT(wires=[n_qubits - 1, 0])

        return qml.probs(wires=range(n_qubits))

    weight_shape = (n_layers, n_qubits, 2)
    return circuit, weight_shape, dev


def _build_iqp_circuit(
    n_qubits: int,
    n_layers: int,
    entanglement: str,
    backend: str,
    diff_method: str,
):
    """IQP (Instantaneous Quantum Polynomial) encoding."""
    dev = qml.device(backend, wires=n_qubits)

    @qml.qnode(dev, interface="torch", diff_method=diff_method)
    def circuit(inputs, weights):
        for layer in range(n_layers):
            # Hadamard layer
            for q in range(n_qubits):
                qml.Hadamard(wires=q)
            # Diagonal encoding
            for q in range(n_qubits):
                idx = q % len(inputs)
                qml.RZ(inputs[idx], wires=q)
            # ZZ interactions
            for q in range(n_qubits - 1):
                i1 = q % len(inputs)
                i2 = (q + 1) % len(inputs)
                qml.CNOT(wires=[q, q + 1])
                qml.RZ(inputs[i1] * inputs[i2], wires=q + 1)
                qml.CNOT(wires=[q, q + 1])
            # Variational rotation
            for q in range(n_qubits):
                qml.RY(weights[layer, q, 0], wires=q)
                qml.RZ(weights[layer, q, 1], wires=q)

        return qml.probs(wires=range(n_qubits))

    weight_shape = (n_layers, n_qubits, 2)
    return circuit, weight_shape, dev


# ── Hybrid model class ──────────────────────────────────────────────────────

class HybridQuantumClassifier(nn.Module):
    """
    Hybrid Quantum-Classical Classifier
    ====================================
    Architecture:
        input → classical_encoder (Linear layers) → quantum_circuit (VQC)
        → classical_head → logits

    The classical encoder reduces dimension to match quantum input size.
    The VQC processes the reduced features.
    The classical head maps VQC output probabilities to class logits.
    """

    def __init__(
        self,
        input_dim: int,
        n_classes: int,
        n_qubits: int = 8,
        n_layers: int = 4,
        encoding: str = "amplitude",
        entanglement: str = "full",
        backend: str = "default.qubit",
        diff_method: str = "adjoint",
        head_dims: List[int] = [64, 32],
    ):
        super().__init__()
        if not HAS_PENNYLANE:
            raise ImportError("PennyLane required: pip install pennylane")

        self.n_qubits = n_qubits
        self.n_layers = n_layers
        self.encoding = encoding
        self.diff_method = _resolve_diff_method(encoding, backend, diff_method)

        # Quantum input dimension
        if encoding == "amplitude":
            self.q_input_dim = 2 ** n_qubits
        elif encoding == "angle":
            self.q_input_dim = n_qubits * n_layers
        elif encoding == "iqp":
            self.q_input_dim = n_qubits
        else:
            raise ValueError(f"Unknown encoding: {encoding}")

        # Classical encoder: reduce input to quantum-compatible size
        self.encoder = nn.Sequential(
            nn.Linear(input_dim, 512),
            nn.LayerNorm(512),
            nn.GELU(),
            nn.Dropout(0.1),
            nn.Linear(512, 256),
            nn.LayerNorm(256),
            nn.GELU(),
            nn.Dropout(0.1),
            nn.Linear(256, self.q_input_dim),
            nn.Tanh(),  # bound inputs for encoding stability
        )

        # Build quantum circuit
        builders = {
            "amplitude": _build_amplitude_circuit,
            "angle": _build_angle_circuit,
            "iqp": _build_iqp_circuit,
        }
        self.circuit, weight_shape, self.dev = builders[encoding](
            n_qubits, n_layers, entanglement, backend, self.diff_method
        )

        # Trainable quantum weights
        self.q_weights = nn.Parameter(
            0.01 * torch.randn(*weight_shape, dtype=torch.float32)
        )

        # Classical head: map quantum output to classes
        q_output_dim = 2 ** n_qubits  # probability vector
        layers = []
        prev_dim = q_output_dim
        for dim in head_dims:
            layers.extend([
                nn.Linear(prev_dim, dim),
                nn.LayerNorm(dim),
                nn.GELU(),
                nn.Dropout(0.1),
            ])
            prev_dim = dim
        layers.append(nn.Linear(prev_dim, n_classes))
        self.head = nn.Sequential(*layers)

        logger.info(
            f"HybridQC: {encoding} encoding, {n_qubits}q × {n_layers}L, "
            f"input={input_dim}→{self.q_input_dim}→{q_output_dim}→{n_classes}, "
            f"backend={backend}, diff={self.diff_method}"
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: (batch_size, input_dim) — flattened or PCA-reduced features
        Returns:
            logits: (batch_size, n_classes)
        """
        # Classical encoding
        z = self.encoder(x)

        # Quantum processing (loop over batch — PennyLane constraint)
        batch_size = z.shape[0]
        q_out = []
        for i in range(batch_size):
            z_i = torch.nan_to_num(z[i], nan=0.0, posinf=0.0, neginf=0.0)
            if self.encoding == "amplitude":
                norm = torch.linalg.vector_norm(z_i)
                if torch.isfinite(norm) and norm > 1e-12:
                    z_i = z_i / norm
                else:
                    z_i = torch.zeros_like(z_i)
                    z_i[0] = 1.0

            probs = self.circuit(z_i, self.q_weights)
            q_out.append(probs)
        q_out = torch.stack(q_out).to(device=z.device, dtype=z.dtype)

        # Classical head
        logits = self.head(q_out)
        return logits


# ── Purely classical control model (same architecture minus quantum) ────────

class ClassicalControlModel(nn.Module):
    """
    Classical-only control with the same parameter budget as the hybrid model.
    Replaces VQC with equivalent-dim MLP layer.
    """

    def __init__(self, input_dim: int, n_classes: int, hidden_dims: List[int] = [256, 128, 64]):
        super().__init__()
        layers = []
        prev = input_dim
        for h in hidden_dims:
            layers.extend([
                nn.Linear(prev, h),
                nn.BatchNorm1d(h),
                nn.GELU(),
                nn.Dropout(0.2),
            ])
            prev = h
        layers.append(nn.Linear(prev, n_classes))
        self.net = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


# ── Factory ─────────────────────────────────────────────────────────────────

def build_quantum_model(
    input_dim: int,
    n_classes: int,
    n_qubits: int = 8,
    n_layers: int = 4,
    encoding: str = "amplitude",
    entanglement: str = "full",
    backend: str = "default.qubit",
    diff_method: str = "adjoint",
    head_dims: List[int] = [64, 32],
) -> nn.Module:
    return HybridQuantumClassifier(
        input_dim=input_dim,
        n_classes=n_classes,
        n_qubits=n_qubits,
        n_layers=n_layers,
        encoding=encoding,
        entanglement=entanglement,
        backend=backend,
        diff_method=diff_method,
        head_dims=head_dims,
    )


def quantum_param_count(model: HybridQuantumClassifier) -> dict:
    """Detailed parameter count: classical vs quantum."""
    q_params = model.q_weights.numel()
    total = sum(p.numel() for p in model.parameters())
    classical = total - q_params
    return {
        "total": total,
        "quantum": q_params,
        "classical": classical,
        "quantum_fraction": q_params / total if total > 0 else 0,
    }
