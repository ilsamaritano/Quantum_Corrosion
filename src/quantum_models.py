"""Hybrid quantum-classical classifier using PennyLane and PyTorch."""

from __future__ import annotations

import logging
from typing import List, Optional, Tuple

import numpy as np

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Optional imports — graceful degradation if libraries not installed
# ---------------------------------------------------------------------------

try:
    import pennylane as qml

    _PENNYLANE_AVAILABLE = True
except ImportError:
    _PENNYLANE_AVAILABLE = False
    logger.warning(
        "PennyLane not available — quantum model functionality disabled."
    )

try:
    import torch
    import torch.nn as nn

    _TORCH_AVAILABLE = True
except ImportError:
    _TORCH_AVAILABLE = False
    logger.warning("PyTorch not available — quantum model functionality disabled.")


# ---------------------------------------------------------------------------
# Encoding helpers (pure PennyLane, called inside QNodes)
# ---------------------------------------------------------------------------

def encode_amplitude(features: np.ndarray, n_qubits: int) -> None:
    """Amplitude encoding of a normalised feature vector into a quantum state.

    The feature vector is padded / truncated to ``2**n_qubits`` and normalised
    to unit norm before encoding.

    Args:
        features: 1-D float array of length ``<= 2**n_qubits``.
        n_qubits: Number of qubits.
    """
    if not _PENNYLANE_AVAILABLE:
        raise ImportError("PennyLane is required for encode_amplitude.")

    dim = 2**n_qubits
    vec = np.zeros(dim, dtype=float)
    vec[: len(features)] = features[:dim]
    norm = np.linalg.norm(vec)
    if norm < 1e-10:
        vec[0] = 1.0
    else:
        vec /= norm
    qml.AmplitudeEmbedding(vec, wires=range(n_qubits), normalize=False)


def encode_angles(features: np.ndarray, n_qubits: int) -> None:
    """Angle encoding via RY rotations on all qubits.

    Each qubit receives one feature scaled to ``[-π, π]``.  Extra features are
    ignored; missing ones are set to zero.

    Args:
        features: 1-D float array.
        n_qubits: Number of qubits (== number of features used).
    """
    if not _PENNYLANE_AVAILABLE:
        raise ImportError("PennyLane is required for encode_angles.")

    for i in range(n_qubits):
        angle = float(features[i]) * np.pi if i < len(features) else 0.0
        qml.RY(angle, wires=i)


# ---------------------------------------------------------------------------
# Variational quantum circuit builder
# ---------------------------------------------------------------------------

def build_vqc(
    n_qubits: int = 4,
    n_layers: int = 3,
    entanglement: str = "full",
    measurement: str = "probs",
):
    """Build a variational quantum circuit (VQC) as a PennyLane QNode.

    Circuit structure per layer:
    1. RY + RZ rotations on every qubit (variational parameters)
    2. CNOT entanglement ring (or full connectivity)

    Args:
        n_qubits: Number of qubits.
        n_layers: Number of variational layers.
        entanglement: ``"full"`` (all-to-all CNOT pairs) or ``"ring"``
            (nearest-neighbour ring).
        measurement: ``"probs"`` returns a probability vector of length
            ``2**n_qubits``; ``"expval"`` returns Pauli-Z expectation values.

    Returns:
        A callable PennyLane QNode.
    """
    if not _PENNYLANE_AVAILABLE:
        raise ImportError("PennyLane is required for build_vqc.")

    dev = qml.device("default.qubit", wires=n_qubits)

    @qml.qnode(dev, interface="torch")
    def circuit(inputs, weights):
        # Angle encoding
        for i in range(n_qubits):
            qml.RY(inputs[i] * np.pi, wires=i)

        # Variational layers
        for layer in range(n_layers):
            for q in range(n_qubits):
                qml.RY(weights[layer, q, 0], wires=q)
                qml.RZ(weights[layer, q, 1], wires=q)
            # Entanglement
            if entanglement == "full":
                for i in range(n_qubits):
                    for j in range(i + 1, n_qubits):
                        qml.CNOT(wires=[i, j])
            else:  # ring
                for i in range(n_qubits):
                    qml.CNOT(wires=[i, (i + 1) % n_qubits])

        if measurement == "expval":
            return [qml.expval(qml.PauliZ(i)) for i in range(n_qubits)]
        return qml.probs(wires=range(n_qubits))

    return circuit


# ---------------------------------------------------------------------------
# Hybrid classifier nn.Module
# ---------------------------------------------------------------------------

def build_hybrid_quantum_classifier(
    n_qubits: int = 4,
    n_layers: int = 3,
    n_classes: int = 4,
    encoding: str = "angle",
    latent_dim: int = 16,
) -> "nn.Module":
    """Convenience factory that returns a :class:`HybridQuantumClassifier`.

    Args:
        n_qubits: Number of qubits.
        n_layers: Variational circuit depth.
        n_classes: Number of output classes.
        encoding: ``"angle"`` or ``"amplitude"``.
        latent_dim: Input feature dimension (after dimensionality reduction).

    Returns:
        Initialised :class:`HybridQuantumClassifier` instance.
    """
    return HybridQuantumClassifier(
        input_dim=latent_dim,
        n_qubits=n_qubits,
        n_layers=n_layers,
        n_classes=n_classes,
        encoding=encoding,
    )


if _TORCH_AVAILABLE and _PENNYLANE_AVAILABLE:

    class HybridQuantumClassifier(nn.Module):
        """Hybrid quantum-classical classifier.

        Architecture:
        ``Linear(input_dim → n_qubits) → QNode (VQC) → Linear(2**n_qubits → n_classes)``

        The quantum circuit uses angle encoding and a variational block with
        RY/RZ gates and CNOT entanglement.

        Args:
            input_dim: Dimension of input feature vector.
            n_qubits: Number of qubits in the VQC.
            n_layers: Number of variational layers.
            n_classes: Number of output classes.
            encoding: ``"angle"`` or ``"amplitude"``.
        """

        def __init__(
            self,
            input_dim: int,
            n_qubits: int = 4,
            n_layers: int = 3,
            n_classes: int = 4,
            encoding: str = "angle",
        ) -> None:
            super().__init__()
            self.n_qubits = n_qubits
            self.n_layers = n_layers
            self.encoding = encoding

            # Classical pre-processing: compress input to n_qubits features
            self.pre = nn.Sequential(
                nn.Linear(input_dim, n_qubits),
                nn.Tanh(),
            )

            # Quantum device + QNode
            dev = qml.device("default.qubit", wires=n_qubits)

            @qml.qnode(dev, interface="torch", diff_method="best")
            def _qnode(inputs, weights):
                # Angle encoding
                for i in range(n_qubits):
                    qml.RY(inputs[i] * np.pi, wires=i)
                # Variational layers
                for layer in range(n_layers):
                    for q in range(n_qubits):
                        qml.RY(weights[layer, q, 0], wires=q)
                        qml.RZ(weights[layer, q, 1], wires=q)
                    for i in range(n_qubits):
                        qml.CNOT(wires=[i, (i + 1) % n_qubits])
                return qml.probs(wires=range(n_qubits))

            self.qnode = _qnode

            # Quantum weight tensor (learnable)
            weight_shape = (n_layers, n_qubits, 2)
            self.q_weights = nn.Parameter(
                torch.randn(*weight_shape) * 0.1
            )

            # Classical output head
            q_out_dim = 2**n_qubits
            self.post = nn.Sequential(
                nn.Linear(q_out_dim, n_classes),
            )

        def forward(self, x: "torch.Tensor") -> "torch.Tensor":
            """Forward pass through the hybrid network.

            Args:
                x: Batch tensor of shape ``(batch_size, input_dim)``.

            Returns:
                Logit tensor of shape ``(batch_size, n_classes)``.
            """
            z = self.pre(x)  # (B, n_qubits)
            q_outs = torch.stack(
                [self.qnode(z[i], self.q_weights) for i in range(z.shape[0])]
            )  # (B, 2**n_qubits)
            return self.post(q_outs)

else:
    # Stub when dependencies are missing
    class HybridQuantumClassifier:  # type: ignore[no-redef]
        """Stub when PennyLane or PyTorch is not installed."""

        def __init__(self, *args, **kwargs) -> None:
            raise ImportError(
                "PennyLane and PyTorch are both required for HybridQuantumClassifier."
            )


# ---------------------------------------------------------------------------
# Training / evaluation
# ---------------------------------------------------------------------------

def train_quantum_model(
    model: "HybridQuantumClassifier",
    train_loader: "torch.utils.data.DataLoader",
    val_loader: "torch.utils.data.DataLoader",
    n_epochs: int = 50,
    lr: float = 0.01,
    optimizer_name: str = "adam",
    device: str = "cpu",
) -> dict:
    """Training loop for a hybrid quantum model with early stopping (patience=10).

    Args:
        model: A :class:`HybridQuantumClassifier` instance.
        train_loader: Training data loader.
        val_loader: Validation data loader.
        n_epochs: Maximum number of epochs.
        lr: Learning rate.
        optimizer_name: ``"adam"`` or ``"sgd"``.
        device: Torch device string.

    Returns:
        History dictionary with keys ``"train_loss"``, ``"val_loss"``,
        ``"train_acc"``, ``"val_acc"``.
    """
    if not _TORCH_AVAILABLE:
        raise ImportError("PyTorch is required for train_quantum_model.")

    import torch

    model = model.to(device)
    criterion = torch.nn.CrossEntropyLoss()

    if optimizer_name.lower() == "adam":
        optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    else:
        optimizer = torch.optim.SGD(model.parameters(), lr=lr, momentum=0.9)

    history = {"train_loss": [], "val_loss": [], "train_acc": [], "val_acc": []}
    best_val_loss = float("inf")
    patience = 10
    patience_counter = 0

    for epoch in range(n_epochs):
        # Training
        model.train()
        t_loss, t_correct, t_total = 0.0, 0, 0
        for X_batch, y_batch in train_loader:
            X_batch, y_batch = X_batch.to(device), y_batch.to(device)
            optimizer.zero_grad()
            logits = model(X_batch.float())
            loss = criterion(logits, y_batch.long())
            loss.backward()
            optimizer.step()
            t_loss += loss.item() * len(y_batch)
            t_correct += (logits.argmax(1) == y_batch).sum().item()
            t_total += len(y_batch)

        # Validation
        model.eval()
        v_loss, v_correct, v_total = 0.0, 0, 0
        with torch.no_grad():
            for X_batch, y_batch in val_loader:
                X_batch, y_batch = X_batch.to(device), y_batch.to(device)
                logits = model(X_batch.float())
                loss = criterion(logits, y_batch.long())
                v_loss += loss.item() * len(y_batch)
                v_correct += (logits.argmax(1) == y_batch).sum().item()
                v_total += len(y_batch)

        train_loss = t_loss / t_total
        val_loss = v_loss / v_total
        train_acc = t_correct / t_total
        val_acc = v_correct / v_total

        history["train_loss"].append(train_loss)
        history["val_loss"].append(val_loss)
        history["train_acc"].append(train_acc)
        history["val_acc"].append(val_acc)

        logger.info(
            "Epoch %d/%d — train_loss: %.4f, val_loss: %.4f, train_acc: %.4f, val_acc: %.4f",
            epoch + 1, n_epochs, train_loss, val_loss, train_acc, val_acc,
        )

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            patience_counter = 0
        else:
            patience_counter += 1
            if patience_counter >= patience:
                logger.info("Early stopping triggered at epoch %d.", epoch + 1)
                break

    return history


def evaluate_quantum_model(
    model: "HybridQuantumClassifier",
    data_loader: "torch.utils.data.DataLoader",
    device: str = "cpu",
) -> Tuple[float, np.ndarray, np.ndarray]:
    """Evaluate the quantum model on a data loader.

    Args:
        model: Trained :class:`HybridQuantumClassifier`.
        data_loader: Data loader with ``(X, y)`` batches.
        device: Torch device string.

    Returns:
        Tuple ``(accuracy, predictions, true_labels)``.
    """
    if not _TORCH_AVAILABLE:
        raise ImportError("PyTorch is required for evaluate_quantum_model.")

    import torch

    model.eval().to(device)
    all_preds: List[int] = []
    all_true: List[int] = []

    with torch.no_grad():
        for X_batch, y_batch in data_loader:
            X_batch = X_batch.to(device).float()
            logits = model(X_batch)
            preds = logits.argmax(1).cpu().numpy().tolist()
            all_preds.extend(preds)
            all_true.extend(y_batch.numpy().tolist())

    y_pred = np.array(all_preds)
    y_true = np.array(all_true)
    accuracy = float((y_pred == y_true).mean())
    return accuracy, y_pred, y_true


def get_quantum_circuit_diagram(
    n_qubits: int = 4,
    n_layers: int = 3,
    encoding: str = "angle",
) -> str:
    """Return a text representation of the quantum circuit.

    Args:
        n_qubits: Number of qubits.
        n_layers: Number of variational layers.
        encoding: Encoding type name (for display only).

    Returns:
        Multi-line ASCII string describing the circuit.
    """
    lines = [
        f"Quantum Circuit: {encoding} encoding, {n_qubits} qubits, {n_layers} layers",
        "=" * 60,
    ]
    for q in range(n_qubits):
        gate_str = f"|0⟩ -[RY(x{q})]"
        for layer in range(n_layers):
            gate_str += f"-[RY(θ{layer}{q})][RZ(φ{layer}{q})]"
            if q < n_qubits - 1:
                gate_str += "-●"
            else:
                gate_str += "-(CNOT ring)"
        gate_str += "-[M]"
        lines.append(f"q{q}: {gate_str}")
    lines.append("=" * 60)
    lines.append(f"Output: probs(wires=0..{n_qubits-1}) → Linear({2**n_qubits} → n_classes)")
    return "\n".join(lines)
