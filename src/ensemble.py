"""
Ensemble Models — Hybrid Quantum-Classical + Deep Learning Fusion
==================================================================
Combines pre-trained EfficientNet-B0 with Quantum VQC for superior accuracy.
Multiple fusion strategies: weighted voting, stacking, soft-voting.
"""

import numpy as np
import torch
import torch.nn as nn
from typing import Dict, List, Tuple, Optional
import logging
from pathlib import Path
import pickle

logger = logging.getLogger(__name__)


class EnsembleClassifier(nn.Module):
    """
    Ensemble combining classical (EfficientNet) and quantum (VQC) models.
    
    Fusion strategies:
    - 'weighted_avg': weighted average of logits
    - 'voting': hard voting from predicted classes
    - 'stacking': learned linear combination via meta-learner
    """
    
    def __init__(
        self,
        classical_model: nn.Module,
        quantum_model: nn.Module,
        n_classes: int = 5,
        fusion_method: str = "weighted_avg",
        classical_weight: float = 0.65,
        quantum_weight: float = 0.35,
    ):
        super().__init__()
        self.classical_model = classical_model
        self.quantum_model = quantum_model
        self.n_classes = n_classes
        self.fusion_method = fusion_method
        
        # Normalize weights
        total = classical_weight + quantum_weight
        self.classical_weight = classical_weight / total
        self.quantum_weight = quantum_weight / total
        
        logger.info(
            f"EnsembleClassifier: fusion={fusion_method}, "
            f"weights=[classical={self.classical_weight:.2f}, quantum={self.quantum_weight:.2f}]"
        )
        
        # Meta-learner for stacking
        if fusion_method == "stacking":
            self.meta_learner = nn.Sequential(
                nn.Linear(2 * n_classes, 64),
                nn.ReLU(),
                nn.Dropout(0.3),
                nn.Linear(64, n_classes)
            )
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass combining both models.
        
        Args:
            x: input tensor
        
        Returns:
            logits combining classical and quantum predictions
        """
        # Classical branch
        classical_logits = self.classical_model(x)
        
        # Quantum branch
        quantum_logits = self.quantum_model(x)
        
        # Fusion
        if self.fusion_method == "weighted_avg":
            fused_logits = (
                self.classical_weight * classical_logits +
                self.quantum_weight * quantum_logits
            )
        
        elif self.fusion_method == "voting":
            # Hard voting
            classical_preds = torch.argmax(classical_logits, dim=1)
            quantum_preds = torch.argmax(quantum_logits, dim=1)
            fused_logits = self._voting_to_logits(classical_preds, quantum_preds)
        
        elif self.fusion_method == "stacking":
            # Concatenate and pass through meta-learner
            combined = torch.cat([classical_logits, quantum_logits], dim=1)
            fused_logits = self.meta_learner(combined)
        
        else:
            raise ValueError(f"Unknown fusion_method: {self.fusion_method}")
        
        return fused_logits
    
    def _voting_to_logits(
        self, 
        classical_preds: torch.Tensor,
        quantum_preds: torch.Tensor
    ) -> torch.Tensor:
        """Convert hard voting to logits (one-hot of winning class)."""
        batch_size = classical_preds.shape[0]
        
        # Majority vote
        votes = torch.zeros(batch_size, self.n_classes, device=classical_preds.device)
        votes.scatter_add_(1, classical_preds.unsqueeze(1), 1.0)
        votes.scatter_add_(1, quantum_preds.unsqueeze(1), 1.0)
        
        # Convert to log-space for numerical stability
        winner = torch.argmax(votes, dim=1)
        logits = torch.full(
            (batch_size, self.n_classes),
            float('-inf'),
            device=classical_preds.device
        )
        logits.scatter_(1, winner.unsqueeze(1), 1.0)
        
        return logits
    
    def get_predictions(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Get ensemble predictions with confidence scores.
        
        Returns:
            (predicted_classes, confidence_scores)
        """
        with torch.no_grad():
            classical_logits = self.classical_model(x)
            quantum_logits = self.quantum_model(x)
        
        # Classical probs
        classical_probs = torch.nn.functional.softmax(classical_logits, dim=1)
        classical_preds = torch.argmax(classical_logits, dim=1)
        classical_conf = torch.max(classical_probs, dim=1)[0]
        
        # Quantum probs
        quantum_probs = torch.nn.functional.softmax(quantum_logits, dim=1)
        quantum_preds = torch.argmax(quantum_logits, dim=1)
        quantum_conf = torch.max(quantum_probs, dim=1)[0]
        
        # Consensus voting
        agreement = (classical_preds == quantum_preds).float()
        
        # Weighted confidence
        ensemble_conf = (
            self.classical_weight * classical_conf +
            self.quantum_weight * quantum_conf
        )
        
        # Consensus boost confidence
        ensemble_conf = ensemble_conf * (0.7 + 0.3 * agreement)
        
        # Final predictions weighted by method
        if self.fusion_method == "voting":
            final_preds = torch.where(
                agreement.bool(),
                classical_preds,  # Unanimous
                torch.where(
                    classical_conf > quantum_conf,
                    classical_preds,
                    quantum_preds
                )
            )
        else:
            logits = self.forward(x)
            final_preds = torch.argmax(logits, dim=1)
        
        return final_preds, ensemble_conf


def load_ensemble_models(
    classical_model_path: Path,
    quantum_model_path: Path,
    pca_reducer_path: Optional[Path] = None,
    device: str = "cuda",
) -> Tuple[nn.Module, nn.Module, Optional[object]]:
    """
    Load pre-trained classical and quantum models.
    
    Args:
        classical_model_path: path to saved EfficientNet weights
        quantum_model_path: path to saved VQC weights
        pca_reducer_path: path to PCA transformer (if using)
        device: 'cuda' or 'cpu'
    
    Returns:
        (classical_model, quantum_model, pca_reducer)
    """
    logger.info(f"Loading ensemble models from disk...")
    
    # Load classical model
    classical_model = torch.load(classical_model_path, map_location=device)
    logger.info(f"  Loaded classical from {classical_model_path}")
    
    # Load quantum model
    quantum_model = torch.load(quantum_model_path, map_location=device)
    logger.info(f"  Loaded quantum from {quantum_model_path}")
    
    # Load PCA reducer if provided
    pca_reducer = None
    if pca_reducer_path and pca_reducer_path.exists():
        with open(pca_reducer_path, 'rb') as f:
            pca_reducer = pickle.load(f)
        logger.info(f"  Loaded PCA reducer from {pca_reducer_path}")
    
    return classical_model, quantum_model, pca_reducer


def evaluate_ensemble(
    ensemble: nn.Module,
    dataloader,
    device: str = "cuda",
    n_classes: int = 5,
) -> Dict[str, float]:
    """
    Evaluate ensemble on a dataset.
    
    Returns:
        dict with accuracy, precision, recall, f1
    """
    from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
    
    ensemble.eval()
    all_preds = []
    all_targets = []
    all_confs = []
    
    with torch.no_grad():
        for batch_x, batch_y in dataloader:
            batch_x = batch_x.to(device)
            logits = ensemble(batch_x)
            preds = torch.argmax(logits, dim=1).cpu().numpy()
            
            all_preds.extend(preds)
            all_targets.extend(batch_y.numpy())

    all_preds = np.array(all_preds)
    all_targets = np.array(all_targets)
    
    metrics = {
        'accuracy': accuracy_score(all_targets, all_preds),
        'precision': precision_score(all_targets, all_preds, average='weighted', zero_division=0),
        'recall': recall_score(all_targets, all_preds, average='weighted', zero_division=0),
        'f1': f1_score(all_targets, all_preds, average='weighted', zero_division=0),
    }
    
    logger.info(
        f"Ensemble Metrics: Acc={metrics['accuracy']:.4f} "
        f"P={metrics['precision']:.4f} R={metrics['recall']:.4f} F1={metrics['f1']:.4f}"
    )
    
    return metrics
