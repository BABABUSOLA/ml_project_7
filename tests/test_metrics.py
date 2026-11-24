#The GATE

#Run pytest tests/ -v

import numpy as np
from src.metrics import ece, brier_multiclass, nll, softmax

def test_ece_perfectly_calibrated():
    probs = np.array([[0.5, 0.5], [0.5, 0.5]]); labels = np.array([0, 1])
    assert abs(ece(probs, labels)) < 1e-9

def test_ece_max_overconfident():
    probs = np.array([[1.0, 0.0], [1.0, 0.0]]); labels = np.array([0, 1])
    assert abs(ece(probs, labels) - 0.5) < 1e-9

def test_brier_perfect():
    probs = np.eye(3); labels = np.arange(3)
    assert abs(brier_multiclass(probs, labels)) < 1e-12

def test_brier_uniform_10class():
    probs = np.full((4, 10), 0.1); labels = np.zeros(4, int)
    assert abs(brier_multiclass(probs, labels) - 0.9) < 1e-9   # 0.81 + 9*0.01

def test_nll_matches_torch():
    import torch, torch.nn.functional as F
    rng = np.random.RandomState(0)
    logits = rng.randn(20, 5); labels = rng.randint(0, 5, 20)
    ref = F.cross_entropy(torch.tensor(logits), torch.tensor(labels)).item()
    assert abs(nll(softmax(logits), labels) - ref) < 1e-6