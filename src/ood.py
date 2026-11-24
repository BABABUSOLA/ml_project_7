#msp is negated because max-probability is higher for in-distribution. Forget that and AUROC comes out near 0.15.

import numpy as np
from src.metrics import softmax

def entropy(p, eps=1e-12):
    return -(p * np.log(p + eps)).sum(-1)

def combine(logits_set):
    """logits_set: (M, N, C) for ensembles/MC-dropout, or (N, C) for a single net.
       Returns p_bar (N, C) and the per-member probs (M, N, C)."""
    if logits_set.ndim == 2:
        logits_set = logits_set[None]
    probs = softmax(logits_set)
    return probs.mean(0), probs

def scores(p_bar, probs):
    """All three computed on p_bar, not by averaging per-model scores.
       Higher = more OOD."""
    s = {'msp': -p_bar.max(-1),
         'entropy': entropy(p_bar)}
    if probs.shape[0] > 1:
        s['mutual_info'] = entropy(p_bar) - entropy(probs).mean(0)
    return s