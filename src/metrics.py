import numpy as np
from sklearn.metrics import roc_auc_score

def softmax(logits):
    z = logits - logits.max(axis=-1, keepdims=True)
    e = np.exp(z)
    return e / e.sum(axis=-1, keepdims=True)

def ece(probs, labels, n_bins=15):
    conf, pred = probs.max(1), probs.argmax(1)
    acc = (pred == labels).astype(float)
    edges, total = np.linspace(0, 1, n_bins + 1), 0.0
    for lo, hi in zip(edges[:-1], edges[1:]):
        m = (conf > lo) & (conf <= hi)
        if m.sum():
            total += m.mean() * abs(acc[m].mean() - conf[m].mean())
    return float(total)

def brier_multiclass(probs, labels):
    onehot = np.zeros_like(probs)
    onehot[np.arange(len(labels)), labels] = 1.0
    return float(((probs - onehot) ** 2).sum(1).mean())

def nll(probs, labels, eps=1e-12):
    return float(-np.log(probs[np.arange(len(labels)), labels] + eps).mean())

def accuracy(probs, labels):
    return float((probs.argmax(1) == labels).mean())

def auroc(id_scores, ood_scores):
    y = np.r_[np.zeros(len(id_scores)), np.ones(len(ood_scores))]
    return float(roc_auc_score(y, np.r_[id_scores, ood_scores]))

def fpr_at_95tpr(id_scores, ood_scores):
    """Threshold keeps 95% of in-distribution; FPR is the OOD fraction wrongly kept."""
    thr = np.quantile(id_scores, 0.95)
    return float((ood_scores <= thr).mean())