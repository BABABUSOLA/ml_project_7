import numpy as np
from scipy import stats
from src.metrics import auroc, accuracy
from src.ood import entropy

# ---- 1. Misclassification detection: Hendrycks & Gimpel [4], the other half ----
def misclassification_auroc(p_bar, labels, score):
    # Does uncertainty separate the model's own errors from its correct answers?
    # Framed as the one-vs-all binary problem from Lecture 8.5.
    correct = (p_bar.argmax(1) == labels)
    return auroc(score[correct], score[~correct])      # errors are the 'positive' class

# ---- 2. Accuracy vs rejection rate ----
def rejection_curve(p_bar, labels, score, rates=np.arange(0, 0.55, 0.05)):
    # Abstain on the most uncertain fraction; measure accuracy on what is kept.
    # This is the act / flag / reject policy from the Application section, quantified.
    order = np.argsort(score)                          # low score = confident
    out = []
    for r in rates:
        keep = order[: int(len(order) * (1 - r))]
        out.append((float(r), accuracy(p_bar[keep], labels[keep])))
    return out

# ---- 3. Paired T-test, Lecture 8.5 ----
def paired_ttest(p_a, p_b, labels):
    # a_i, b_i are per-example errors; c_i = a_i - b_i; T = c_bar * sqrt(K) / s
    a = (p_a.argmax(1) != labels).astype(float)
    b = (p_b.argmax(1) != labels).astype(float)
    c = a - b
    K = len(c); c_bar = c.mean(); s = c.std(ddof=1)
    T = c_bar * np.sqrt(K) / s if s > 0 else 0.0
    p = 2 * (1 - stats.t.cdf(abs(T), df=K - 1))
    return {'T': float(T), 'df': K - 1, 'p': float(p), 'significant': bool(p < 0.05)}