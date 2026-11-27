import os
import numpy as np
import pandas as pd
from scipy import stats

from src.metrics import accuracy, auroc
from src.calibration import fit_temperature, apply_temperature
from src.ood import combine, scores
from src.analysis import load_logits, build_methods, _out


def misclassification_auroc(p_bar, labels, score):
    """Does uncertainty separate the model's own errors from its correct
    answers? One-vs-all framing: errors are the positive class."""
    correct = (p_bar.argmax(1) == labels)
    return auroc(score[correct], score[~correct])


def rejection_curve(p_bar, labels, score, rates=np.arange(0, 0.55, 0.05)):
    """Abstain on the most uncertain fraction; accuracy on what is kept."""
    order = np.argsort(score)                      # low score = confident
    out = []
    for r in rates:
        keep = order[: int(len(order) * (1 - r))]
        out.append({'reject_rate': round(float(r), 2),
                    'coverage': round(1 - float(r), 2),
                    'accuracy': accuracy(p_bar[keep], labels[keep])})
    return out


def paired_ttest(p_a, p_b, labels):
    """Hypothesis testing (paired t-test). c_i = a_i - b_i on per-example
    errors; T = c_bar * sqrt(K) / s at K-1 degrees of freedom."""
    a = (p_a.argmax(1) != labels).astype(float)
    b = (p_b.argmax(1) != labels).astype(float)
    c = a - b
    K, c_bar, s = len(c), c.mean(), c.std(ddof=1)
    T = c_bar * np.sqrt(K) / s if s > 0 else 0.0
    p = 2 * (1 - stats.t.cdf(abs(T), df=K - 1))
    return {'mean_err_A': a.mean(), 'mean_err_B': b.mean(),
            'T': float(T), 'df': K - 1, 'p': float(p),
            'significant_at_0.05': bool(p < 0.05)}


def run_extra(logits_dir, out_dir='results', members=5, dataset='cifar10'):
    """Misclassification detection, rejection curves and paired t-tests."""
    ens, mcd = load_logits(logits_dir, members, dataset)
    y_test, y_val = ens[0]['test_labels'], ens[0]['val_labels']
    methods = build_methods(ens, mcd)

    pbars, mis, rej = {}, [], []
    for name, M in methods.items():
        pv, _ = combine(M['val_logits'])
        T = fit_temperature(np.log(pv + 1e-12), y_val)
        p_bar, probs = combine(apply_temperature(M['test_logits'], T))
        pbars[name] = p_bar

        for sname, sc in scores(p_bar, probs).items():
            mis.append({'method': name, 'score': sname,
                        'auroc_misclassification':
                            misclassification_auroc(p_bar, y_test, sc)})
            if sname == 'entropy':
                for row in rejection_curve(p_bar, y_test, sc):
                    rej.append({'method': name, **row})

    sig = []
    for A, B in [('ensemble', 'baseline'),
                 ('ensemble', 'mcdropout'),
                 ('mcdropout', 'baseline')]:
        if A in pbars and B in pbars:        # skip pairs needing an absent method
            sig.append({'A': A, 'B': B, **paired_ttest(pbars[A], pbars[B], y_test)})

    out = _out(out_dir, dataset)
    dfs = {}
    for key, rows in [('misclassification', mis),
                      ('rejection_curve', rej),
                      ('significance', sig)]:
        dfs[key] = pd.DataFrame(rows)
        dfs[key].to_csv(f'{out}/{key}.csv', index=False)
    return dfs