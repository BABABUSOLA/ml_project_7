import os
import numpy as np
import pandas as pd

from src.metrics import accuracy, nll, ece, brier_multiclass, auroc, fpr_at_95tpr
from src.calibration import fit_temperature, apply_temperature
from src.ood import combine, scores

KEYS = ['val_logits', 'test_logits', 'ood_logits']


def _normalise(d):
    """Logits files written before the OOD rename store 'svhn_logits'.
    Map them forward so old CIFAR files and new MNIST files both load."""
    if 'ood_logits' not in d and 'svhn_logits' in d:
        d['ood_logits'] = d['svhn_logits']
        if 'svhn_labels' in d:
            d['ood_labels'] = d['svhn_labels']
    return d


def load_logits(logits_dir, members=5, dataset='cifar10'):
    ens = [_normalise(dict(np.load(f'{logits_dir}/{dataset}_ens_s{s}.npz')))
           for s in range(members)]
    path = f'{logits_dir}/{dataset}_mcdropout.npz'
    mcd = _normalise(dict(np.load(path))) if os.path.exists(path) else None
    return ens, mcd


def build_methods(ens, mcd):
    m = {'baseline': {k: ens[0][k][None] for k in KEYS},
         'ensemble': {k: np.stack([e[k] for e in ens]) for k in KEYS}}
    if mcd is not None:
        m['mcdropout'] = {k: mcd[k] for k in KEYS}
    return m


def _out(out_dir, dataset):
    p = os.path.join(out_dir, dataset)
    os.makedirs(p, exist_ok=True)
    return p


def run_analysis(logits_dir, out_dir='results', members=5, dataset='cifar10'):
    ens, mcd = load_logits(logits_dir, members, dataset)
    y_test, y_val = ens[0]['test_labels'], ens[0]['val_labels']
    rows = []
    for name, M in build_methods(ens, mcd).items():
        for scaled in (False, True):
            T = 1.0
            if scaled:
                pv, _ = combine(M['val_logits'])
                T = fit_temperature(np.log(pv + 1e-12), y_val)
            p_bar, probs   = combine(apply_temperature(M['test_logits'], T))
            po,    probs_o = combine(apply_temperature(M['ood_logits'], T))
            row = {'method': name, 'scaling': 'T' if scaled else 'none',
                   'T': round(T, 3),
                   'acc': accuracy(p_bar, y_test), 'nll': nll(p_bar, y_test),
                   'ece': ece(p_bar, y_test), 'brier': brier_multiclass(p_bar, y_test)}
            s_id, s_ood = scores(p_bar, probs), scores(po, probs_o)
            for k in s_id:
                row[f'auroc_{k}'] = auroc(s_id[k], s_ood[k])
                row[f'fpr95_{k}'] = fpr_at_95tpr(s_id[k], s_ood[k])
            rows.append(row)
    df = pd.DataFrame(rows)
    df.to_csv(f'{_out(out_dir, dataset)}/main_table.csv', index=False)
    return df


def size_sweep(logits_dir, out_dir='results', sizes=(1, 3, 5), members=5,
               dataset='cifar10'):
    ens, _ = load_logits(logits_dir, members, dataset)
    y_test = ens[0]['test_labels']
    rows = []
    for M in sizes:
        p_bar, _ = combine(np.stack([ens[i]['test_logits'] for i in range(M)]))
        rows.append({'M': M, 'acc': accuracy(p_bar, y_test),
                     'ece': ece(p_bar, y_test),
                     'brier': brier_multiclass(p_bar, y_test)})
    df = pd.DataFrame(rows)
    df.to_csv(f'{_out(out_dir, dataset)}/size_sweep.csv', index=False)
    return df


def single_member_spread(logits_dir, members=5, dataset='cifar10'):
    ens, _ = load_logits(logits_dir, members, dataset)
    y_test = ens[0]['test_labels']
    accs = [accuracy(combine(e['test_logits'])[0], y_test) for e in ens]
    eces = [ece(combine(e['test_logits'])[0], y_test) for e in ens]
    return {'acc_mean': float(np.mean(accs)), 'acc_std': float(np.std(accs, ddof=1)),
            'ece_mean': float(np.mean(eces)), 'ece_std': float(np.std(eces, ddof=1))}