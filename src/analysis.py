import os
import numpy as np
import pandas as pd

from src.metrics import accuracy, nll, ece, brier_multiclass, auroc, fpr_at_95tpr
from src.calibration import fit_temperature, apply_temperature
from src.ood import combine, scores

KEYS = ['val_logits', 'test_logits', 'svhn_logits']


def load_logits(logits_dir, members=5):
    """Read every saved logits file into memory."""
    ens = [dict(np.load(f'{logits_dir}/ens_s{s}.npz')) for s in range(members)]
    path = f'{logits_dir}/mcdropout.npz'
    mcd = dict(np.load(path)) if os.path.exists(path) else None
    return ens, mcd


def build_methods(ens, mcd):
    """Three methods, each shaped (M, N, C). Baseline is one member with M=1."""
    keys = ['val_logits', 'test_logits', 'svhn_logits']
    m = {'baseline': {k: ens[0][k][None] for k in keys},
         'ensemble': {k: np.stack([e[k] for e in ens]) for k in keys}}
    if mcd is not None:
        m['mcdropout'] = {k: mcd[k] for k in keys}
    return m


def run_analysis(logits_dir, out_dir='results', members=5):
    """Main table: 3 methods x {unscaled, temperature-scaled}."""
    ens, mcd = load_logits(logits_dir, members)
    y_test, y_val = ens[0]['test_labels'], ens[0]['val_labels']

    rows = []
    for name, M in build_methods(ens, mcd).items():
        for scaled in (False, True):
            T = 1.0
            if scaled:
                pv, _ = combine(M['val_logits'])
                T = fit_temperature(np.log(pv + 1e-12), y_val)

            p_bar, probs    = combine(apply_temperature(M['test_logits'], T))
            sv,    probs_sv = combine(apply_temperature(M['svhn_logits'], T))

            row = {'method': name,
                   'scaling': 'T' if scaled else 'none',
                   'T': round(T, 3),
                   'acc':   accuracy(p_bar, y_test),
                   'nll':   nll(p_bar, y_test),
                   'ece':   ece(p_bar, y_test),
                   'brier': brier_multiclass(p_bar, y_test)}

            s_id, s_ood = scores(p_bar, probs), scores(sv, probs_sv)
            for k in s_id:
                row[f'auroc_{k}'] = auroc(s_id[k], s_ood[k])
                row[f'fpr95_{k}'] = fpr_at_95tpr(s_id[k], s_ood[k])
            rows.append(row)

    os.makedirs(out_dir, exist_ok=True)
    df = pd.DataFrame(rows)
    df.to_csv(f'{out_dir}/main_table.csv', index=False)
    return df


def size_sweep(logits_dir, out_dir='results', sizes=(1, 3, 5), members=5):
    """How much does the ensemble gain per added member? Free - subsets saved logits."""
    ens, _ = load_logits(logits_dir, members)
    y_test = ens[0]['test_labels']
    rows = []
    for M in sizes:
        p_bar, _ = combine(np.stack([ens[i]['test_logits'] for i in range(M)]))
        rows.append({'M': M,
                     'acc':   accuracy(p_bar, y_test),
                     'ece':   ece(p_bar, y_test),
                     'brier': brier_multiclass(p_bar, y_test)})
    os.makedirs(out_dir, exist_ok=True)
    df = pd.DataFrame(rows)
    df.to_csv(f'{out_dir}/size_sweep.csv', index=False)
    return df


def single_member_spread(logits_dir, members=5):
    """The M members are M independent single networks - report mean +/- std."""
    ens, _ = load_logits(logits_dir, members)
    y_test = ens[0]['test_labels']
    accs = [accuracy(combine(e['test_logits'])[0], y_test) for e in ens]
    eces = [ece(combine(e['test_logits'])[0], y_test) for e in ens]
    return {'acc_mean': float(np.mean(accs)), 'acc_std': float(np.std(accs, ddof=1)),
            'ece_mean': float(np.mean(eces)), 'ece_std': float(np.std(eces, ddof=1))}
