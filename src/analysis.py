import numpy as np, pandas as pd
from src.metrics import accuracy, nll, ece, brier_multiclass, auroc, fpr_at_95tpr
from src.calibration import fit_temperature, apply_temperature
from src.ood import combine, scores


def build_methods(ens, mcd):
    keys = ['val_logits', 'test_logits', 'svhn_logits']
    return {
        'baseline':  {k: ens[0][k][None] for k in keys},
        'ensemble':  {k: np.stack([e[k] for e in ens]) for k in keys},
        'mcdropout': {k: mcd[k] for k in keys},
    }


def run_analysis(logits_dir, out_dir='results', members=5):
    ens = [dict(np.load(f'{logits_dir}/ens_s{s}.npz')) for s in range(members)]
    mcd = dict(np.load(f'{logits_dir}/mcdropout.npz'))
    y_test, y_val = ens[0]['test_labels'], ens[0]['val_labels']

    rows = []
    for name, M in build_methods(ens, mcd).items():
        for scaled in [False, True]:
            T = 1.0
            if scaled:
                pv, _ = combine(M['val_logits'])
                T = fit_temperature(np.log(pv + 1e-12), y_val)
            p_bar, probs    = combine(apply_temperature(M['test_logits'], T))
            sv,    probs_sv = combine(apply_temperature(M['svhn_logits'], T))

            row = {'method': name, 'scaling': 'T' if scaled else 'none',
                   'T': round(T, 3),
                   'acc': accuracy(p_bar, y_test), 'nll': nll(p_bar, y_test),
                   'ece': ece(p_bar, y_test),
                   'brier': brier_multiclass(p_bar, y_test)}
            s_id, s_ood = scores(p_bar, probs), scores(sv, probs_sv)
            for k in s_id:
                row[f'auroc_{k}'] = auroc(s_id[k], s_ood[k])
                row[f'fpr95_{k}'] = fpr_at_95tpr(s_id[k], s_ood[k])
            rows.append(row)

    import os; os.makedirs(out_dir, exist_ok=True)
    df = pd.DataFrame(rows)
    df.to_csv(f'{out_dir}/main_table.csv', index=False)
    return df


def size_sweep(ens, y_test, sizes=(1, 3, 5)):
    out = []
    for M in sizes:
        sub = np.stack([ens[i]['test_logits'] for i in range(M)])
        p_bar, _ = combine(sub)
        out.append({'M': M, 'acc': accuracy(p_bar, y_test),
                    'ece': ece(p_bar, y_test),
                    'brier': brier_multiclass(p_bar, y_test)})
    return pd.DataFrame(out)