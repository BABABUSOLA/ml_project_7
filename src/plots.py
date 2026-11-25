import os
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')                      # no display needed
import matplotlib.pyplot as plt
from sklearn.metrics import roc_curve

from src.metrics import ece
from src.calibration import fit_temperature, apply_temperature
from src.ood import combine, scores, entropy
from src.analysis import load_logits, build_methods

LABELS = {'baseline': 'Single network',
          'ensemble': 'Deep ensemble',
          'mcdropout': 'MC dropout'}


# ---------------------------------------------------------------- helpers
def reliability(probs, labels, ax, n_bins=15, label='', color=None):
    """Confidence against observed accuracy, per bin. Diagonal = calibrated."""
    conf, acc = probs.max(1), (probs.argmax(1) == labels).astype(float)
    edges = np.linspace(0, 1, n_bins + 1)
    xs, ys = [], []
    for lo, hi in zip(edges[:-1], edges[1:]):
        m = (conf > lo) & (conf <= hi)
        if m.sum():
            xs.append(conf[m].mean()); ys.append(acc[m].mean())
    ax.plot(xs, ys, 'o-', label=label, color=color, ms=4)


def confidence_hist(id_conf, ood_conf, ax):
    ax.hist(id_conf, bins=40, alpha=.55, density=True, label='CIFAR-10 (ID)')
    ax.hist(ood_conf, bins=40, alpha=.55, density=True, label='SVHN (OOD)')
    ax.set_xlabel('max softmax probability'); ax.set_ylabel('density')
    ax.legend(fontsize=8)


def roc(id_scores, ood_scores, ax, label=''):
    y = np.r_[np.zeros(len(id_scores)), np.ones(len(ood_scores))]
    fpr, tpr, _ = roc_curve(y, np.r_[id_scores, ood_scores])
    ax.plot(fpr, tpr, label=label)


# ---------------------------------------------------------------- driver
def make_plots(logits_dir, out_dir='results', members=5, n_bins=15):
    """Produce and save every figure. Returns the list of files written."""
    os.makedirs(out_dir, exist_ok=True)
    ens, mcd = load_logits(logits_dir, members)
    y_test, y_val = ens[0]['test_labels'], ens[0]['val_labels']
    methods = build_methods(ens, mcd)
    written = []

    # cache p_bar before/after scaling for every method
    cache = {}
    for name, M in methods.items():
        pv, _ = combine(M['val_logits'])
        T = fit_temperature(np.log(pv + 1e-12), y_val)
        cache[name] = {
            'T': T,
            'none': combine(M['test_logits']),
            'scaled': combine(apply_temperature(M['test_logits'], T)),
            'svhn_scaled': combine(apply_temperature(M['svhn_logits'], T)),
        }

    # 1 ---- reliability diagrams, one panel per method -------------------
    fig, axes = plt.subplots(1, 3, figsize=(12, 3.8), sharey=True)
    for ax, (name, c) in zip(axes, cache.items()):
        ax.plot([0, 1], [0, 1], 'k--', lw=1, label='perfect')
        for key, lbl in [('none', 'unscaled'), ('scaled', f'T={c["T"]:.2f}')]:
            p = c[key][0]
            reliability(p, y_test, ax,
                        label=f'{lbl} (ECE {ece(p, y_test, n_bins):.4f})')
        ax.set_title(LABELS[name]); ax.set_xlabel('confidence')
        ax.legend(fontsize=7, loc='upper left')
    axes[0].set_ylabel('observed accuracy')
    fig.tight_layout(); f = f'{out_dir}/reliability.png'
    fig.savefig(f, dpi=150); plt.close(fig); written.append(f)

    # 2 ---- confidence histograms, ID vs OOD -----------------------------
    fig, axes = plt.subplots(1, 3, figsize=(12, 3.4), sharey=True)
    for ax, (name, c) in zip(axes, cache.items()):
        confidence_hist(c['scaled'][0].max(1), c['svhn_scaled'][0].max(1), ax)
        ax.set_title(LABELS[name])
    fig.tight_layout(); f = f'{out_dir}/confidence_hist.png'
    fig.savefig(f, dpi=150); plt.close(fig); written.append(f)

    # 3 ---- OOD ROC, one panel per method, one line per score ------------
    fig, axes = plt.subplots(1, 3, figsize=(12, 3.8), sharey=True)
    for ax, (name, c) in zip(axes, cache.items()):
        p_id, probs_id = c['scaled']
        p_ood, probs_ood = c['svhn_scaled']
        s_id, s_ood = scores(p_id, probs_id), scores(p_ood, probs_ood)
        for k in s_id:
            roc(s_id[k], s_ood[k], ax, label=k)
        ax.plot([0, 1], [0, 1], 'k--', lw=1)
        ax.set_title(LABELS[name]); ax.set_xlabel('FPR'); ax.legend(fontsize=8)
    axes[0].set_ylabel('TPR')
    fig.tight_layout(); f = f'{out_dir}/ood_roc.png'
    fig.savefig(f, dpi=150); plt.close(fig); written.append(f)

    # 4 ---- accuracy against coverage ------------------------------------
    path = f'{out_dir}/rejection_curve.csv'
    if os.path.exists(path):
        df = pd.read_csv(path)
        fig, ax = plt.subplots(figsize=(5, 3.6))
        for name, g in df.groupby('method'):
            ax.plot(g['coverage'], g['accuracy'], 'o-', ms=4,
                    label=LABELS.get(name, name))
        ax.set_xlabel('coverage (fraction answered)'); ax.set_ylabel('accuracy on retained')
        ax.invert_xaxis(); ax.legend(fontsize=8); ax.grid(alpha=.3)
        fig.tight_layout(); f = f'{out_dir}/rejection_curve.png'
        fig.savefig(f, dpi=150); plt.close(fig); written.append(f)

    # 5 ---- ensemble size sweep ------------------------------------------
    path = f'{out_dir}/size_sweep.csv'
    if os.path.exists(path):
        df = pd.read_csv(path)
        fig, (a1, a2) = plt.subplots(1, 2, figsize=(8, 3.4))
        a1.plot(df['M'], df['acc'], 'o-'); a1.set_xlabel('M'); a1.set_ylabel('accuracy')
        a2.plot(df['M'], df['ece'], 'o-', color='tab:red')
        a2.set_xlabel('M'); a2.set_ylabel('ECE')
        for a in (a1, a2):
            a.set_xticks(df['M']); a.grid(alpha=.3)
        fig.tight_layout(); f = f'{out_dir}/size_sweep.png'
        fig.savefig(f, dpi=150); plt.close(fig); written.append(f)

    return written