import numpy as np, matplotlib.pyplot as plt
from sklearn.metrics import roc_curve

def reliability(probs, labels, n_bins=15, ax=None, title=''):
    conf, acc = probs.max(1), (probs.argmax(1) == labels).astype(float)
    edges = np.linspace(0, 1, n_bins + 1)
    xs, ys = [], []
    for lo, hi in zip(edges[:-1], edges[1:]):
        m = (conf > lo) & (conf <= hi)
        if m.sum():
            xs.append(conf[m].mean()); ys.append(acc[m].mean())
    ax = ax or plt.gca()
    ax.plot([0,1],[0,1],'k--',lw=1); ax.plot(xs, ys, 'o-')
    ax.set_xlabel('confidence'); ax.set_ylabel('accuracy'); ax.set_title(title)

def confidence_hist(id_conf, ood_conf, ax=None):
    ax = ax or plt.gca()
    ax.hist(id_conf, bins=40, alpha=.6, density=True, label='CIFAR-10')
    ax.hist(ood_conf, bins=40, alpha=.6, density=True, label='SVHN')
    ax.set_xlabel('max softmax probability'); ax.legend()

def roc(id_scores, ood_scores, ax=None, label=''):
    y = np.r_[np.zeros(len(id_scores)), np.ones(len(ood_scores))]
    fpr, tpr, _ = roc_curve(y, np.r_[id_scores, ood_scores])
    ax = ax or plt.gca(); ax.plot(fpr, tpr, label=label)
    ax.set_xlabel('FPR'); ax.set_ylabel('TPR'); ax.legend()