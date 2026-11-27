# Uncertainty-Aware Image Classification with Deep Ensembles

**CPTS 570 · Project 7** · Hung-Liang (Leon) Tsai · Oluwabusola Adeleke

Neural networks are confidently wrong — they assign 99% certainty to misclassified images and to inputs they have never seen. We compare deep ensembles, MC dropout and temperature scaling across three datasets to find which gives confidence you can trust, and whether that signal detects out-of-distribution inputs.

---

## Quick start

Reproduce every table and figure from cached model outputs. **No GPU, no training.**

```bash
git clone https://github.com/BABABUSOLA/ml_project_7.git
cd ml_project_7
python -m venv .venv && source .venv/bin/activate     # Windows: .venv\Scripts\Activate.ps1
pip install -r requirements.txt

python scripts/run_all.py --art . --dataset cifar10  --skip-train --skip-dump
python scripts/run_all.py --art . --dataset cifar100 --skip-train --skip-dump
python scripts/run_all.py --art . --dataset mnist    --skip-train --skip-dump
```

Each writes `results/<dataset>/` — the main table, size sweep, misclassification AUROC, significance tests, and five figures. Under a minute per track.

`--skip-train --skip-dump` is what makes this fast: it reuses the committed logits instead of retraining. Drop those flags to rebuild from scratch, which needs a GPU — see [Retraining](#retraining-from-scratch).

> **Note on repository size.** The saved logits are committed so that the analysis is reproducible without a GPU, which makes the clone large (~80 MB). The trained checkpoints are *not* committed; see [Retraining](#retraining-from-scratch).

Or open `notebooks/demo.ipynb`, which does the same thing interactively.

---

## What this project measures

Accuracy alone does not tell you whether a classifier can be trusted. Two further properties matter:

- **Calibration** — does 90% confidence mean right 90% of the time? Measured with ECE, Brier score and NLL, before and after temperature scaling.
- **Out-of-distribution detection** — can the model flag inputs unlike its training data? Measured with AUROC and FPR at 95% TPR against a disjoint dataset.

### The uncertainty decomposition

Every method here produces *M* predictive distributions per image, differing only in where they come from. With `p̄ = (1/M) Σ pₘ`, the entropy of that average decomposes:

```
     H[p̄]        =    H[p̄] − (1/M)Σ H[pₘ]    +    (1/M)Σ H[pₘ]
   ─────────           ───────────────────         ─────────────
     total               epistemic                   aleatoric
                     (= mutual information)
```

**Total** and **epistemic** uncertainty are reported as separate OOD scores, not merged into one "uncertainty" number. They diverge: members agreeing on a near-uniform output give high total but near-zero disagreement; members confidently predicting different classes give the reverse. Which one detects OOD better turned out to depend on the dataset — see [Key findings](#key-findings).

---

## Key findings

**1 · Ensembling buys calibration, not accuracy.** On CIFAR-10, going from one network to five improves accuracy by 1.2% relative while reducing ECE by **73%**. The benefit is almost entirely in confidence quality.

**2 · A single scalar recovers most of it.** Temperature scaling takes one network from ECE 0.0318 to 0.0128 in under a second of fitting — within 0.0008 of an unscaled *three*-member ensemble costing three times the training. The five-member ensemble is still best at 0.0064, so it is not dominated, but calibration per GPU-hour strongly favours post-hoc scaling.

**3 · Which uncertainty score is best is not predictable.** Taking AUROC for mutual information minus AUROC for entropy — positive means disagreement is the better detector:

| Dataset | Accuracy | Deep ensemble | MC dropout |
|---|---|---|---|
| MNIST | 0.9966 | **+0.0095** | −0.0020 |
| CIFAR-10 | 0.9386 | −0.0188 | −0.0406 |
| CIFAR-100 | 0.7531 | −0.0144 | **+0.0032** |

Mutual information is worst on CIFAR-10 for both methods, best for the ensemble on MNIST, and best for MC dropout on CIFAR-100 — and the two methods disagree about its direction on two of the three datasets. CIFAR-10 is the extreme in both columns, so no ordering by difficulty produces a monotone pattern.

After CIFAR-100 we proposed that the aleatoric term mutual information discards is useful where class ambiguity is low and a contaminant where it is high. We recorded a prediction from it — that MI would fall *further* behind entropy on MNIST — **before training the MNIST models. It was refuted.** A fallback hypothesis, that Fashion-MNIST is a harder near-OOD problem, is also contradicted: MNIST is by far the *easiest* detection task of the three (FPR@95 of 0.011 against CIFAR-10's 0.378).

The defensible conclusion is narrower: total and epistemic uncertainty are distinct quantities whose relative usefulness varies by dataset and by method, so they must be reported separately. A single conflated "uncertainty" number would have given a different answer on each track with no way to notice.

**4 · Calibration does not confer OOD detection.** The best-calibrated configuration reaches ECE 0.0064, yet still admits 37.8% of SVHN images at a 95% true-positive operating point. Error triage is where these estimates are reliable: rejecting the 10% most uncertain inputs removes **72% of all errors**, taking CIFAR-10 accuracy from 0.950 to 0.986.

Full discussion, with significance tests and figures, in [`report.pdf`](report.pdf).

---

## Results

### CIFAR-10

| Method | Scaling | *T* | Accuracy | NLL | ECE | Brier | AUROC (ent.) | FPR@95 | Train (min) |
|---|---|---|---|---|---|---|---|---|---|
| Single network | — | 1.000 | 0.9386 | 0.2278 | 0.0318 | 0.0986 | 0.9072 | 0.5728 | 38.2 |
| Single network | *T* | 1.433 | 0.9386 | 0.1968 | 0.0128 | 0.0941 | 0.9055 | 0.5456 | 38.2 |
| MC dropout (*S*=50) | — | 1.000 | 0.9303 | 0.2057 | 0.0130 | 0.1011 | 0.9293 | 0.4663 | 38.4 |
| MC dropout (*S*=50) | *T* | 1.103 | 0.9303 | 0.2040 | 0.0097 | 0.1007 | 0.9305 | 0.4549 | 38.4 |
| Deep ensemble (*M*=5) | — | 1.000 | 0.9499 | 0.1497 | 0.0086 | 0.0727 | 0.9376 | 0.3926 | 191.0 |
| **Deep ensemble (*M*=5)** | ***T*** | 1.116 | **0.9498** | **0.1472** | **0.0064** | **0.0724** | **0.9395** | **0.3780** | 191.0 |

Across the five members individually, accuracy is 0.9400 ± 0.0018 and ECE 0.0327 ± 0.0011, so the ensemble sits 5.4 and 21.1 standard deviations from its own members' means.

### CIFAR-100

| Method | Scaling | *T* | Accuracy | NLL | ECE | Brier | AUROC (ent.) | AUROC (MI) |
|---|---|---|---|---|---|---|---|---|
| Single network | — | 1.000 | 0.7531 | 0.9917 | 0.0900 | 0.3549 | 0.8284 | — |
| Single network | *T* | 1.304 | 0.7531 | 0.9242 | 0.0329 | 0.3429 | 0.8404 | — |
| MC dropout (*S*=50) | — | 1.000 | 0.7324 | 0.9505 | 0.0160 | 0.3641 | 0.8157 | 0.8196 |
| MC dropout (*S*=50) | *T* | 1.030 | 0.7323 | 0.9492 | **0.0114** | 0.3639 | 0.8171 | **0.8204** |
| Deep ensemble (*M*=5) | — | 1.000 | 0.7844 | 0.7885 | 0.0178 | 0.3014 | 0.8435 | 0.8298 |
| **Deep ensemble (*M*=5)** | ***T*** | 1.051 | **0.7841** | **0.7846** | 0.0160 | **0.3013** | **0.8452** | 0.8308 |

Every absolute figure is worse than CIFAR-10's, as expected on a 100-class problem with 500 images per class. What matters is which orderings survive — see [Key findings](#key-findings) point 3.

### MNIST

| Method | Scaling | *T* | Accuracy | ECE | AUROC (MSP) | AUROC (ent.) | AUROC (MI) | FPR@95 (best) |
|---|---|---|---|---|---|---|---|---|
| Single network | — | 1.000 | 0.9966 | 0.0009 | 0.9039 | 0.8986 | — | 0.1759 |
| MC dropout (*S*=50) | *T* | 0.908 | 0.9966 | 0.0009 | 0.9910 | **0.9922** | 0.9902 | **0.0109** |
| Deep ensemble (*M*=5) | *T* | 0.996 | **0.9969** | 0.0010 | 0.9827 | 0.9825 | **0.9920** | 0.0150 |

Two things stand out. **Temperature scaling has nothing to correct** — ECE is already 0.0009 and the fitted temperature is 1.000, so the cost argument in finding 2 is vacuous on an easy task. And **MC dropout is *under*-confident**, with *T* = 0.908, the only value below 1 anywhere in the study: averaging 50 stochastic passes over a network that is already right 99.7% of the time flattens a distribution that needed no flattening.

### Significance

Paired *t*-tests on per-example errors, *K* = 10,000:

| Comparison | Error A | Error B | *T* | *p* |
|---|---|---|---|---|
| Ensemble vs single network | 0.0502 | 0.0614 | −6.37 | < 10⁻⁴ |
| Ensemble vs MC dropout | 0.0502 | 0.0697 | −9.18 | < 10⁻⁴ |
| MC dropout vs single network | 0.0697 | 0.0614 | +3.48 | 0.0005 |

MC dropout is significantly *worse* than the baseline at full coverage — but its accuracy–coverage curve crosses above the single network at roughly 74% coverage, because it ranks its own errors better (misclassification AUROC 0.9328 vs 0.9214).

### Detection difficulty across tracks

| Dataset | Best AUROC | Best FPR@95 |
|---|---|---|
| MNIST | 0.9922 | 0.0109 |
| CIFAR-10 | 0.9395 | 0.3780 |
| CIFAR-100 | 0.8452 | 0.7480 |

A sixty-fold increase in false-alarm rate across the three tracks — far larger than any difference between the methods on a single one.

### Figures

In `results/<dataset>/`: reliability diagrams, confidence histograms, OOD ROC curves, accuracy–coverage curves, and ensemble performance against *M*.

---

## Methods

| Method | Configuration | Reference |
|---|---|---|
| Deep ensembles | *M* = 5 members, independent seeds governing initialisation **and** data order; softmax outputs averaged | Lakshminarayanan et al., [1612.01474](https://arxiv.org/abs/1612.01474) |
| MC dropout | Dropout retained at inference after each residual stage, *S* = 50 stochastic passes | Gal & Ghahramani, [1506.02142](https://arxiv.org/abs/1506.02142) |
| Temperature scaling | Single scalar fitted by L-BFGS on held-out validation | Guo et al., [1706.04599](https://arxiv.org/abs/1706.04599) |
| MSP as OOD score | Negated maximum softmax probability | Hendrycks & Gimpel, [1610.02136](https://arxiv.org/abs/1610.02136) |
| Mutual information | `H[p̄] − (1/M)Σ H[pₘ]` | Gal et al., [1703.02910](https://arxiv.org/abs/1703.02910) |

**Backbone.** ResNet-18 adapted for 32×32 input — the 7×7 stride-2 stem replaced with 3×3 stride-1 and the initial max-pool removed, since the ImageNet configuration destroys too much resolution before the first residual block. MNIST is resized from 28×28 and repeated to three channels, so a single architecture serves all three datasets and differences are attributable to the uncertainty method rather than to capacity.

**Optimisation.** SGD with Nesterov momentum 0.9, learning rate 0.1, weight decay 5×10⁻⁴, batch size 128, cosine annealing over a fixed 50-epoch budget.

---

## Datasets

| Track | In-distribution | Split | OOD set |
|---|---|---|---|
| CIFAR-10 | 10 classes, 32×32 RGB | 45,000 / 5,000 / 10,000 | SVHN test (26,032) |
| CIFAR-100 | 100 classes, 32×32 RGB | 45,000 / 5,000 / 10,000 | SVHN test (26,032) |
| MNIST | 10 classes, 28×28 grayscale, resized to 32×32 and repeated to 3 channels | 55,000 / 5,000 / 10,000 | Fashion-MNIST test (10,000) |

Each dataset ships train and test as disjoint files; only the training file is partitioned, and the partition is generated once from a fixed seed and committed under `configs/`.

**The OOD set is normalised with the in-distribution statistics, never its own.** At deployment an input's OOD status is precisely what is being inferred, so using its own statistics would leak that information into preprocessing and inflate detection performance.

---

## Protocol

Decisions fixed before any training, to keep the comparison honest:

- **The split is committed.** Generated once from seed 1234, stored in `configs/split_indices_<dataset>.json`, never regenerated. `get_split()` warns loudly if it ever has to create one, because a regenerated split would mean models evaluated on data they were trained on.
- **Validation has exactly one job.** Models train to a fixed epoch budget with no early stopping, so the validation split is used only for fitting the temperature and selecting the rejection threshold — never for model selection. Using it for both would bias *T* towards 1.
- **Test is untouched** until the protocol was frozen.
- **Ensemble scores are computed on the averaged predictive distribution**, not by averaging per-model scores, which would discard the disagreement signal.
- **The OOD threshold** is the 95th percentile of in-distribution *validation* scores, so no OOD sample informs the operating point at which it is later judged.
- **Costs are reported alongside performance.** An *M*-member ensemble costs *M*× the training and inference of a single network; comparison at constant architecture rather than at parameter parity makes that trade-off explicit.

---

## Repository layout

```
src/
  data.py            dataset specs, transforms, deterministic splits
  models.py          ResNet-18 with CIFAR adaptation and optional dropout
  train.py           resumable training, per-epoch checkpointing
  inference.py       dump_logits — saves logits for val / test / OOD
  metrics.py         ECE, Brier, NLL, AUROC, FPR@95
  calibration.py     temperature scaling
  ood.py             MSP, predictive entropy, mutual information
  analysis.py        main table, size sweep, per-member spread
  analysis_extra.py  misclassification, rejection curves, paired t-tests
  plots.py           all five figures
scripts/run_all.py   one-command pipeline per dataset
tests/               unit tests for the metrics
configs/             committed splits
logits/              cached model outputs — the reproducibility artefact
notebooks/demo.ipynb interactive walkthrough
results/<dataset>/   generated tables and figures
```

---

## Running the tests

```bash
pytest tests/ -v
```

Five tests: ECE on perfectly-calibrated and maximally-overconfident cases, Brier on perfect and uniform predictions, and NLL against `torch.nn.functional.cross_entropy`.

Each metric was additionally hand-computed on a five-example case and checked by the other author. The unit tests confirm the code does what was written; the hand-check confirms what was written is the right formula.

---

## Retraining from scratch

Requires a GPU. Roughly 3.8 GPU-hours per dataset.

```python
# In Colab, after mounting Drive and cloning
!python scripts/run_all.py --art /content/drive/MyDrive/proj7 --dataset cifar10
```

Trains five ensemble members plus one MC-dropout model, dumps logits, runs the full analysis. Both training and logit dumping skip work already complete, so an interrupted run resumes rather than restarting.

**Expect small numerical differences.** cuDNN selects kernels non-deterministically, so retrained weights differ in the last decimals even with identical seeds. Accuracy might land at 0.9497 rather than 0.9499. The committed logits exist so the *analysis* reproduces exactly even where training cannot.

Checkpoints are written to `<art>/checkpoints/` and are not committed — ~86 MB each.

---

## Limitations

- Ensemble size *M* = 5; scaling beyond this is not characterised, with the sweep covering *M* ∈ {1, 3, 5}.
- One OOD partner per track, and the partners are not comparable: CIFAR vs SVHN is far-OOD, MNIST vs Fashion-MNIST shares resolution and colour depth. The tracks therefore differ in OOD character as well as task difficulty, which is a confound in any comparison across them.
- Near-OOD behaviour on the CIFAR tracks, where the shift is semantic rather than categorical, is not characterised.
- A single architecture, so conclusions may not transfer to other backbones or to transformer models.
- Comparison at constant architecture rather than matched parameter or FLOP budgets, so the ensemble retains a capacity advantage; cost columns are reported to make this explicit.
- The variation in score ranking rests on three datasets and two methods. Six measurements suffice to refute a proposed mechanism but not to establish an alternative.
- The test set is evaluated repeatedly across experiments. No model or threshold was selected on it, but repeated inspection is a weak form of information leakage.

---

## References

1. Gal & Ghahramani (2016). *Dropout as a Bayesian Approximation.* ICML. [arXiv:1506.02142](https://arxiv.org/abs/1506.02142)
2. Lakshminarayanan, Pritzel & Blundell (2017). *Simple and Scalable Predictive Uncertainty Estimation using Deep Ensembles.* NeurIPS. [arXiv:1612.01474](https://arxiv.org/abs/1612.01474)
3. Guo, Pleiss, Sun & Weinberger (2017). *On Calibration of Modern Neural Networks.* ICML. [arXiv:1706.04599](https://arxiv.org/abs/1706.04599)
4. Hendrycks & Gimpel (2017). *A Baseline for Detecting Misclassified and Out-of-Distribution Examples.* ICLR. [arXiv:1610.02136](https://arxiv.org/abs/1610.02136)
5. Gal, Islam & Ghahramani (2017). *Deep Bayesian Active Learning with Image Data.* ICML. [arXiv:1703.02910](https://arxiv.org/abs/1703.02910)

