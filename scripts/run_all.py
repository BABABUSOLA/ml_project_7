"""End-to-end: train -> dump logits -> analyse, for one dataset.

Idempotent: re-running only fills gaps, because train() and dump_logits()
both skip work that is already complete.

Usage:
    python scripts/run_all.py --art /content/drive/MyDrive/proj7 --dataset cifar10
    python scripts/run_all.py --art . --dataset cifar10  --skip-train --skip-dump
    python scripts/run_all.py --art . --dataset cifar100 --skip-train --skip-dump
    python scripts/run_all.py --art . --dataset mnist --skip-train --skip-dump

"""
import argparse, glob, os, sys, time


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--art', default='artifacts',
                    help='root holding checkpoints/ and logits/')
    ap.add_argument('--dataset', default='cifar10',
                    choices=['cifar10', 'cifar100', 'mnist'])
    ap.add_argument('--members', type=int, default=5)
    ap.add_argument('--epochs',  type=int, default=50)
    ap.add_argument('--passes',  type=int, default=50,
                    help='MC-dropout forward passes')
    ap.add_argument('--skip-train', action='store_true')
    ap.add_argument('--skip-dump',  action='store_true')
    a = ap.parse_args()

    ds = a.dataset                      # defined once, before anything uses it
    CKPT, LOG = f'{a.art}/checkpoints', f'{a.art}/logits'
    for d in (CKPT, LOG, 'results'):
        os.makedirs(d, exist_ok=True)
    print(f'[run_all] dataset={ds}  art={a.art}  members={a.members}')

    # ---- 1. train ------------------------------------------------------
    if not a.skip_train:
        from src.train import train
        t0 = time.time()
        for s in range(a.members):
            train(seed=s, epochs=a.epochs, ckpt_dir=CKPT, dataset=ds)
        train(seed=0, epochs=a.epochs, dropout_p=0.2,
              tag=f'{ds}_mcdropout', ckpt_dir=CKPT, dataset=ds)
        print(f'[train] {(time.time()-t0)/60:.1f} min total')

    # ---- 2. logits -----------------------------------------------------
    if not a.skip_dump:
        from src.inference import dump_logits
        for s in range(a.members):
            dump_logits(f'{CKPT}/{ds}_ens_s{s}.pt',
                        f'{LOG}/{ds}_ens_s{s}.npz')
        dump_logits(f'{CKPT}/{ds}_mcdropout.pt',
                    f'{LOG}/{ds}_mcdropout.npz', mc_passes=a.passes)

    # ---- 3. analyse ----------------------------------------------------
    from src.analysis import run_analysis, size_sweep
    from src.analysis_extra import run_extra
    from src.plots import make_plots

    run_analysis(logits_dir=LOG, out_dir='results', members=a.members, dataset=ds)
    size_sweep(logits_dir=LOG,  out_dir='results', members=a.members, dataset=ds)
    run_extra(logits_dir=LOG,   out_dir='results', members=a.members, dataset=ds)
    make_plots(LOG, 'results', members=a.members, dataset=ds)

    print(f'[done] wrote for {ds}:')
    for f in sorted(glob.glob(f'results/{ds}/*')):
        print('  ', f, f'{os.path.getsize(f)/1e3:.0f} KB')


if __name__ == '__main__':
    sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    main()