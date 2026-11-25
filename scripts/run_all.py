"""End-to-end: train -> dump logits -> analyse. Idempotent; re-running only
fills gaps. Usage:  python scripts/run_all.py --art /content/drive/MyDrive/proj7"""
import argparse, os, sys, time

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--art', default='artifacts', help='root for checkpoints/ and logits/')
    ap.add_argument('--members', type=int, default=5)
    ap.add_argument('--epochs',  type=int, default=50)
    ap.add_argument('--passes',  type=int, default=50, help='MC-dropout forward passes')
    ap.add_argument('--skip-train', action='store_true')
    ap.add_argument('--skip-dump',  action='store_true')
    a = ap.parse_args()

    CKPT, LOG = f'{a.art}/checkpoints', f'{a.art}/logits'
    for d in (CKPT, LOG, 'results'):
        os.makedirs(d, exist_ok=True)

    # ---- 1. train ------------------------------------------------------
    if not a.skip_train:
        from src.train import train
        t0 = time.time()
        for s in range(a.members):
            train(seed=s, epochs=a.epochs, ckpt_dir=CKPT)
        train(seed=0, epochs=a.epochs, dropout_p=0.2, tag='mcdropout', ckpt_dir=CKPT)
        print(f'[train] {(time.time()-t0)/60:.1f} min total')

    # ---- 2. logits -----------------------------------------------------
    if not a.skip_dump:
        from src.inference import dump_logits
        for s in range(a.members):
            out = f'{LOG}/ens_s{s}.npz'
            if os.path.exists(out):
                print(f'[dump] {out} exists, skipping'); continue
            dump_logits(f'{CKPT}/ens_s{s}.pt', out)
        out = f'{LOG}/mcdropout.npz'
        if os.path.exists(out):
            print(f'[dump] {out} exists, skipping')
        else:
            dump_logits(f'{CKPT}/mcdropout.pt', out, mc_passes=a.passes)

    # ---- 3. analyse ----------------------------------------------------
    from src.analysis import run_analysis
    run_analysis(logits_dir=LOG, out_dir='results', members=a.members)
    print('[done] results/ written')

if __name__ == '__main__':
    sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    main()