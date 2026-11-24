import os, time, random
import numpy as np, torch, torch.nn as nn
from src.data import loaders
from src.models import make_resnet18

def set_seed(seed):
    random.seed(seed); np.random.seed(seed)
    torch.manual_seed(seed); torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.benchmark = False

def train(seed=0, epochs=50, dropout_p=0.0, ckpt_dir='checkpoints',
          tag=None, force=False):
    set_seed(seed)
    tag = tag or (f'dropout_s{seed}' if dropout_p > 0 else f'ens_s{seed}')
    os.makedirs(ckpt_dir, exist_ok=True)
    ckpt = f'{ckpt_dir}/{tag}.pt'

    # already finished? return before building model or loaders
    if os.path.exists(ckpt) and not force:
        st = torch.load(ckpt, map_location='cpu')
        if st['epoch'] + 1 >= epochs:
            print(f'{tag}: complete ({epochs} epochs, {st["elapsed"]/60:.1f} min) - skipping')
            return ckpt, st['elapsed']

    L = loaders(seed=seed)
    model = make_resnet18(dropout_p=dropout_p).cuda()
    opt = torch.optim.SGD(model.parameters(), lr=0.1, momentum=0.9,
                          weight_decay=5e-4, nesterov=True)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=epochs)
    crit = nn.CrossEntropyLoss()

    start, elapsed = 0, 0.0
    if os.path.exists(ckpt) and not force:        # partial run - resume
        st = torch.load(ckpt)
        model.load_state_dict(st['model']); opt.load_state_dict(st['opt'])
        sched.load_state_dict(st['sched'])
        start, elapsed = st['epoch'] + 1, st.get('elapsed', 0.0)
        print(f'resuming {tag} from epoch {start}')

    for ep in range(start, epochs):
        t0 = time.time(); model.train()
        for x, y in L['train']:
            x, y = x.cuda(), y.cuda()
            opt.zero_grad(); crit(model(x), y).backward(); opt.step()
        sched.step(); elapsed += time.time() - t0
        torch.save({'model': model.state_dict(), 'opt': opt.state_dict(),
                    'sched': sched.state_dict(), 'epoch': ep,
                    'elapsed': elapsed, 'seed': seed, 'dropout_p': dropout_p}, ckpt)
        print(f'{tag} epoch {ep+1}/{epochs}  {elapsed/60:.1f} min')
    return ckpt, elapsed