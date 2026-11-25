import os, time, random
import numpy as np, torch, torch.nn as nn
from src.data import loaders
from src.models import make_resnet18

def set_seed(seed):
    random.seed(seed); np.random.seed(seed)
    torch.manual_seed(seed); torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.benchmark = False

def train(seed=0, epochs=50, dropout_p=0.0, ckpt_dir='checkpoints',
          tag=None, force=False, dataset='cifar10'):
    set_seed(seed)
    tag = tag or (f'{dataset}_dropout_s{seed}' if dropout_p > 0
                  else f'{dataset}_ens_s{seed}')
    os.makedirs(ckpt_dir, exist_ok=True)
    ckpt = f'{ckpt_dir}/{tag}.pt'

    # already finished? return before building model or loaders
    if os.path.exists(ckpt) and not force:
        st = torch.load(ckpt, map_location='cpu')
        if st['epoch'] + 1 >= epochs:
            print(f'{tag}: complete ({epochs} epochs, {st["elapsed"]/60:.1f} min) - skipping')
            return ckpt, st['elapsed']

    L = loaders(seed=seed, dataset=dataset)
    model = make_resnet18(num_classes=SPECS[dataset]['n_classes'],
                          dropout_p=dropout_p).cuda()
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
        
        sched.step()
        ep_time = time.time() - t0
        elapsed += ep_time
        eta = (epochs - ep - 1) * ep_time / 60
        print(f'{tag} epoch {ep+1}/{epochs}  {ep_time:.0f}s  total {elapsed/60:.1f} min  eta {eta:.0f} min')
        torch.save({'model': model.state_dict(), 'opt': opt.state_dict(),
                    'sched': sched.state_dict(), 'epoch': ep,
                    'elapsed': elapsed, 'seed': seed, 'dropout_p': dropout_p,
                    'dataset': dataset}, ckpt) # record it for inference
        print(f'{tag} epoch {ep+1}/{epochs}  {elapsed/60:.1f} min')
    return ckpt, elapsed