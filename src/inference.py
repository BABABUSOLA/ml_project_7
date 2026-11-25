import os
import numpy as np
import torch

from src.models import make_resnet18
from src.data import loaders, SPECS


def enable_mc_dropout(model):
    """Dropout on, BatchNorm stays in eval. model.train() would break BatchNorm.
    Dropout2d must be listed explicitly - it is not a subclass of Dropout."""
    model.eval()
    for m in model.modules():
        if isinstance(m, (torch.nn.Dropout, torch.nn.Dropout2d)):
            m.train()


@torch.no_grad()
def dump_logits(ckpt_path, out_path, mc_passes=None, seed=0, force=False):
    """Run one trained model over val/test/svhn and save the raw logits.

    mc_passes=None -> single deterministic pass, arrays are (N, C)
    mc_passes=T    -> dropout left on, T stochastic passes, arrays are (T, N, C)

    Skips entirely if out_path already exists, unless force=True.
    """
    if os.path.exists(out_path) and not force:
        d = np.load(out_path)
        print(f'{os.path.basename(out_path)}: exists '
              f'{d["test_logits"].shape} - skipping')
        return out_path

    st = torch.load(ckpt_path)
    ds = st.get('dataset', 'cifar10')
    model = make_resnet18(num_classes=SPECS[ds]['n_classes'],
                          dropout_p=st.get('dropout_p', 0.0)).cuda()
    model.load_state_dict(st['model'])
    model.eval()

    if mc_passes:
        enable_mc_dropout(model)
        torch.manual_seed(seed)          # reproducible dropout masks

    L = loaders(dataset=ds)
    out = {}
    for name in ['val', 'test', 'svhn']:
        Ls, Ys = [], []
        for x, y in L[name]:
            x = x.cuda()
            if mc_passes:
                Ls.append(np.stack([model(x).cpu().numpy()
                                    for _ in range(mc_passes)]))
            else:
                Ls.append(model(x).cpu().numpy())
            Ys.append(y.numpy())
        out[f'{name}_logits'] = np.concatenate(Ls, axis=1 if mc_passes else 0)
        out[f'{name}_labels'] = np.concatenate(Ys)

    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    np.savez_compressed(out_path, **out)
    print(f'{os.path.basename(out_path)}: written {out["test_logits"].shape}')
    return out_path