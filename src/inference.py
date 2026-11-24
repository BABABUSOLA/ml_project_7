import numpy as np, torch
from src.models import make_resnet18

def enable_mc_dropout(model):
    """Dropout on, BatchNorm stays in eval. model.train() would break BatchNorm.
    Dropout2d must be listed explicitly — it is not a subclass of Dropout."""
    model.eval()
    for m in model.modules():
        if isinstance(m, (torch.nn.Dropout, torch.nn.Dropout2d)):
            m.train()

@torch.no_grad()
def dump_logits(ckpt_path, out_path, mc_passes=None, seed=0):
    from src.data import loaders
    st = torch.load(ckpt_path)
    model = make_resnet18(dropout_p=st.get('dropout_p', 0.0)).cuda()
    model.load_state_dict(st['model']); model.eval()
    if mc_passes:
        enable_mc_dropout(model); torch.manual_seed(seed)

    L = loaders()
    out = {}
    for name in ['val', 'test', 'svhn']:
        Ls, Ys = [], []
        for x, y in L[name]:
            x = x.cuda()
            if mc_passes:
                Ls.append(np.stack([model(x).cpu().numpy() for _ in range(mc_passes)]))
            else:
                Ls.append(model(x).cpu().numpy())
            Ys.append(y.numpy())
        out[f'{name}_logits'] = np.concatenate(Ls, axis=1 if mc_passes else 0)
        out[f'{name}_labels'] = np.concatenate(Ys)
    np.savez_compressed(out_path, **out)
    return out_path