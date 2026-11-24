import os, json
import numpy as np
import torch
from torch.utils.data import DataLoader, Subset
from torchvision import datasets, transforms

SPLIT_SEED = 1234
N_VAL = 5000
CIFAR_MEAN = (0.4914, 0.4822, 0.4465)
CIFAR_STD  = (0.2470, 0.2435, 0.2616)
SPLIT_PATH = 'configs/split_indices.json'

_train_tf = transforms.Compose([
    transforms.RandomCrop(32, padding=4),
    transforms.RandomHorizontalFlip(),
    transforms.ToTensor(),
    transforms.Normalize(CIFAR_MEAN, CIFAR_STD),
])
# NOTE: SVHN uses this same transform — CIFAR-10 statistics, not its own.
_eval_tf = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize(CIFAR_MEAN, CIFAR_STD),
])

def get_split(n_total=50000):
    """Generated once, committed, never regenerated."""
    if os.path.exists(SPLIT_PATH):
        return json.load(open(SPLIT_PATH))
    rng = np.random.RandomState(SPLIT_SEED)
    perm = rng.permutation(n_total)
    split = {'val': perm[:N_VAL].tolist(), 'train': perm[N_VAL:].tolist()}
    os.makedirs('configs', exist_ok=True)
    json.dump(split, open(SPLIT_PATH, 'w'))
    return split

def loaders(batch_size=128, seed=0, root='./data'):
    split = get_split()
    tr_full = datasets.CIFAR10(root, train=True,  download=True, transform=_train_tf)
    va_full = datasets.CIFAR10(root, train=True,  download=True, transform=_eval_tf)
    test    = datasets.CIFAR10(root, train=False, download=True, transform=_eval_tf)
    svhn    = datasets.SVHN(root, split='test',   download=True, transform=_eval_tf)

    g = torch.Generator(); g.manual_seed(seed)      # seed controls data order
    return {
        'train': DataLoader(Subset(tr_full, split['train']), batch_size=batch_size,
                            shuffle=True, num_workers=2, generator=g, drop_last=False),
        'val':   DataLoader(Subset(va_full, split['val']), batch_size=256, shuffle=False, num_workers=2),
        'test':  DataLoader(test, batch_size=256, shuffle=False, num_workers=2),
        'svhn':  DataLoader(svhn, batch_size=256, shuffle=False, num_workers=2),
    }