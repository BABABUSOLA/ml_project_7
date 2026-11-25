# --- src/data.py, generalised for multiple datasets -------------------------
import os, json
import numpy as np
import torch
from torch.utils.data import DataLoader, Subset
from torchvision import datasets, transforms

SPLIT_SEED = 1234
N_VAL = 5000
CIFAR_MEAN = (0.4914, 0.4822, 0.4465)
CIFAR_STD  = (0.2470, 0.2435, 0.2616)

# CIFAR-100's own channel statistics; the OOD set still uses the ID set's stats
C100_MEAN = (0.5071, 0.4866, 0.4409)
C100_STD  = (0.2673, 0.2564, 0.2762)

SPECS = {
    'cifar10':  {'cls': datasets.CIFAR10,  'n_classes': 10,
                 'mean': CIFAR_MEAN, 'std': CIFAR_STD},
    'cifar100': {'cls': datasets.CIFAR100, 'n_classes': 100,
                 'mean': C100_MEAN,  'std': C100_STD},
}


def split_path(dataset):
    return f'configs/split_indices_{dataset}.json'


def get_split(dataset='cifar10', n_total=50000):
    """One split per dataset, generated once and committed."""
    p = split_path(dataset)
    if os.path.exists(p):
        return json.load(open(p))
    rng = np.random.RandomState(SPLIT_SEED)
    perm = rng.permutation(n_total)
    split = {'val': perm[:N_VAL].tolist(), 'train': perm[N_VAL:].tolist()}
    os.makedirs('configs', exist_ok=True)
    json.dump(split, open(p, 'w'))
    return split


def loaders(dataset='cifar10', batch_size=128, seed=0, root='./data'):
    spec = SPECS[dataset]
    mean, std = spec['mean'], spec['std']

    train_tf = transforms.Compose([
        transforms.RandomCrop(32, padding=4),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize(mean, std),
    ])
    eval_tf = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(mean, std),
    ])

    split = get_split(dataset)
    tr_full = spec['cls'](root, train=True,  download=True, transform=train_tf)
    va_full = spec['cls'](root, train=True,  download=True, transform=eval_tf)
    test    = spec['cls'](root, train=False, download=True, transform=eval_tf)
    # SVHN is normalised with the IN-DISTRIBUTION statistics, never its own
    svhn    = datasets.SVHN(root, split='test', download=True, transform=eval_tf)

    g = torch.Generator(); g.manual_seed(seed)
    return {
        'train': DataLoader(Subset(tr_full, split['train']), batch_size=batch_size,
                            shuffle=True, num_workers=2, generator=g),
        'val':   DataLoader(Subset(va_full, split['val']), batch_size=256, shuffle=False, num_workers=2),
        'test':  DataLoader(test, batch_size=256, shuffle=False, num_workers=2),
        'svhn':  DataLoader(svhn, batch_size=256, shuffle=False, num_workers=2),
    }