import os, json
import numpy as np
import torch
from torch.utils.data import DataLoader, Subset
from torchvision import datasets, transforms

SPLIT_SEED = 1234
N_VAL = 5000

# Each in-distribution dataset names its own OOD partner, its own channel
# statistics, its own training-set size and whether horizontal flips are valid.
SPECS = {
    'cifar10': {
        'cls': datasets.CIFAR10,  'n_classes': 10,  'n_train': 50000,
        'mean': (0.4914, 0.4822, 0.4465), 'std': (0.2470, 0.2435, 0.2616),
        'ood_cls': datasets.SVHN, 'ood_name': 'SVHN',
        'gray': False, 'hflip': True,
    },
    'cifar100': {
        'cls': datasets.CIFAR100, 'n_classes': 100, 'n_train': 50000,
        'mean': (0.5071, 0.4866, 0.4409), 'std': (0.2673, 0.2564, 0.2762),
        'ood_cls': datasets.SVHN, 'ood_name': 'SVHN',
        'gray': False, 'hflip': True,
    },
    'mnist': {
        'cls': datasets.MNIST,    'n_classes': 10,  'n_train': 60000,
        # grayscale stats, repeated across the three channels after Grayscale(3)
        'mean': (0.1307,) * 3, 'std': (0.3081,) * 3,
        'ood_cls': datasets.FashionMNIST, 'ood_name': 'Fashion-MNIST',
        'gray': True,
        # a horizontally flipped digit is not that digit, so no flips here
        'hflip': False,
    },
}


def _transforms(spec, train):
    """28x28 grayscale is resized to 32x32 and repeated to three channels, so
    every dataset reaches the network in the same shape and the architecture
    is held constant across tracks."""
    t = []
    if spec['gray']:
        t += [transforms.Resize(32), transforms.Grayscale(num_output_channels=3)]
    if train:
        t += [transforms.RandomCrop(32, padding=4)]
        if spec['hflip']:
            t += [transforms.RandomHorizontalFlip()]
    t += [transforms.ToTensor(), transforms.Normalize(spec['mean'], spec['std'])]
    return transforms.Compose(t)


def split_path(dataset):
    return f'configs/split_indices_{dataset}.json'


def get_split(dataset='cifar10'):
    p = split_path(dataset)
    if os.path.exists(p):
        return json.load(open(p))
    n_total = SPECS[dataset]['n_train']
    rng = np.random.RandomState(SPLIT_SEED)
    perm = rng.permutation(n_total)
    split = {'val': perm[:N_VAL].tolist(), 'train': perm[N_VAL:].tolist()}
    os.makedirs('configs', exist_ok=True)
    json.dump(split, open(p, 'w'))
    return split


def _ood_set(spec, root, eval_tf):
    """SVHN takes split='test'; MNIST-family datasets take train=False."""
    if spec['ood_cls'] is datasets.SVHN:
        return datasets.SVHN(root, split='test', download=True, transform=eval_tf)
    return spec['ood_cls'](root, train=False, download=True, transform=eval_tf)


def loaders(dataset='cifar10', batch_size=128, seed=0, root='./data'):
    spec = SPECS[dataset]
    train_tf = _transforms(spec, train=True)
    eval_tf  = _transforms(spec, train=False)

    split   = get_split(dataset)
    tr_full = spec['cls'](root, train=True,  download=True, transform=train_tf)
    va_full = spec['cls'](root, train=True,  download=True, transform=eval_tf)
    test    = spec['cls'](root, train=False, download=True, transform=eval_tf)
    # the OOD set is preprocessed with the IN-DISTRIBUTION statistics, never its own
    ood     = _ood_set(spec, root, eval_tf)

    g = torch.Generator(); g.manual_seed(seed)
    return {
        'train': DataLoader(Subset(tr_full, split['train']), batch_size=batch_size,
                            shuffle=True, num_workers=2, generator=g),
        'val':   DataLoader(Subset(va_full, split['val']), batch_size=256,
                            shuffle=False, num_workers=2),
        'test':  DataLoader(test, batch_size=256, shuffle=False, num_workers=2),
        'ood':   DataLoader(ood,  batch_size=256, shuffle=False, num_workers=2),
    }