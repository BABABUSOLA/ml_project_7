#Sanity: T between 1.2 and 2.5. T < 1 means something is wrong upstream.

import torch, torch.nn.functional as F

def fit_temperature(val_logits, val_labels):
    logits = torch.tensor(val_logits, dtype=torch.float32)
    labels = torch.tensor(val_labels, dtype=torch.long)
    log_T = torch.zeros(1, requires_grad=True)        # optimise log T so T stays > 0
    opt = torch.optim.LBFGS([log_T], lr=0.01, max_iter=200)
    def closure():
        opt.zero_grad()
        loss = F.cross_entropy(logits / log_T.exp(), labels)
        loss.backward(); return loss
    opt.step(closure)
    return float(log_T.detach().exp())

def apply_temperature(logits, T):
    return logits / T