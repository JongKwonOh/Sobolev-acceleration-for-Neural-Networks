import torch
from torch.autograd import Variable
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

def get_optimizer(optim_type, model, optimizer_args):
    # adam or lbfgs
    if optim_type=='adam':
        return torch.optim.Adam([{'params': model.parameters()}], **optimizer_args)

    elif optim_type=='lbfgs':
        return torch.optim.LBFGS([{'params': model.parameters()}], **optimizer_args)

    else :
        raise ValueError(f"Unknown optimizer type: {optim_type}")