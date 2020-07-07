import torch
import torchvision
from torch import nn
import torch.nn.functional as F

def log_sum_exp(inputs, keepdim=True, mask=None):
    if mask is not None:
        mask = 1. - mask
        max_offset = -1e12 * mask
    else:
        max_offset = 0.

    s, _ = torch.max(inputs + max_offset, dim=-1, keepdim=True)

    inputs_offset = inputs - s
    if mask is not None:
        inputs_offset.masked_fill_(mask.bool(), -float('inf'))
    outputs = s + inputs_offset.exp().sum(dim=-1, keepdim=True).log()

    if not keepdim:
        outputs = outputs.squeeze(-1)
    return outputs

class Shape(nn.Module):
    def __init__(self):
        super().__init__()
    
    def forward(self, x):
        print(x.shape)
        return x

class View(nn.Module):
    def __init__(self, *args):
        super().__init__()
        self.shape = args
        
    def forward(self, x):
        return x.view(*self.shape)
    
class Transform:
    def __init__(self, lst_trans):
        self.lst_trans = lst_trans

    def __call__(self, x):
        lst_x = []
        for trans in self.lst_trans:
            lst_x.append(trans(x))
        return lst_x

class Normalize(nn.Module):
    def __init__(self):
        super().__init__()
    
    def forward(self, x):
        return x / x.norm(dim=-1, keepdim=True)

class SimCLR(nn.Module):
    def __init__(self, tau=.01):
        super().__init__()
        self.tau = tau 
    
    def forward(self, lst_z):
        k = len(lst_z)
        z = torch.cat(lst_z)
        n = z.size(0)//k
        
        unfold = torch.nn.Unfold(kernel_size=n,
            dilation=1, padding=0, stride=n)
        
        l = F.cosine_similarity(z[:, :, None], 
                                z[:, None, :, None], 
                                dim=-2)[:, :, 0]

        mask = (1 - torch.eye(l.size(0))).to(l.device)
        l = log_sum_exp(l/self.tau, mask=mask) - l/self.tau
        
        l *= mask
        l = unfold(l[None, None, :])[0].reshape(n, n, k, k)
        l = l.permute(2, 3, 0, 1)
        l *= torch.eye(n).to(z.device)[None, None, :]
        l = l.sum() / (n*(k**2 - k))
        return l

