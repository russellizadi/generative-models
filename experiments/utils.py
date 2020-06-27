import torch
import torchvision
from torch import nn
import torch.nn.functional as F

class Shape(nn.Module):
    def __init__(self):
        super(Shape, self).__init__()
    
    def forward(self, x):
        print(x.shape)
        return x

class View(nn.Module):
    def __init__(self, *args):
        super(View, self).__init__()
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