import torch
import numpy as np

def gen_input(cl, batch_size, num_classes, rand_size, seed=None):
    if seed is not None:
        torch.manual_seed(seed)
    
    noise = torch.normal(0,1,(batch_size,rand_size))
    onehot = torch.zeros((batch_size,num_classes))
    onehot[np.arange(batch_size), cl] = 1
    noise[np.arange(batch_size),:num_classes]= onehot

    return noise

def weights_init(m):
    classname = m.__class__.__name__
    if classname.find('Conv') != -1:
        torch.nn.init.normal_(m.weight, 0.0, 0.02)
    elif classname.find('BatchNorm') != -1:
        torch.nn.init.normal_(m.weight, 1.0, 0.02)
        torch.nn.init.zeros_(m.bias)