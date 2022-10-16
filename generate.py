import torch
from models import netD, netG
import matplotlib.pyplot as plt
import argparse
from utils import gen_input
from train import load_checkpoint
import numpy as np


if __name__ == '__main__':

    parser = argparse.ArgumentParser(description='Inference')
    parser.add_argument('--rand_size', type=int, default=110)
    parser.add_argument('--n_samples', type=int, default=10)
    parser.add_argument('--digit', type=int, default=3)
    parser.add_argument('--checkpoint', type=str, default='checkpoint_60.pth')
    parser.add_argument('--checkpoint_path',type=str, default='checkpoints_w/')
    parser.add_argument('--cost_sensitive', type=bool, default=True)

    args = parser.parse_args()

    num_classes = 10

    if torch.cuda.is_available():
        device = 'cuda'
    else:
        device = 'cpu'

    netG = netG(args.rand_size).to(device)
    netD = netD().to(device)
    
    netD, netG= load_checkpoint(netD, netG, checkpoint= args.checkpoint, checkpoint_path = args.checkpoint_path)
    
    f, axes = plt.subplots(1,args.n_samples, figsize= (16,12))
    for m in range(args.n_samples):
        noise = gen_input(args.digit, 1, num_classes, args.rand_size).cuda()
        fake_img = netG(noise).cpu().detach().numpy()
        axes[m].imshow(np.squeeze(fake_img), cmap= 'gray')
        axes[m].axis('off')
    plt.show()