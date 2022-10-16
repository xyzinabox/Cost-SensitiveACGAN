import torch
import torchvision
from torchvision import transforms
from models import netD, netG
import numpy as np
import os
from utils import gen_input, weights_init
import  matplotlib.pyplot as plt
from pytorch_msssim import ssim, ms_ssim
import argparse


def load_MNIST():
    transform = transforms.Compose([
                transforms.ToTensor(),
                transforms.Normalize((0.5),(0.5))
            ])
    data = torchvision.datasets.MNIST('Data/',download=True, transform= transform)

    return data

def make_imbalanced(data, imba_rate):

    data_imba = []
    for i in data:
        if i[1]==0:
            if np.random.random()>imba_rate:
                data_imba.append(i)
        else:
            data_imba.append(i)

    return data_imba

def train(netD, netG, data_loader):
    
    loss_G = 0
    loss_D = 0

    for j,data in enumerate(data_loader):

        img, labels = data

        img.to(device)
        labels.to(device)

        rand_cl = torch.randint(0,num_classes,(len(img),)).to(device)
        noise = gen_input(rand_cl, len(img), num_classes, args.rand_size).to(device)
        
        netD.cuda()

        real_fake, cl = netD(img)
        real_labels = torch.ones_like(real_fake).to(device)

        dis_loss = dis_criterion(real_fake,real_labels)
       
        aux_loss = aux_criterion(cl, labels)
        
        err_1D = dis_loss+2*aux_loss
        err_1D.backward()
        
        fake_img = netG(noise)
        real_fake, cl = netD(fake_img.detach())

        fake_labels = torch.zeros_like(real_fake).to(device)
        dis_loss = dis_criterion(real_fake, fake_labels)
        aux_loss = aux_criterion(cl, rand_cl)
        
        err_2D = dis_loss


        err_D = err_1D+err_2D
        err_2D.backward()
        optimizerD.step()

        netG.zero_grad()
        
        real_fake, cl = netD(fake_img)
        real_labels = torch.ones_like(real_fake).to(device)
        dis_loss = dis_criterion(real_fake,real_labels)
        aux_loss = aux_criterion(cl, rand_cl)
        # aux_loss = aux_criterion(cl, labels)
        err_G = dis_loss + aux_loss
        err_G.backward()

        optimizerG.step()

        loss_D +=err_D
        loss_G +=err_G

    return loss_D, loss_G
    
         
def update_cs_weights(netD, netG, data_loader):
    aux_criterion = torch.nn.CrossEntropyLoss(reduction='none')
    for data in data_loader:

        img, labels = data

        img.to(device)
        labels.to(device)

        rand_cl = torch.randint(0,num_classes,(len(img),)).to(device)
        noise = gen_input(rand_cl, (len(img),), num_classes, args.rand_size).to(device)

        real_fake, cl = netD(img)
        err_1D = aux_criterion(cl, labels).detach().cpu()

        for k in range(labels.size()[0]):
            
            after_error[labels[k]]+=err_1D[k]
            num_per_cl[labels[k]]+=1
           
    weights = torch.nn.functional.softmax((torch.tensor(np.array(after_error)/np.array(num_per_cl)).to(device).float()))

    return weights

def load_checkpoint(netD, netG, checkpoint='checkpoint_60.pth', last_checkpoint = False, checkpoint_path= 'checkpoints_w/', for_train= False):
    
    if last_checkpoint:
      checkpoint = os.listdir(checkpoint_path)[-1]
    
    state = torch.load(checkpoint_path+checkpoint)
    start_epoch = state['epoch']
    netD.load_state_dict(state['D_state_dict'])
    netG.load_state_dict(state['G_state_dict'])

    if not for_train:
        return netD, netG
        
    optimizerD.load_state_dict(state['optimizerD_disc'])
    optimizerG.load_state_dict(state['optimizerG_disc'])

    try:
      weights = state['weights']
    except:
      return netD, netG, optimizerD, optimizerG
    return netD, netG, optimizerD, optimizerG, weights

if __name__ == '__main__':

    parser = argparse.ArgumentParser(description='Training')
    parser.add_argument('--rand_size', type=int, default=110)
    parser.add_argument('--batch_size', type=int, default=100)
    parser.add_argument('--imba_rate', type=int, default=0.95)
    parser.add_argument('--epochs', type=int, default=160)
    parser.add_argument('--imba',type=bool, default=True)
    parser.add_argument('--cost_sensitive', type=bool, default=True)

    args = parser.parse_args()

    num_classes = 10

    if torch.cuda.is_available():
        device = 'cuda'
    else:
        device = 'cpu'

    data = load_MNIST()

    if args.imba:
        data = make_imbalanced(data, args.imba_rate)
    
    data_loader = torch.utils.data.DataLoader(data,
                                            batch_size=args.batch_size,
                                            shuffle=True,
                                            )

    netG = netG(args.rand_size).to(device)
    netD = netD().to(device)

    optimizerD = torch.optim.Adam(netD.parameters(), lr=0.0002)
    optimizerG = torch.optim.Adam(netG.parameters(), lr=0.0002)

    netD.apply(weights_init)
    netG.apply(weights_init)

    dis_criterion = torch.nn.BCELoss()
    aux_criterion = torch.nn.CrossEntropyLoss()
    weights = torch.Tensor([1/num_classes for _ in range(num_classes)]).to(device)


    for epoch in range(args.epochs+1):

        n = len(data_loader)
        
        if args.cost_sensitive:
            aux_criterion = torch.nn.CrossEntropyLoss(weight= weights)
            after_error = [0 for _ in range(num_classes)]
            num_per_cl = [0 for _ in range(num_classes)]
        else:
            aux_criterion = torch.nn.CrossEntropyLoss()
        
        
        
        loss_D, loss_G = train(data_loader)

        if not epoch%10:
            state = {'epoch': epoch, 'D_state_dict': netD.state_dict(), 'G_state_dict': netG.state_dict(),
                    'optimizerD_disc': optimizerD.state_dict(), 'optimizerG_disc': optimizerG.state_dict(), 'weights': weights}
            torch.save(state,f"checkpoints_w/checkpoint_{state['epoch']}.pth")
            print(f'Epoch {epoch} --- Discriminator loss: {loss_D/n} --- Generator loss: {loss_G/n}')

        if args.cost_sensitive:
            weights = update_cs_weights(data_loader)
    

        
        