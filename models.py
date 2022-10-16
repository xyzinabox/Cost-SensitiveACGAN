import torch
import torch.nn as nn

class netD(nn.Module):
    def __init__(self, num_classes=10):
        super(netD, self).__init__()
        
        # Convolution 1 
        self.conv1 = nn.Sequential(
            nn.Conv2d(1, 8, 5, 1, 0, bias=False),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Dropout(0.3, inplace=False),
        )
        # Convolution 2 
        self.conv2 = nn.Sequential(
            nn.Conv2d(8, 16, 3, 1, 0, bias=False),
            nn.BatchNorm2d(16),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Dropout(0.3, inplace=False),
        )
        # Convolution 3 
        self.conv3 = nn.Sequential(
            nn.Conv2d(16, 32, 5, 1, 0, bias=False),
            nn.BatchNorm2d(32),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Dropout(0.3, inplace=False),
        )
        # Convolution 4
        self.conv4 = nn.Sequential(
            nn.Conv2d(32, 64, 3, 1, 0, bias=False),
            nn.BatchNorm2d(64),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Dropout(0.3, inplace=False),
        )
        # # Convolution 5
        # self.conv5 = nn.Sequential(
        #     nn.Conv2d(128, 256, 3, 1, 0, bias=False),
        #     nn.BatchNorm2d(256),
        #     nn.LeakyReLU(0.2, inplace=True),
        #     nn.Dropout(0.5, inplace=False),
        # )
        # # Convolution 6
        # self.conv6 = nn.Sequential(
        #     nn.Conv2d(256, 512, 3, 1, 0, bias=False),
        #     nn.BatchNorm2d(512),
        #     nn.LeakyReLU(0.2, inplace=True),
        #     nn.Dropout(0.5, inplace=False),
        # )
        # discriminator fc
        self.fc_dis = nn.Linear(16*16*64, 1)
        # aux-classifier fc
        self.fc_aux = nn.Linear(16*16*64, num_classes)
        # softmax and sigmoid
       
        self.sigmoid = nn.Sigmoid()

    def forward(self, input):

        conv1 = self.conv1(input)
        conv2 = self.conv2(conv1)
        conv3 = self.conv3(conv2)
        conv4 = self.conv4(conv3)
        # conv5 = self.conv5(conv4)
        # conv6 = self.conv6(conv5)
        flat6 = conv4.view(-1, 16*16*64)

        fc_dis = self.fc_dis(flat6)
        fc_aux = self.fc_aux(flat6)

        # classes = self.softmax(fc_aux)
        realfake = self.sigmoid(fc_dis).view(-1, 1).squeeze(1)
        
        return realfake, fc_aux

class netG(nn.Module):
    def __init__(self, nz):
        super(netG, self).__init__()
        
        self.nz = nz

        # first linear layer
        self.fc1 = nn.Linear(nz,256)
        # Transposed Convolution 2 5x5
        self.tconv2 = nn.Sequential(
            nn.ConvTranspose2d(256, 256, 5, 2, 0, bias=False),
            nn.BatchNorm2d(256),
            nn.ReLU(True),
        )
        # Transposed Convolution 3 13x13
        self.tconv3 = nn.Sequential(
            nn.ConvTranspose2d(256, 128, 5, 2, 0, bias=False),
            nn.BatchNorm2d(128),
            nn.ReLU(True),
        )
        # Transposed Convolution 4 17x17
        self.tconv4 = nn.Sequential(
            nn.ConvTranspose2d(128, 64, 5, 1, 0, bias=False),
            nn.BatchNorm2d(64),
            nn.ReLU(True),
        )
        # Transposed Convolution 5 21x21
        self.tconv5 = nn.Sequential(
            nn.ConvTranspose2d(64, 32, 5, 1, 0, bias=False),
            nn.BatchNorm2d(32),
            nn.ReLU(True),
        )
        # Transposed Convolution 5 28x28
        self.tconv6 = nn.Sequential(
            nn.ConvTranspose2d(32, 1, 8, 1, 0, bias=False),
            nn.Tanh(),
        )

    def forward(self, input):
        
        fc1 = self.fc1(input)
        fc1 = fc1.view(-1, 256, 1, 1)
        tconv2 = self.tconv2(fc1)
        tconv3 = self.tconv3(tconv2)
        tconv4 = self.tconv4(tconv3)
        tconv5 = self.tconv5(tconv4)
        tconv5 = self.tconv6(tconv5)
        output = tconv5
        return output



class netDv1(nn.Module):
    def __init__(self, num_classes=10):
        super(netDv1, self).__init__()
        
        # Convolution 1 
        self.conv1 = nn.Sequential(
            nn.Conv2d(1, 64, 3, 1, 0, bias=False),
            nn.LeakyReLU(0.2, inplace=True),
            
        )
        # Convolution 2 
        self.conv2 = nn.Sequential(
            nn.Conv2d(64, 128, 3, 1, 0, bias=False),
            nn.BatchNorm2d(128),
            nn.LeakyReLU(0.2, inplace=True),
            
        )
        # Convolution 3 
        self.conv3 = nn.Sequential(
            nn.Conv2d(128, 128, 3, 1, 0, bias=False),
            nn.BatchNorm2d(128),
            nn.LeakyReLU(0.2, inplace=True),
           
        )
        # Convolution 4
        self.conv4 = nn.Sequential(
            nn.Conv2d(128, 256, 3, 1, 0, bias=False),
            nn.BatchNorm2d(256),
            nn.LeakyReLU(0.2, inplace=True),
            nn.MaxPool2d(2, stride=2)
        )
        # Convolution 5
        self.conv5 = nn.Sequential(
            nn.Conv2d(256, 256, 3, 1, 0, bias=False),
            nn.BatchNorm2d(256),
            nn.LeakyReLU(0.2, inplace=True),
            
        )
        # Convolution 6
        self.conv6 = nn.Sequential(
            nn.Conv2d(256, 256, 3, 1, 0, bias=False),
            nn.BatchNorm2d(256),
            nn.LeakyReLU(0.2, inplace=True),
            
        )
        # discriminator fc
        self.fc_dis = nn.Linear(6*6*256, 1)
        # aux-classifier fc
        self.fc_aux = nn.Linear(6*6*256, num_classes)
       
        self.sigmoid = nn.Sigmoid()

    def forward(self, input):

        conv1 = self.conv1(input)
        conv2 = self.conv2(conv1)
        conv3 = self.conv3(conv2)
        conv4 = self.conv4(conv3)
        conv5 = self.conv5(conv4)
        conv6 = self.conv6(conv5)
        flat6 = conv6.view(-1, 6*6*256)

        fc_dis = self.fc_dis(flat6)
        fc_aux = self.fc_aux(flat6)

        # classes = self.softmax(fc_aux)
        realfake = self.sigmoid(fc_dis).view(-1, 1).squeeze(1)
        
        return realfake, fc_aux

class netGv1(nn.Module):
    def __init__(self, nz):
        super(netGv1, self).__init__()
        
        self.nz = nz

        # first linear layer
        self.fc1 = nn.Linear(nz,256)
        # Transposed Convolution 2 5x5
        self.tconv2 = nn.Sequential(
            nn.ConvTranspose2d(110, 256, 5, 2, 0, bias=False),
            nn.InstanceNorm2d(256),
            nn.ReLU(True),
        )
        # Transposed Convolution 3 13x13
        self.tconv3 = nn.Sequential(
            nn.ConvTranspose2d(256, 128, 5, 2, 0, bias=False),
            nn.InstanceNorm2d(128),
            nn.ReLU(True),
        )
        # Transposed Convolution 4 17x17
        self.tconv4 = nn.Sequential(
            nn.ConvTranspose2d(128, 64, 5, 1, 0, bias=False),
            nn.InstanceNorm2d(64),
            nn.ReLU(True),
        )
        # Transposed Convolution 5 21x21
        self.tconv5 = nn.Sequential(
            nn.ConvTranspose2d(64, 64, 5, 1, 0, bias=False),
            nn.InstanceNorm2d(64),
            nn.ReLU(True),
        )
        # Transposed Convolution 5 28x28
        self.tconv6 = nn.Sequential(
            nn.ConvTranspose2d(64, 1, 8, 1, 0, bias=False),
            nn.Tanh(),
        )

    def forward(self, input):
        
        fc1 = self.fc1(input)
        fc1 = fc1.view(-1, 256, 1, 1)
        tconv2 = self.tconv2(fc1)
        tconv3 = self.tconv3(tconv2)
        tconv4 = self.tconv4(tconv3)
        tconv5 = self.tconv5(tconv4)
        tconv5 = self.tconv6(tconv5)
        output = tconv5
        return output