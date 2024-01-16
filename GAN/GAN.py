# 参考：https://github.com/TeeyoHuang/Generative-Adversarial-Nets-pytorch
# 论文：https://arxiv.org/abs/1406.2661

import argparse
import os
import numpy as np
import math

import torchvision.transforms as transforms
from torchvision.utils import save_image

from torch.utils.data import DataLoader
from torchvision import datasets
from torch.autograd import Variable

import torch.nn as nn
import torch.nn.functional as F
import torch

parser = argparse.ArgumentParser()
parser.add_argument('--n_epochs', type=int, default=100, help='number of epochs of training')
parser.add_argument('--batch_size', type=int, default=64, help='size of the batches')
parser.add_argument('--lr', type=float, default=0.0002, help='adam: learning rate')
parser.add_argument('--beta1', type=float, default=0.5, help='adam: decay of first order momentum of gradient')
parser.add_argument('--beta2', type=float, default=0.999, help='adam: decay of second order momentum of gradient')
parser.add_argument('--latent_dim', type=int, default=100, help='dimensionality of the latent space')
parser.add_argument('--img_size', type=int, default=28, help='size of each image dimension')
parser.add_argument('--channels', type=int, default=1, help='number of image channels')
parser.add_argument('--sample_interval', type=int, default=400, help='interval betwen image samples')
parser.add_argument('--results_dir', type=str, default='./result_images', help='directory to save the results')
args = parser.parse_args()
print(args)

os.makedirs(args.results_dir, exist_ok=True)
C,H,W = args.channels, args.img_size, args.img_size

class Generator(nn.Module):
    def __init__(self):
        super(Generator, self).__init__()
        self.model = nn.Sequential(
            nn.Linear(args.latent_dim, 256),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Linear(256, 512),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Linear(512, 1024),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Linear(1024, C*H*W),
            nn.Tanh(),
        )

    def forward(self, z):
        img = self.model(z)
        img = img.view(img.size(0), C,H,W)
        return img

class Discriminator(nn.Module):
    def __init__(self):
        super(Discriminator, self).__init__()

        self.model = nn.Sequential(
            nn.Linear(C*H*W, 512),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Linear(512, 256),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Linear(256, 1),
            nn.Sigmoid()
        )

    def forward(self, img):
        img_flat = img.view(img.size(0), -1)
        probability = self.model(img_flat)

        return probability

# Loss function
adversarial_loss = torch.nn.BCELoss()

# Initialize generator and discriminator
generator = Generator()
discriminator = Discriminator()

if torch.cuda.is_available():
    generator.cuda()
    discriminator.cuda()
    adversarial_loss.cuda()

# Configure data loader
os.makedirs('./data', exist_ok=True)
dataloader = torch.utils.data.DataLoader(
    datasets.MNIST('./data', train=True, download=True,
                   transform=transforms.Compose([
                       transforms.ToTensor(),
                       transforms.Normalize((0.5, ), (0.5,))
                   ])),
    batch_size=args.batch_size, shuffle=True, drop_last=True)

# optimizers
optimizer_G = torch.optim.Adam(generator.parameters(), lr=args.lr, betas=(args.beta1, args.beta2))
optimizer_D = torch.optim.Adam(discriminator.parameters(), lr=args.lr, betas=(args.beta1, args.beta2))

# ----------
#  Training
# ----------
for epoch in range(args.n_epochs):
    for i, (imgs, _) in enumerate(dataloader):

        Batch_size = args.batch_size
        # Adversarial ground truths
        valid = Variable(torch.ones(Batch_size, 1).cuda(),requires_grad=False)
        fake = Variable(torch.zeros(Batch_size, 1).cuda(),requires_grad=False)

        # Configure input
        real_imgs = Variable(imgs.type(torch.FloatTensor).cuda())

        # -----------------
        #  Train Generator
        # -----------------

        optimizer_G.zero_grad()
        # Sample noise as generator input
        z = Variable(torch.FloatTensor(np.random.normal(0, 1, (Batch_size, args.latent_dim))).cuda())
        # Generate a batch of images
        gen_imgs = generator(z)
        # Loss measures generator's ability to fool the discriminator
        PRO_D_fake = discriminator(gen_imgs)
        g_loss = adversarial_loss(PRO_D_fake, valid)

        g_loss.backward()
        optimizer_G.step()

        # ---------------------
        #  Train Discriminator
        # ---------------------

        optimizer_D.zero_grad()
        # Measure discriminator's ability to classify real from generated samples
        PRO_D_real = discriminator(real_imgs)
        PRO_D_fake = discriminator(gen_imgs.detach())

        real_loss = adversarial_loss(PRO_D_real, valid)
        fake_loss = adversarial_loss(PRO_D_fake, fake)
        d_loss = (real_loss + fake_loss)

        d_loss.backward()
        optimizer_D.step()

        print ("[Epoch %d/%d] [Batch %d/%d] [D loss: %f] [G loss: %f]"
                % (epoch, args.n_epochs, i, len(dataloader),d_loss.data.cpu(), g_loss.data.cpu()))
        print ("[PRO_D_real: %f ]     [PRO_D_fake: %f] " %( torch.mean(PRO_D_real.data.cpu()),torch.mean(PRO_D_fake.data.cpu()) ))

        batches_done = epoch * len(dataloader) + i
        if batches_done % args.sample_interval == 0:
            save_image(gen_imgs.data[:25], args.results_dir+'/%d-%d.png' % (epoch, batches_done), nrow=5, normalize=True)
