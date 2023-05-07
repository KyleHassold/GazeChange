import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision
import torchvision.transforms as transforms
from torch.autograd import Variable
import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm
import os

from MPIIFaceGazeDataset import get_dataloaders, crop_eyes, insert_eyes
from GAN import Generator, Discriminator


# Number of training epochs
num_epochs = 5
# Learning rate for optimizers
lr = 0.0002
# Beta1 hyperparameter for Adam optimizers
beta1 = 0.5
beta2 = 0.999
# Number of workers for dataloader
workers = 2
# batch_size = 32
batch_size = 512
sample_interval = 400

def plot_images(imgs,n_row=4,n_col=4):
  _, axs = plt.subplots(n_row, n_col, figsize=(12, 12))
  axs = axs.flatten()
  for img, ax in zip(imgs, axs):
      ax.imshow(img.detach()[0].permute(1, 2, 0))
  plt.show()

train_dataloader, val_dataloader = get_dataloaders(batch_size)

if __name__ == "__main__":
    train_dataloader, val_dataloader = get_dataloaders(batch_size)

    cuda = True if torch.cuda.is_available() else False

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    Tensor = torch.cuda.FloatTensor if cuda else torch.FloatTensor

    gen = Generator()
    disc = Discriminator()
    gen = gen.cuda()
    disc = disc.cuda()
    pixelwise_loss = nn.L1Loss()
    adverserial_loss = nn.BCELoss()

    opt_g = torch.optim.Adam(gen.parameters(),lr=lr,betas=(beta1, beta2))
    opt_d = torch.optim.Adam(disc.parameters(),lr=lr,betas=(beta1, beta2))

    # ----------
    #  Training
    # ----------

    for epoch in range(num_epochs):
        gen.train()
        disc.train()

        train_g_loss = []
        val_g_loss = []
        train_d_loss = []
        val_d_loss = []
        plt_imgs = None
        for i, (real_imgs,labels) in enumerate(train_dataloader):
            real_imgs, labels = real_imgs.to(device), labels.to(device)

            # Adversarial ground truths
            valid = Variable(Tensor(real_imgs.size(0), 1).fill_(1.0), requires_grad=False)
            fake = Variable(Tensor(real_imgs.size(0), 1).fill_(0.0), requires_grad=False)

            valid.to(device)
            fake.to(device)
            # -----------------
            #  Train Generator
            # -----------------

            opt_g.zero_grad()

            # Sample noise as generator input
            eyes = crop_eyes(real_imgs, labels)

            # Generate a batch of images
            gen_eyes = gen(eyes)
            gen_imgs = insert_eyes(real_imgs, labels, gen_eyes)
            
            # Loss measures generator's ability to fool the discriminator
            # g_loss = BCE_loss(disc(gen_imgs), valid)
            g_loss = pixelwise_loss(gen_imgs, real_imgs)

            train_g_loss.append(g_loss.item())
            g_loss.backward()
            opt_g.step()

            # ---------------------
            #  Train Discriminator
            # ---------------------

            opt_d.zero_grad()

            # Measure discriminator's ability to classify real from generated samples
            real_loss = adverserial_loss(disc(real_imgs), valid)
            fake_loss = adverserial_loss(disc(gen_imgs.detach()), fake)
            d_loss = (real_loss + fake_loss) / 2

            train_d_loss.append(d_loss.item())

            d_loss.backward()
            opt_d.step()

            
            print(
                "Training [Epoch %d/%d] [Batch %d/%d] [D loss: %f] [G loss: %f]"
                % (epoch, num_epochs, i, len(train_dataloader), d_loss.item(), g_loss.item())
            )

            if i % 400 == 0:
                gen_imgs = gen_imgs.cpu()
                print('Training Generated Outputs')
                plot_images(gen_imgs[:16])

            
        
        gen.eval()
        disc.eval()
        for i, (real_imgs,labels) in enumerate(val_dataloader):
         with torch.no_grad():
            real_imgs, labels = real_imgs.to(device), labels.to(device)

            valid = Variable(Tensor(real_imgs.size(0), 1).fill_(1.0), requires_grad=False)
            fake = Variable(Tensor(real_imgs.size(0), 1).fill_(0.0), requires_grad=False)

            valid.to(device)
            fake.to(device)

            eyes = crop_eyes(real_imgs, labels)

            gen_eyes = gen(eyes)
            gen_imgs = insert_eyes(real_imgs, labels, gen_eyes)
            
            g_loss = pixelwise_loss(gen_imgs, real_imgs) 
            val_g_loss.append(val_g_loss.item())

            real_loss = adverserial_loss(disc(real_imgs), valid)
            fake_loss = adverserial_loss(disc(gen_imgs.detach()), fake)
            d_loss = (real_loss + fake_loss) / 2
            val_d_loss.append(val_d_loss.item()) 

            if i % 400 == 0:
                gen_imgs = gen_imgs.cpu()
                print('Validation Generated Outputs')
                plot_images(gen_imgs[:16])
