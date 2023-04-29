### Imports ###

import torch
import torch.nn as nn
import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm

from MPIIFaceGazeDataset import get_dataloaders, crop_eyes, insert_eyes


### Consts ###

### Classes ###

class Generator(nn.Module):
    def __init__(self):
        super(Generator, self).__init__()

        c = 32
        self.acti = nn.Tanh()
        self.sig = nn.Sigmoid()
        self.max_pool = nn.MaxPool2d(2)
        self.dropout = nn.Dropout(0.25)

        self.layer1_conv1 = nn.Conv2d(3, c, 3, padding=1)
        self.layer1_conv2 = nn.Conv2d(c, c, 3, padding=1)

        self.layer2_conv1 = nn.Conv2d(c, 2 * c, 3, padding=1)
        self.layer2_conv2 = nn.Conv2d(2 * c, 2 * c, 3, padding=1)

        self.layer3_conv1 = nn.Conv2d(2 * c, 4 * c, 3, padding=1)
        self.layer3_conv2 = nn.Conv2d(4 * c, 4 * c, 3, padding=1)

        self.layer4_conv1 = nn.Conv2d(4 * c, 4 * c, 3, padding=1)
        self.layer4_conv2 = nn.Conv2d(4 * c, 4 * c, 3, padding=1)

        self.up_conv1 = nn.ConvTranspose2d(4 * c, 4 * c, 2, stride=2)
        self.layer3_conv3 = nn.Conv2d(8 * c, 2 * c, 3, padding=1)
        self.layer3_conv4 = nn.Conv2d(2 * c, 2 * c, 3, padding=1)

        self.up_conv2 = nn.ConvTranspose2d(2 * c, 2 * c, 2, stride=2)
        self.layer2_conv3 = nn.Conv2d(4 * c, c, 3, padding=1)
        self.layer2_conv4 = nn.Conv2d(c, c, 3, padding=1)

        self.up_conv3 = nn.ConvTranspose2d(c, c, 2, stride=2)
        self.layer1_conv3 = nn.Conv2d(2 * c, c, 3, padding=1)
        self.layer1_conv4 = nn.Conv2d(c, c, 3, padding=1)
        self.layer1_conv5 = nn.Conv2d(c, 3, 1)

    def forward(self, img):
        out1 = self.acti(self.layer1_conv1(img))
        out1 = self.acti(self.layer1_conv2(out1))

        out2 = self.max_pool(out1)
        out2 = self.acti(self.layer2_conv1(out2))
        out2 = self.acti(self.layer2_conv2(out2))

        out3 = self.max_pool(out2)
        out3 = self.acti(self.layer3_conv1(out3))
        out3 = self.acti(self.layer3_conv2(out3))

        out = self.max_pool(out3)
        out = self.acti(self.layer4_conv1(out))
        out = self.acti(self.layer4_conv2(out))
        out = self.up_conv1(out)

        out = torch.cat((out3, out), dim=1)
        out = self.dropout(out)
        out = self.acti(self.layer3_conv3(out))
        out = self.acti(self.layer3_conv4(out))
        out = self.up_conv2(out)

        out = torch.cat((out2, out), dim=1)
        out = self.dropout(out)
        out = self.acti(self.layer2_conv3(out))
        out = self.acti(self.layer2_conv4(out))
        out = self.up_conv3(out)

        out = torch.cat((out1, out), dim=1)
        out = self.dropout(out)
        out = self.acti(self.layer1_conv3(out))
        out = self.acti(self.layer1_conv4(out))
        out = self.sig(self.layer1_conv5(out))

        return out


class Discriminator(nn.Module):
    def __init__(self):
        super(Discriminator, self).__init__()
        self.relu = nn.LeakyReLU(0.2)

        self.conv1 = nn.Conv2d(3, 16, 3, 2, 1)
        self.conv2 = nn.Conv2d(16, 32, 3, 2, 1)
        self.conv3 = nn.Conv2d(32, 64, 5, 3, 2)
        self.conv4 = nn.Conv2d(64, 1, 3, 1, 1)
        self.flat = nn.Flatten()
        self.lin = nn.Linear(25*25, 1)
        self.sig = nn.Sigmoid()

    def forward(self, img):
        out = self.relu(self.conv1(img))
        out = self.relu(self.conv2(out))
        out = self.relu(self.conv3(out))
        out = self.relu(self.conv4(out))
        out = self.flat(out)
        out = self.sig(self.lin(out))

        return out


### Run Code ###

if __name__ == "__main__":
    train_data, val_data = get_dataloaders(32)

    gen = Generator()
    disc = Discriminator()
    gen = gen.cuda()
    pixelwise_loss = torch.nn.L1Loss()
    # optim = torch.optim.Adam(gen.parameters(), lr=0.001)
    # for i, (img, labels) in enumerate(tqdm(train_data)):
    #     img = img.cuda()
    #     optim.zero_grad()
    #
    #     eyes = crop_eyes(img, labels)
    #     pred = gen(eyes)
    #     new_imgs = insert_eyes(img, labels, pred)
    #
    #     loss = pixelwise_loss(new_imgs, img)
    #     loss.backward()
    #     optim.step()
    #     if i % 10 == 0:
    #         print(loss.item())
    #     if i > 100:
    #         break

    gen = gen.cpu()
    for img, labels in train_data:
        img = img
        eyes = crop_eyes(img, labels)
        plt.imshow(eyes[0].permute(1, 2, 0))
        plt.show()
        pred = gen(eyes)
        plt.imshow(pred.detach()[0].permute(1, 2, 0))
        plt.show()

        new_imgs = insert_eyes(img, labels, pred)
        loss = pixelwise_loss(new_imgs, img)
        loss.backward()
        print(loss.item())
        plt.imshow(new_imgs.detach()[0].permute(1, 2, 0))
        plt.show()

        pred2 = disc(new_imgs)
