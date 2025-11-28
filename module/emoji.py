from module.basic import trans_totrain, trans_toeval
import module.basic as basic
import argparse
import os
import random
import torch
import torch.nn as nn
import torch.nn.parallel
import torch.optim as optim
import torch.utils.data
import torchvision.datasets as dset
import torchvision.transforms as transforms
import torchvision.utils as vutils
import numpy as np
import matplotlib.pyplot as plt
from torchvision.transforms import v2
import matplotlib.animation as animation
from PIL import Image

from module.gan import *

image_size = 64
batch_size = 64

augmentation = transforms.Compose([
        v2.RandomAffine(degrees=(-20,20), translate=(0.05,0.05), scale=(0.9,1.1), shear=(-1, 1)),
        v2.RandomHorizontalFlip(),
        v2.ColorJitter(brightness=(0.7,1.3), contrast=(0.5,1.5), saturation=(0.5,1.5))
        # v2.RandomRotation(30)
    ])

class  emoji():
    def __init__(self, show=True):
        self.device = torch.device("cuda:0" if (torch.cuda.is_available()) else "cpu")
        self.preparedata()

        if show:
            self.prepare_model(0.0002, 0.0002, 'netD.pt', 'netG.pt')
        else:
            self.prepare_model(0.0002, 0.0002)

        self.train_img_path = 'train\\ganimg'
        os.makedirs(self.train_img_path, exist_ok=True)
        pass
    
    def preparedata(self):
        dataset = dset.ImageFolder(root='Newemoji', transform=transforms.Compose([
            transforms.Resize(image_size),
            v2.ToImage(), 
            v2.ToDtype(torch.float32, scale=True)
            ]))
        self.dataloader = torch.utils.data.DataLoader(dataset, batch_size=batch_size, shuffle=True, num_workers=2)

        pass

    def show_example(self):
        batch = next(iter(self.dataloader))[0].to(self.device)[:64]
        aug_batch = [augmentation(img) for img in batch]

        batch = np.transpose(vutils.make_grid(batch, padding=2, normalize=True).cpu(), (1,2,0))
        aug_batch = np.transpose(vutils.make_grid(aug_batch, padding=2, normalize=True).cpu(), (1,2,0))

        plt.figure(figsize=(12, 6))

        plt.subplot(1, 2, 1)
        plt.title("original image")
        plt.axis('off')
        plt.imshow(batch)

        plt.subplot(1, 2, 2)
        plt.title("augmentation image")
        plt.axis('off')
        plt.imshow(aug_batch)

        plt.show()

    def show_gen(self):
        batch = next(iter(self.dataloader))[0].to(self.device)[:64]
        batch = np.transpose(vutils.make_grid(batch, padding=2, normalize=True).cpu(), (1,2,0))
        gen = self.genimg()
        plt.figure(figsize=(12, 6))

        plt.subplot(1, 2, 1)
        plt.title("real image")
        plt.axis('off')
        plt.imshow(batch)

        plt.subplot(1, 2, 2)
        plt.title("fake image")
        plt.axis('off')
        plt.imshow(gen)

        plt.show()

    
    def prepare_model(self, lrG, lrD, loadD=None, loadG=None):
        self.netD = Discriminator(1).to(self.device)
        if loadD is None:
            self.netD.apply(basic.weights_init) 
        else:
            self.netD.load_state_dict(torch.load(loadD, weights_only=True))
        
        self.netG = Generator(1).to(self.device)
        if loadD is None:
            self.netG.apply(basic.weights_init)
        else:
            self.netG.load_state_dict(torch.load(loadG, weights_only=True))

        
        self.print_model()

        self.criterion = nn.BCELoss()
        self.fixed_noise = torch.randn(64, nz, 1, 1, device=self.device)

        self.optimizerG = optim.Adam(self.netG.parameters(), lr=lrG, betas=(0.5, 0.999))
        self.optimizerD = optim.Adam(self.netD.parameters(), lr=lrD, betas=(0.5, 0.999))
        
    def print_model(self):
        print(self.netG)
        print(self.netD)
    
    def genimg(self, size=64):
        with torch.no_grad():
            fack = self.netG(torch.randn(size, nz, 1, 1, device=self.device)).detach().cpu()
            img_grid = vutils.make_grid(fack, padding=2, normalize=True)
        np_img = img_grid.numpy()
        return np.transpose(np_img, (1, 2, 0))


    def train(self, num_epoch=500):
        for imgpath in os.listdir(self.train_img_path):
            os.remove(os.path.join(self.train_img_path, imgpath))
        plt.figure(figsize=(8, 8))
        plt.ion()
        print("begin Training...")
        G_loss = []
        D_loss = []
        img_list = []
        iter = 0
        imgidx = 0
        for epoch in range(num_epoch):
            for i, data in enumerate(self.dataloader, 0):
                data = torch.stack([basic.trans_toeval(augmentation(img)) for img in data[0]])
                # PART. D network
                ## Train with all-real batch
                self.netD.zero_grad()
                real = data.to(self.device)
                b_size = real.size(0)
                label = torch.full((b_size,), real_label, dtype=torch.float, device=self.device)
                output = self.netD(real).view(-1)
                errD_real = self.criterion(output, label)
                errD_real.backward()
                D_x = output.mean().item()

                ## Train with all-fake batch
                noise = torch.randn(b_size, nz, 1, 1, device=self.device)
                fack = self.netG(noise)
                label.fill_(fake_label)
                output = self.netD(fack.detach()).view(-1)
                errD_fake = self.criterion(output, label)
                errD_fake.backward()
                D_G_z1 = output.mean().item()

                errD = errD_real + errD_fake
                self.optimizerD.step()

                # PART. G network
                self.netG.zero_grad()
                label.fill_(real_label)
                output = self.netD(fack).view(-1)

                errG = self.criterion(output, label)
                errG.backward()
                D_G_z2 = output.mean().item()
                self.optimizerG.step()

                if i%50 == 0:
                    print(f'[{epoch}/{num_epoch}][{i}/{len(self.dataloader)}]\tLoss_D: {errD.item():.4f}\tLoss_G: {errG.item():.4f}\tD(x): {D_x:.4f}\tD(G(z)): {D_G_z1:.4f} / {D_G_z2:.4f}')

                G_loss.append(errG.item())
                D_loss.append(errD.item())

                if iter%500 ==0 or ((i == len(self.dataloader)-1)):
                    with torch.no_grad():
                        fack = self.netG(self.fixed_noise).detach().cpu()
                        img_grid = vutils.make_grid(fack, padding=2, normalize=True)
                    np_img = img_grid.numpy()
                    plt.savefig(f"{self.train_img_path}\\generated_{imgidx:04}.png")

                    plt.clf()  # 清空当前内容
                    plt.axis("off")
                    plt.title(f"Generated Images at iter {iter:04}")
                    plt.imshow(np.transpose(np_img, (1, 2, 0)))  # 转换为 HWC 格式
                    plt.show()  # 显示图像
                    plt.pause(0.01)  # 暂停以更新图像显示
                    imgidx+=1
                iter+=1

        plt.ioff()
        plt.show()
        torch.save(self.netG.state_dict(), "netG.pt")
        torch.save(self.netD.state_dict(), "netD.pt")




    def train_animation(self):
        folder_path = self.train_img_path
        image_files = [f for f in os.listdir(folder_path) if f.endswith(('png', 'jpg', 'jpeg'))]

        plt.figure(figsize=(8, 8)) 
        plt.ion()

        for image_file in image_files:
            image_path = os.path.join(folder_path, image_file)
            
            img = Image.open(image_path)
            img = np.array(img)
            
            plt.clf()
            plt.imshow(img)
            plt.axis('off')
            plt.title(f"{image_file}")
            plt.draw()
            plt.pause(0.05) 

        plt.ioff()
        plt.show()
