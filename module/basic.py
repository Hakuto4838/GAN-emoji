from PIL import Image
from torchvision.transforms import v2
import PyQt5.QtWidgets as qt
import torch
import matplotlib.pyplot as plt   # 資料視覺化套件
import numpy as np
import os
import torchvision.utils as vutils
import torch.nn as nn

trans = [
        v2.RandomHorizontalFlip(),
        v2.RandomRotation(30),
        v2.ColorJitter(brightness=(0.8,1.2), contrast=(0.8,1.2), saturation=(0.8, 1.2)),
        v2.RandomAffine(degrees=(-10,10), translate=(0.2,0.2), scale=(0.8,1.2), shear=(-10, 10))
    ]
ToTensor = [
    v2.ToImage(), 
    v2.ToDtype(torch.float32, scale=True),
    v2.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225))
]


trans_toshow = v2.Compose(trans)
trans_toeval = v2.Compose(ToTensor)

trans_totrain = v2.Compose(trans + ToTensor)


def random_aug(images):
    
    return [trans_toshow(img) for img in images]

def loadimg(path=None):
    if(path is None):
        path, _  = qt.QFileDialog.getOpenFileName(caption="select image", filter="Images (*.png)")
    if path:
        name = os.path.basename(path)
        return path, Image.open(path), os.path.splitext(name)[0]

def loadDir(path=None):
    images = []
    names = []
    if(path is None):
        path = qt.QFileDialog.getExistingDirectory()
    if path:
        for filename in os.listdir(path):
            if filename.endswith(('.bmp','.png', '.jpg', '.jpeg')):
                image = Image.open(os.path.join(path, filename))
                names.append(os.path.splitext(filename)[0])
                images.append(image)
    return images, names

def show_images(images, Wintitle:str='show', titles:list=None):

    # 創建一個 3x3 的子圖佈局
    fig, axes = plt.subplots(3, 3, figsize=(8, 8))
    for i, ax in enumerate(axes.flat):
        ax.imshow(images[i])
        if(titles is not None):
            ax.set_title(titles[i], fontsize=10)
        ax.axis('off')  # 隱藏坐標軸

    # 顯示圖像
    fig.canvas.manager.set_window_title(Wintitle)
    plt.tight_layout(rect=[0, 0.03, 1, 0.95])  # 調整布局，避免重疊
    plt.show()

def show_image64(images, title = ''):
    plt.clf()
    plt.figure(figsize=(8,8))
    plt.axis('off')
    plt.title(title)
    grid = vutils.make_grid(images, padding=2, normalize=True).cpu()
    plt.imshow(np.transpose(grid, (1,2,0)))

    plt.draw()
    plt.pause(0.1)

def print_bar(label, data, title=''):
    plt.clf()
    values = data.cpu().detach().numpy()
    bars = plt.bar(label, values)

    for bar, value in zip(bars, values):
        plt.text(
            bar.get_x() + bar.get_width() / 2,
            bar.get_height() + 0.01,          
            f'{value:.3f}',                   
            ha='center', fontsize=7          
        )

    plt.title(title)
    plt.xlabel('Classes', fontsize=12)
    plt.ylabel('Probability', fontsize=12)



    plt.draw()
    plt.pause(0.1)
    
def weights_init(m):
    classname = m.__class__.__name__
    if classname.find('Conv') != -1:
        nn.init.normal_(m.weight.data, 0.0, 0.02)
    elif classname.find('BatchNorm') != -1:
        nn.init.normal_(m.weight.data, 1.0, 0.02)
        nn.init.constant_(m.bias.data, 0)