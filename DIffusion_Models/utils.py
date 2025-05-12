import torch
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
import torchvision
from torchvision import datasets, transforms
import matplotlib.pyplot as plt
import numpy as np
import random
import tqdm
import os
from torchvision.utils import save_image

def to_uint8(img_tensor):
    img_tensor = (img_tensor.clamp(-1, 1) + 1) * 127.5
    return img_tensor.to(torch.uint8)

def get_imagenet_data_loaders(download_path, shuffle=False, batch_size=256, num_workers=10):
    train_dir = os.path.join(download_path, "train")
    test_dir = os.path.join(download_path, "test")

    transform = transforms.Compose([
        transforms.Resize((64, 64)), 
        transforms.ToTensor(),
        # transforms.Normalize([0.5]*3, [0.5]*3)  # [-1, 1]
    ])

    train_dataset = datasets.ImageFolder(train_dir, transform=transform)
    test_dataset = datasets.ImageFolder(test_dir, transform=transform)

    assert len(train_dataset) == 50000, f"Train dataset has {len(train_dataset)} images, expected 50000."
    assert len(test_dataset) == 10000, f"Test dataset has {len(test_dataset)} images, expected 10000."

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=shuffle, num_workers=num_workers)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False, num_workers=num_workers)

    return train_loader, test_loader, train_dataset, test_dataset

def save_generated_images(images, folder, start_idx=0, prefix='img'):
    for i, img in enumerate(images):
        save_image(img, os.path.join(folder, f'{prefix}_{start_idx + i:04d}.png'))

def cheb_loss2(out, power):
    n = out.size(-1)
    xi = (torch.range(0,n-1)*(2*torch.pi/n)).to(out.device)
    fhat_x = torch.fft.fft(out, dim=-1)
    dx = torch.fft.ifft(fhat_x*((1+xi**2)**(power/2))).abs()
    
    fhat_y = torch.fft.fft(out, dim=-2)
    dy = torch.fft.ifft(fhat_y*((1+xi**2)**(power/2)).view(-1,1)).abs()

    return dx.mean() + dy.mean()

def plot_images(images, ncols=4, save_name='sampled_images.png'):
    nrows = (len(images) + ncols - 1) // ncols
    plt.figure(figsize=(ncols * 2, nrows * 2))
    for i in range(len(images)):
        plt.subplot(nrows, ncols, i + 1)
        img = images[i].permute(1, 2, 0).numpy()
        plt.imshow(img)
        plt.axis("off")
    plt.tight_layout()
    plt.savefig(f'./save/{save_name}')
    plt.show()