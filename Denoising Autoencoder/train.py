from tqdm import tqdm_notebook
import pickle as pkl
import numpy as np
import json
import os

import torch
from torch.autograd import Variable
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torchvision import datasets, transforms

import matplotlib.pyplot as plt
import copy
import yaml

from model import get_model
from data_loader import get_train_data, get_test_data

def make_freq_noisy(data, amp=0.3, noise_amplitude=2) :
    freq=noise_amplitude*np.pi
    x = torch.linspace(0,1, data.size()[-1])
    y = torch.linspace(0,1, data.size()[-1])
    xx,yy = torch.meshgrid(x,y)
    return data + (amp*torch.sin(freq*xx+freq*yy)).to(data.device)

def cheb_loss2(out, power, device):
    n = out.size(-1)
    xi = (torch.range(0,n-1)*(2*np.pi/n)).to(device)
    fhat_x = torch.fft.fft(out, dim=-1)
    dx = torch.fft.ifft(fhat_x*((1+xi**2)**(power/2))).abs()
    
    fhat_y = torch.fft.fft(out, dim=-2)
    dy = torch.fft.ifft(fhat_y*((1+xi**2)**(power/2)).view(-1,1)).abs()
    return dx, dy #dfFFT_x.pow(2).mean() + dfFFT_y.pow(2).mean()

def evaluate(loader, model, criterion, device):
    model.eval()
    err = 0
    for data, _ in loader:
        data = data.to(device)
        data_noise = make_freq_noisy(data)
        err += criterion(model(data_noise), data)*len(data)
        
    return err/len(loader.dataset)

def main():
    with open('train_config.yaml', 'r') as file:
        config = yaml.safe_load(file)

    noise_amplitude = config['noise_amplitude']; power = config['power']
    z_dim = config['model']['z_dim']
    optimizer_args = config['optimizer']['args']

    epoch_num = config['epoch']; print_epoch = config['print_epoch']
    batch_size = config['batch_size']
    save_path_template = config['save_path']

    device = torch.device(f"cuda:{config['device_num']}" if torch.cuda.is_available() else "cpu")
    save_path = save_path_template.format(power, noise_amplitude)
    
    train_loader = get_train_data(batch_size=batch_size)
    test_loader = get_test_data(batch_size=batch_size)
    first_batch = train_loader.__iter__().__next__()
    
    model = get_model(z_dim=z_dim).to(device)
    optimizer = torch.optim.Adam([{'params': model.parameters()}], **optimizer_args)
    criterion = nn.MSELoss()
    errs = []
        
    model.train()
    i=1
    for epoch in tqdm_notebook(range(epoch_num)):
        for data, target in train_loader:
            data = data.to(device)
            data_noise = make_freq_noisy(data)
            optimizer.zero_grad()
            output = model(data_noise)
            loss_L2 = criterion(output, data)
            der_x, der_y = cheb_loss2(output-data, power, device)
            loss_H1 = der_x.pow(2).mean() + der_y.pow(2).mean()
            (loss_H1).backward()
            optimizer.step()
            
            if i % print_epoch == 0:
                test_err = evaluate(test_loader, model, criterion, device)
                print("Train Step : {} | Loss : {:3f} | Loss : {:3f} | TestLoss : {:3f}"\
                    .format(i, loss_L2.item(), loss_H1.item(), test_err.item()))
                errs.append([loss_H1.item(), test_err.item()])
            i += 1
                
    torch.save([errs, model], save_path)
    
    print('Save')
    
    return

if __name__ == "__main__":
    main()