import torch
from torch.autograd import Variable
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset

import numpy as np
from tqdm import tqdm
import matplotlib.pyplot as plt
import yaml
from model import get_model
from func import get_functions
from optim import get_optimizer

def calculate_derivative(y, x, device) :
    return torch.autograd.grad(y, x, create_graph=True, \
                        grad_outputs=torch.ones(y.size()).to(device))[0]


def main():
    with open('config.yaml', 'r') as file:
        config = yaml.safe_load(file)

    training_type = config['training_type']

    func_type = config['func']['func_type']
    domain = config['func']['domain']
    N = config['func']['N']

    model_type = config['model']['model_type']
    num_features = config['model']['num_features']

    optimizer_name = config['optimizer']['name']
    optimizer_args = config['optimizer']['args']

    train_loop = config['train_loop']
    epoch = config['epoch']
    save_path_template = config['save_path']

    device = torch.device(f"cuda:{config['device_num']}" if torch.cuda.is_available() else "cpu")
    save_path = save_path_template.format(model_type, training_type, func_type, optimizer_name)

    target, target_derivative, target_der2 = get_functions(func_type)

    criterion = torch.nn.MSELoss()

    x = torch.linspace(domain[0], domain[1], N).view(-1, 1).to(device)
    x = x.requires_grad_(True)
    u = target(x)
    u_x = target_derivative(x)
    u_xx = target_der2(x)
    errs = []
    
    if training_type == 'L2':
        for loop in tqdm(range(train_loop)):
            u_model = get_model(model_type, num_features).to(device)
            optimizer = get_optimizer(optimizer_name, u_model, optimizer_args)
            err_list = []
            
            for i in tqdm(range(epoch)):
                if optimizer_name.lower() == "lbfgs":
                    def closure():
                        optimizer.zero_grad()  
                        output = u_model(x)   
                        loss = criterion(output, u) 
                        loss.backward(retain_graph=True) 
                        return loss

                    loss = optimizer.step(closure) 
                else:
                    optimizer.zero_grad() 
                    output = u_model(x)   
                    loss = criterion(output, u) 
                    loss.backward(retain_graph=True)
                    optimizer.step() 

                output = u_model(x)
                err = (output - u).pow(2).mean().item()
                err_list.append(err)

                if i % 1000 == 0:
                    print(f'Loss: {loss.item():.6f}, Err: {err:.6f}')
            
            errs.append(err_list)

    elif training_type == 'H1':
        for loop in tqdm(range(train_loop)):
            u_model = get_model(model_type, num_features).to(device)
            optimizer = get_optimizer(optimizer_name, u_model, optimizer_args)
            err_list = []
            
            for i in tqdm(range(epoch)):
                if optimizer_name.lower() == "lbfgs":
                    def closure():
                        optimizer.zero_grad()  
                        output = u_model(x)   
                        output_x = calculate_derivative(output, x, device)
                        loss = criterion(output, u)+criterion(output_x, u_x)
                        loss.backward(retain_graph=True) 
                        return loss

                    loss = optimizer.step(closure) 
                else:
                    optimizer.zero_grad() 
                    output = u_model(x)  
                    output_x = calculate_derivative(output, x, device) 
                    loss = criterion(output, u)+crtierion(output_x, u_x)
                    loss.backward(retain_graph=True)
                    optimizer.step()  

                output = u_model(x)
                err = (output - u).pow(2).mean().item()
                err_list.append(err)

                if i % 1000 == 0:
                    print(f'Loss: {loss.item():.6f}, Err: {err:.6f}')
            
            errs.append(err_list)

    elif training_type == 'H2':
        for loop in tqdm(range(train_loop)):
            u_model = get_model(model_type, num_features).to(device)
            optimizer = get_optimizer(optimizer_name, u_model, optimizer_args)
            err_list = []
            
            for i in tqdm(range(epoch)):
                if optimizer_name.lower() == "lbfgs":
                    def closure():
                        optimizer.zero_grad()  
                        output = u_model(x)   
                        output_x = calculate_derivative(output, x, device)
                        output_xx = calculate_derivative(output_x, x, device) 
                        loss = criterion(output, u)+criterion(output_x, u_x)+criterion(output_xx, u_xx)
                        loss.backward(retain_graph=True) 
                        return loss

                    loss = optimizer.step(closure)  
                else:
                    optimizer.zero_grad() 
                    output = u_model(x)  
                    output_x = calculate_derivative(output, x, device) 
                    output_xx = calculate_derivative(output_x, x, device) 
                    loss = criterion(output, u)+crtierion(output_x, u_x)+crtierion(output_xx, u_xx)
                    loss.backward(retain_graph=True)
                    optimizer.step() 

                output = u_model(x)
                err = (output - u).pow(2).mean().item()
                err_list.append(err)

                if i % 1000 == 0:
                    print(f'Loss: {loss.item():.6f}, Err: {err:.6f}')
            
            errs.append(err_list)

    else :
        raise ValueError(f"Unknown Training type: {training_type}")
    
    torch.save(np.array(errs).mean(axis=0), save_path)
    print(f"Loss saved to {save_path}")
    
    torch.save(u_model.state_dict(), save_path+'.pth')
    print(f"Model saved to {save_path}")

if __name__ == "__main__":
    main()
