from tqdm import tqdm_notebook
from torch.fft import fft,ifft
import numpy as np
import time
import torch
from torch.autograd import Variable
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
import matplotlib.pyplot as plt
import yaml
from model import get_model
from func import get_functions
from domain import get_domain
from matrix import FDM_matrix, cheb_matrix
import pickle
from copy import deepcopy

def calculate_derivative(y, x, device) :
    return torch.autograd.grad(y, x, create_graph=True, \
                        grad_outputs=torch.ones(y.size()).to(device))[0]

def chebfft_loss(NN, target, X, Y, xdercheb, ydercheb, device):
    xder = calculate_derivative(NN, X, device)
    yder = calculate_derivative(NN, Y, device)
    
    # xdercheb = (der_mat@target.T*2.0/(domain['x_range'][1] - domain['x_range'][0])).T
    # ydercheb = der_mat@target*2.0/(domain['y_range'][1] - domain['y_range'][0])
    
    cheb_loss = torch.mean((xder - xdercheb)**2) + torch.mean((yder - ydercheb)**2)
    l2_loss = torch.mean((NN - target)**2)
    
    loss = cheb_loss + l2_loss
    
    return cheb_loss, l2_loss, loss

def FDM_loss(NN, target, X, Y, xderfdm, yderfdm, device):
    xder = calculate_derivative(NN, X, device)
    yder = calculate_derivative(NN, Y, device)
    
    # xderfdm = xder_mat@target
    # yderfdm = (yder_mat@target.T).T
    
    fdm_loss = torch.mean((xder[1:-1, :] - xderfdm)**2) + torch.mean((yder[:, 1:-1] - yderfdm)**2)
    l2_loss = torch.mean((NN - target)**2)
    
    loss = fdm_loss + l2_loss
    
    return fdm_loss, l2_loss, loss

def der_loss(NN, target, X, Y, xder_acc, yder_acc, device):
    xder_NN = calculate_derivative(NN, X, device)
    yder_NN = calculate_derivative(NN, Y, device)
    
#     xder_acc = xder_f(X, Y)
#     yder_acc = yder_f(X, Y)
    
    derivative_loss = torch.mean((xder_NN - xder_acc)**2) + torch.mean((yder_NN - yder_acc)**2)
    l2_loss = torch.mean((NN - target)**2)
    
    loss = derivative_loss + l2_loss
    
    return derivative_loss, l2_loss, loss

def main():
    with open('config.yaml', 'r') as file:
        config = yaml.safe_load(file)

    train_type = config['train_type']; func_type = config['func_type']
    domain = config['domain']
    
    num_features = config['model']['num_features']
    optimizer_args = config['optimizer']['args']

    train_loop = config['train_loop']
    epochs = config['epoch']; print_epoch = config['print_epoch']
    save_path_template = config['save_path']

    device = torch.device(f"cuda:{config['device_num']}" if torch.cuda.is_available() else "cpu")
    save_path = save_path_template.format(func_type, train_type)

    X, Y = get_domain(train_type, domain, device)
    f, xder_f, yder_f = get_functions(func_type)
    target = f(X, Y) # ; xder = xder_f(X, Y); yder = yder_f(X, Y)
    
    # Get Matrix & Derivative (cheb, FDM and exact)
    if train_type == 'cheb':
        der_matrix = cheb_matrix(target, device)
        xdercheb = (der_matrix@target.T*2.0/(domain['x_range'][1] - domain['x_range'][0])).T
        ydercheb = der_matrix@target*2.0/(domain['y_range'][1] - domain['y_range'][0])
        
    elif train_type == 'FDM':
        x = torch.linspace(domain['x_range'][0], domain['x_range'][1], domain['N'])
        y = torch.linspace(domain['y_range'][0], domain['y_range'][1], domain['N'])
        xder_matrix = FDM_matrix(x, device)
        yder_matrix = FDM_matrix(y, device)
        xderfdm = xder_matrix@target
        yderfdm = (yder_matrix@target.T).T
        
    elif train_type == 'exact':
        xder_acc = xder_f(X, Y)
        yder_acc = yder_f(X, Y)
        
    elif train_type == 'L2':
        pass
    else :
        raise ValueError(f"Unknown Training type: {train_type}")
        
    #########
    mean_err_list = []; mean_time_list = []; mean_epochs_list = []
    
    if train_type=='cheb':
        for l in range(train_loop):
            model = get_model(num_features).to(device)
            optimizer = torch.optim.Adam([{'params': model.parameters()}], **optimizer_args)
            
            err_list = []
            epochs_list = []
            time_list = []
            flag_point = deepcopy(config['flag_point'])
            
            print('--------------------------------------')
            print(f"Loop : {l+1} / {train_loop}")
            start_time = time.time()
            
            for epoch in tqdm_notebook(range(epochs)):
                optimizer.zero_grad()
                y_pred = model(X, Y).view(target.size())
                cheb_loss, l2_loss, loss = chebfft_loss(y_pred, target, X, Y, xdercheb, ydercheb, device)
                loss.backward(retain_graph=True)
                optimizer.step()

                err_list.append((torch.mean((target - y_pred)**2)).item())
                
                # if err_list[-1] > 1 and epoch > 50000 :
                #     print(err_list[-1])
                #     break 
                
                if len(flag_point)==0 : break

                if torch.mean((target - y_pred)**2) < flag_point[0]:
                    epochs_list.append(epoch)
                    time_list.append((time.time() - start_time))
                    del flag_point[0]

                if epoch % print_epoch == 0 or epoch==epochs-1:
                    print(f"Epoch [{epoch+1}/{epochs}], Cheb Loss : {cheb_loss.item():.8f}, l2 loss : {l2_loss.item():.8f}, Total loss : {loss.item():.8f}")
                       
            end_time = time.time()

            elapsed_time = end_time - start_time
            
            print('--------------------------------------')
            print('Time to run :', round(elapsed_time,4),'s')   
            
        mean_err_list.append(err_list)
        mean_time_list.append(time_list)
        mean_epochs_list.append(epochs_list) 
            
    elif train_type=='FDM':
        for l in range(train_loop):
            model = get_model(num_features).to(device)
            optimizer = torch.optim.Adam([{'params': model.parameters()}], **optimizer_args)
            
            err_list = []
            epochs_list = []
            time_list = []
            flag_point = deepcopy(config['flag_point'])
            
            print('--------------------------------------')
            print(f"Loop : {l+1} / {train_loop}")
            start_time = time.time()
            
            for epoch in tqdm_notebook(range(epochs)):
                optimizer.zero_grad()
                y_pred = model(X, Y).view(target.size())
                fdm_loss, l2_loss, loss = FDM_loss(y_pred, target, X, Y, xderfdm, yderfdm, device)
                loss.backward(retain_graph=True)
                optimizer.step()

                err_list.append((torch.mean((target - y_pred)**2)).item())
                
                # if err_list[-1] > 1 and epoch > 50000 :
                #     print(err_list[-1])
                #     break 
                
                if len(flag_point)==0 : break

                if torch.mean((target - y_pred)**2) < flag_point[0]:
                    epochs_list.append(epoch)
                    time_list.append((time.time() - start_time))
                    del flag_point[0]

                if epoch % print_epoch == 0 or epoch==epochs-1:
                    print(f"Epoch [{epoch+1}/{epochs}], FDM_Loss : {fdm_loss.item():.8f}, l2_loss : {l2_loss.item():.8f}, Total_loss : {loss.item():.8f}")
            
            end_time = time.time()

            elapsed_time = end_time - start_time
            
            print('--------------------------------------')
            print('Time to run :', round(elapsed_time,4),'s') 
            
        mean_err_list.append(err_list)
        mean_time_list.append(time_list)
        mean_epochs_list.append(epochs_list)
            
    elif train_type == 'exact':
        for l in range(train_loop):
            model = get_model(num_features).to(device)
            optimizer = torch.optim.Adam([{'params': model.parameters()}], **optimizer_args)
            
            err_list = []
            epochs_list = []
            time_list = []
            flag_point = deepcopy(config['flag_point'])
            print('--------------------------------------')
            print(f"Loop : {l+1} / {train_loop}")
            start_time = time.time()
            
            for epoch in tqdm_notebook(range(epochs)):
                optimizer.zero_grad()
                y_pred = model(X, Y).view(target.size())
                derivative_loss, l2_loss, loss = der_loss(y_pred, target, X, Y, xder_acc, yder_acc, device)
                loss.backward(retain_graph=True)
                optimizer.step()

                err_list.append((torch.mean((target - y_pred)**2)).item())
                
                # if err_list[-1] > 1 and epoch > 50000 :
                #     print(err_list[-1])
                #     break 
                
                if len(flag_point)==0 : break

                if torch.mean((target - y_pred)**2) < flag_point[0]:
                    epochs_list.append(epoch)
                    time_list.append((time.time() - start_time))
                    del flag_point[0]

                if epoch % print_epoch == 0 or epoch==epochs-1:
                    print(f"Epoch [{epoch+1}/{epochs}], Derivative_Loss : {derivative_loss.item():.8f}, l2_loss : {l2_loss.item():.8f}, Total_loss : {loss.item():.8f}")
           
            end_time = time.time()

            elapsed_time = end_time - start_time
            
            print('--------------------------------------')
            print('Time to run :', round(elapsed_time,4),'s')
            
        mean_err_list.append(err_list)
        mean_time_list.append(time_list)
        mean_epochs_list.append(epochs_list)
            
    elif train_type=='L2':
        for l in range(train_loop):
            model = get_model(num_features).to(device)
            optimizer = torch.optim.Adam([{'params': model.parameters()}], **optimizer_args)
            l2_loss = nn.MSELoss()
            err_list = []
            epochs_list = []
            time_list = []
            flag_point = deepcopy(config['flag_point'])
            
            print('--------------------------------------')
            print(f"Loop : {l+1} / {train_loop}")
            start_time = time.time()
            
            for epoch in tqdm_notebook(range(epochs)):
                optimizer.zero_grad()
                y_pred = model(X, Y).view(target.size())
                loss = l2_loss(y_pred, target)
                loss.backward(retain_graph=True)
                optimizer.step()

                err_list.append((torch.mean((target - y_pred)**2)).item())
                
                # if err_list[-1] > 1 and epoch > 50000 :
                #     print(err_list[-1])
                #     break 
                
                if len(flag_point)==0 : break

                if torch.mean((target - y_pred)**2) < flag_point[0]:
                    epochs_list.append(epoch)
                    time_list.append((time.time() - start_time))
                    del flag_point[0]

                if epoch % print_epoch == 0 or epoch==epochs-1:
                    print(f"Epoch [{epoch+1}/{epochs}], L2_Loss : {loss.item():.8f}")
            
            end_time = time.time()

            elapsed_time = end_time - start_time
            
            print('--------------------------------------')
            print('Time to run :', round(elapsed_time,4),'s')
            
        mean_err_list.append(err_list)
        mean_time_list.append(time_list)
        mean_epochs_list.append(epochs_list)
        
    err_list = np.array(mean_err_list).mean(axis=0)
    time_list = np.array(mean_time_list).mean(axis=0)
    epochs_list = np.array(mean_epochs_list).mean(axis=0)
    
    save_info = {'err_list' : err_list, 'epochs' : epochs_list, 'Running Time' : time_list}
    
    with open(f"{save_path}","wb") as f:
        pickle.dump(save_info, f)
        
    print("Save Information - [Error, Flag point(Epochs), Flag point(Run Time)]")

if __name__ == "__main__":
    main()