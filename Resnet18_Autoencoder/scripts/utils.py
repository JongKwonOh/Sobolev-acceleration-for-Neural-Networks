from tqdm import tqdm
import numpy as np
import torch
import matplotlib.pyplot as plt
import os


def train_epoch(cae, device, dataloader, loss_fn, optimizer, training_type):
    """The training loop of autoencoder.
    
    Args:
        cae (classes.resnet_autoencoder.AE): the autoencoder model with - by default- random initilized weights.
        device (str): if exists, the accelarator device used from the machine and supported from the pytorch else cpu.
        dataloader (DataLoader): loader with the training data.
        loss_fn (torch.nn.modules.loss): the loss function of the autoencoder
        optimizer (torch.optim): the optimizer of the autoencoder 

    Returns:
        (float): the mean of training loss
    """
    # Set train mode for both the encoder and the decoder
    cae.train()
    train_loss = []
    # Iterate the dataloader (we do not need the label values, this is unsupervised learning)
    for _, (x_batch, y_batch) in tqdm(enumerate(dataloader), total=int(len(dataloader.dataset)/dataloader.batch_size)): # with "_" we just ignore the labels (the second element of the dataloader tuple)
        # Move tensor to the proper device
        x_batch = x_batch.to(device)
        
        # CAE data
        decoded_batch,_ = cae(x_batch)
        # Evaluate loss
        if training_type=='L2':
            l2_loss = loss_fn(decoded_batch, x_batch)
            loss = l2_loss
        elif training_type=='H1':
            l2_loss = loss_fn(decoded_batch, x_batch)
            h1_loss = cheb_loss2(decoded_batch - x_batch, power=1, device=device)
            loss = h1_loss
        # Backward pass
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        # Print batch loss
        train_loss.append(l2_loss.detach().cpu().numpy())

    return np.mean(train_loss)


def test_epoch(cae, device, dataloader, loss_fn):
    """The validation loop of autoencoder on the test dataset.
    
    Args:
        cae (classes.resnet_autoencoder.AE): the autoencoder model.
        device (str): if exists, the accelarator device used from the machine and supported from the pytorch else cpu.
        dataloader (DataLoader): loader with the test data.
        loss_fn (torch.nn.modules.loss): the loss function of the autoencoder.

    Returns:
        (float): the validation loss.
    """
    # Set evaluation mode for encoder and decoder
    cae.eval()
    with torch.no_grad(): # No need to track the gradients
        # Define the lists to store the outputs for each batch
        decoded_data = []
        original_data = []
        for x_batch, _ in dataloader:
            # Move tensor to the proper device
            x_batch = x_batch.to(device)
            
            # CAE data
            decoded_batch,_ = cae(x_batch)
            # Append the network output and the original image to the lists
            decoded_data.append(decoded_batch.cpu())
            original_data.append(x_batch.cpu())
        # Create a single tensor with all the values in the lists
        decoded_data = torch.cat(decoded_data)
        original_data = torch.cat(original_data)
        # Evaluate global loss
        val_loss = loss_fn(decoded_data, original_data)

    return val_loss.data


def plot_ae_outputs(cae, dataset_opt, epoch, dataset, device, training_type, n=10):
    """
    각 클래스별로 10장의 이미지를 선택하여 AutoEncoder의 재구성 이미지를 10x10 그리드로 저장합니다.
    
    Args:
        cae (classes.resnet_autoencoder.AE): 학습된 AutoEncoder 모델.
        dataset_opt (str): 데이터셋 이름 (예: 'train_dataset', 'test_dataset')
        epoch (int): 현재 에폭.
        dataset: 이미지와 타겟을 가진 데이터셋.
        device (str): 사용 장치 (예: 'cpu' 또는 'cuda').
        training_type (str): 저장 경로에 사용될 학습 타입 문자열.
        n (int): 클래스 수 (기본 10, 각 클래스당 10장씩 사용).
    """
    targets = np.array(dataset.targets)
    t_idx = np.arange(100).reshape(10, 10)
    plt.figure(figsize=(20, 20))
    
    for i in range(n):
        for j, idx in enumerate(t_idx[i]):
            img, _ = dataset[idx]
            img_input = img.unsqueeze(0).to(device)
            cae.eval()
            with torch.no_grad():
                rec_img, _ = cae(img_input)
            rec_img = rec_img.cpu().squeeze()
            
            ax = plt.subplot(n, n, i * n + j + 1)
            plt.imshow(rec_img.permute(1, 2, 0))
            ax.get_xaxis().set_visible(False)
            ax.get_yaxis().set_visible(False)
            if j == 0:
                ax.set_ylabel(f'Class {i}', fontsize=18)
    
    if not os.path.isdir('output'):
        os.mkdir('output')
    out_dir = os.path.join('output', training_type)
    if not os.path.isdir(out_dir):
        os.mkdir(out_dir)
    
    plt.suptitle(f'Reconstructed Images from {dataset_opt} at epoch {epoch}', fontsize=16)
    plt.tight_layout(rect=[0, 0, 1, 0.95])
    plt.savefig(f'{out_dir}/{epoch}_epoch_from_{dataset_opt}.png')
    plt.close()


def checkpoint(model, epoch, val_loss, filename, training_type):
    """Saving the model at a specific state.

    Args:
        model (classes.resnet_autoencoder.AE): the trained autoencoder model.
        epoch (int): the present epoch in progress.
        val_loss (float): the validation loss.
        filename (str): the relative path of the file where the model will be stored.
    """
    filename = filename.format(training_type, epoch)
    # torch.save(model.state_dict(), filename)filename = filename.format(training_type, epoch)
    
    folder = os.path.dirname(filename)
    os.makedirs(folder, exist_ok=True)

    torch.save({
            'epoch': epoch,
            'model_state_dict': model.state_dict(),
            # 'optimizer_state_dict': optimizer.state_dict(),
            'val_loss': val_loss,
            }, filename)

def resume(model, filename):
    """Load the trained autoencoder model.

    Args:
        model (classes.resnet_autoencoder.AE): the untrained autoencoder model.
        filename (str): the relative path of the file where the model is stored.

    Results:
        model (classes.resnet_autoencoder.AE): the loaded autoencoder model.
        epoch (int): the last epoch of the training procedure of the model.
        loss (float): the validation loss of the last epoch.
    """
    # checkpoint = torch.load(filename)
    # model.load_state_dict(checkpoint['model_state_dict'])
    # # optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
    # epoch = checkpoint['epoch']
    # loss = checkpoint['val_loss']
    checkpoint = torch.load(filename)
    model.load_state_dict(checkpoint['model_state_dict'])
    epoch = checkpoint['epoch']
    loss = checkpoint['val_loss']

    return model, epoch, loss


def cheb_loss2(out, power, device):
    n = out.size(-1)
    xi = (torch.range(0,n-1)*(2*np.pi/n)).to(device)
    fhat_x = torch.fft.fft(out, dim=-1)
    dx = torch.fft.ifft(fhat_x*((1+xi**2)**(power/2))).abs()
    
    fhat_y = torch.fft.fft(out, dim=-2)
    dy = torch.fft.ifft(fhat_y*((1+xi**2)**(power/2)).view(-1,1)).abs()
    return dx.pow(2).mean() + dy.pow(2).mean() #dfFFT_x.pow(2).mean() + dfFFT_y.pow(2).mean()