
from scripts.config import BATCH_SIZE, EPOCHS, MODEL_FILENAME_TEMPLATE, EARLY_STOP_THRESH, LR, WORKERS, TRAINING_TYPE
from classes.resnet_autoencoder import AE
from scripts.data_loading import get_imagenet_data_loaders
from scripts.utils import train_epoch, test_epoch, plot_ae_outputs, checkpoint, resume
import torch
import datetime
import argparse
import pickle

if __name__=='__main__': 
    parser = argparse.ArgumentParser()
    parser.add_argument('--gpu', default=0)
    args = parser.parse_args()

    if torch.cuda.is_available():
         device = f'cuda:{args.gpu}' 
    elif torch.backends.mps.is_available():
        device = 'mps'
        print("Is the current version of PyTorch built with MPS activated?",torch.backends.mps.is_built())
    else:
        device = 'cpu'

    print("Using device:", device)

    print("Defining model...")
    cae = AE('light')
    # Define the training parameters
    params_to_optimize = [
        {'params': cae.parameters()}
    ]
    # Define the loss function
    loss_fn = torch.nn.MSELoss()
    # Define the optimizer
    optim = torch.optim.Adam(params_to_optimize, lr=LR, weight_decay=1e-05) 
    # Move model to the selected device
    cae.to(device)

    print('Loading data...')
    download_path = "your_data_path"
    train_loader, test_loader, train_dataset, test_dataset = get_imagenet_data_loaders(download_path=download_path, shuffle=True, batch_size=BATCH_SIZE, num_workers=WORKERS)
    
    # Initialize varialbes
    best_val_loss = 1000000
    best_epoch = 0
    #Training loop 
    val_errs = []
    t1 = datetime.datetime.now()
    print("Start training..")
    for epoch in range(EPOCHS):
        print('> Epoch ' + str(epoch + 1))
        train_loss =train_epoch(cae,device,train_loader,loss_fn,optim, TRAINING_TYPE)
        print("Evaluating on test set...")
        val_loss = test_epoch(cae,device,test_loader,loss_fn)
        print('\n EPOCH {}/{} \t train loss {:.8f} \t val loss {:.8f}.'.format(epoch + 1, EPOCHS,train_loss,val_loss))

        val_errs.append(val_loss.item())
        checkpoint(cae, epoch, val_loss.item(), MODEL_FILENAME_TEMPLATE, TRAINING_TYPE)
        with open(f'./save/{TRAINING_TYPE}_val_loss.pkl', 'wb') as f:
            pickle.dump(val_errs, f)
    print("Total training time:",datetime.datetime.now()-t1)