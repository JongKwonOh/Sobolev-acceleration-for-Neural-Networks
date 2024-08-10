import torch
import torchvision.datasets as datasets
import torchvision.transforms as transforms

def get_train_data(root='./data/02/', download=True, transform=transforms.ToTensor(), batch_size=256):
    train_data = datasets.MNIST(root=root, train=True, download=download, transform=transform)
    train_loader = torch.utils.data.DataLoader(dataset=train_data,
                                           batch_size = batch_size, shuffle = True)
    return train_loader

def get_test_data(root='./data/02/', download=True, transform=transforms.ToTensor(), batch_size=256):
    test_data = datasets.MNIST(root=root, train=False, download=download, transform=transform)
    test_loader = torch.utils.data.DataLoader(dataset=test_data,
                                            batch_size = batch_size, shuffle = True)
    return test_loader
