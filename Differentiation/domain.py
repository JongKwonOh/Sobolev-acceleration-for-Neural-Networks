import torch

def get_domain(train_type, domain, device):
    if train_type == 'FDM':
        X = torch.linspace(domain['x_range'][0], domain['x_range'][1], domain['N']).to(device)
        Y = torch.linspace(domain['y_range'][0], domain['y_range'][1], domain['N']).to(device)
        
        X, Y = torch.meshgrid(X, Y)
        X = X.requires_grad_(True); Y.requires_grad_(True)
        
        return X, Y
    
    else : # Chebyshev Node
        N = domain['N']
        Xmin, Xmax = domain['x_range'][0], domain['x_range'][1]
        Ymin, Ymax = domain['y_range'][0], domain['y_range'][1]
        Xc = 0.5 * (Xmin + Xmax)
        Yc = 0.5 * (Ymin + Ymax)

        x = torch.cos(torch.pi*torch.arange(0,N+1)/N).to(device)
        X = (0.5*(Xmax - Xmin)*x + Xc).to(device)

        y = torch.cos(torch.pi*torch.arange(0,N+1)/N).to(device)
        Y = (0.5*(Ymax - Ymin)*y + Yc).to(device)

        X, Y = torch.meshgrid(X, Y)

        X = X.T; Y = Y.T

        X.requires_grad_(True); Y.requires_grad_(True)
        
        return X, Y
    