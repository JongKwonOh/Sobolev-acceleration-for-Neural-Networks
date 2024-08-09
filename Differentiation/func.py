import torch
import numpy as np

def get_functions(func_type):
    if func_type == "Acklev":
        def target(x, y):
            poly_1 = -20 * torch.exp(-0.2 * torch.sqrt(0.5 * (x ** 2 + y ** 2)))
            poly_2 = - torch.exp(0.5 * (torch.cos(2 * np.pi * x) + torch.cos(2 * np.pi * y)))
            poly_3 = torch.exp(torch.tensor(1.0)) + 20
            return (poly_1 + poly_2 + poly_3)

        def target_x(x, y):
            poly_1 = -1/(5*torch.sqrt(torch.tensor(2.0)))*(x/torch.sqrt(x**2+y**2))*(-20 * torch.exp(-0.2 * torch.sqrt(0.5 * (x ** 2 + y ** 2))))
            poly_2 = torch.pi * torch.sin(2*torch.pi*x) * (torch.exp(0.5 * (torch.cos(2 * np.pi * x) + torch.cos(2 * np.pi * y))))
    
            return poly_1 + poly_2
        
        def target_y(x, y):
            poly_1 = -1/(5*torch.sqrt(torch.tensor(2.0)))*(y/torch.sqrt(x**2+y**2))*(-20 * torch.exp(-0.2 * torch.sqrt(0.5 * (x ** 2 + y ** 2))))
            poly_2 = torch.pi * torch.sin(2*torch.pi*y) * (torch.exp(0.5 * (torch.cos(2 * np.pi * x) + torch.cos(2 * np.pi * y))))
            
            return poly_1 + poly_2

    else:
        raise ValueError(f"Unknown function type: {func_type}")

    return target, target_x, target_y