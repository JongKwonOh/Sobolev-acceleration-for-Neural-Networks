import torch

def get_functions(func_type):
    if func_type == "f1":
        k1 = 2
        k2 = 20
        L = 1

        def target(x):  # f1
            return torch.sin(k1 * torch.pi * x) + L * torch.sin(k2 * torch.pi * x)

        def target_derivative(x):  # df1/dx
            return (k1*torch.pi)*torch.cos(k1*torch.pi*x) + L*(k2*torch.pi)*torch.cos(k2*torch.pi*x)

        def target_der2(x):
            return -((k1*torch.pi)**2)*torch.sin(k1*torch.pi*x) - L*((k2*torch.pi)**2)*torch.sin(k2*torch.pi*x)

    elif func_type == "f2":
        def target(x):  # f2
            return x + torch.sin(torch.pi * 2 * x.pow(4))

        def target_derivative(x):  # df2/dx
            return 1 + torch.pi * 8 * x.pow(3) * torch.cos(torch.pi * 2 * x.pow(4))

        def target_der2(x):
            return 24*torch.pi*x.pow(2)*torch.cos(torch.pi * 2 * x.pow(4)) - (8*torch.pi*x.pow(3)).pow(2)*torch.sin(torch.pi * 2 * x.pow(4))

    elif func_type == 'f3':
        def target(x):
            return torch.exp(x) * torch.sin(20*x)

        def target_derivative(x):
            return torch.exp(x) * (torch.sin(20*x) + 20*torch.cos(20*x))

        def target_der2(x):
            return torch.exp(x) * (-399*torch.sin(20*x) + 40*torch.cos(20*x))

    else:
        raise ValueError(f"Unknown function type: {func_type}")

    return target, target_derivative, target_der2