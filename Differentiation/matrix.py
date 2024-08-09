import torch

def FDM_matrix(X, device):
    N = X.size(0) - 2

    h = torch.tensor([[X[i] - X[i-2]]*(N+2) for i in range(2, N+2)]).to(device)

    if N <= 0 : return 0
    
    out = torch.zeros((N, N+2)).to(device)
    
    for i in range(N):
        out[i] = torch.cat((torch.zeros(i), torch.tensor([-1, 0, 1]), torch.zeros(N - 1 - i)), dim=0)
        
    return out/h
    
def cheb_matrix(f, device):
    N = f.size(0) - 1
    x = torch.cos(torch.pi * torch.arange(0, N+1) / N).to(device)
    out = torch.zeros((N+1, N+1)).to(device)
    middle_range = torch.arange(1, N).to(device)

    # i == 0 cases
    out[0, 0] = (2 * N**2 + 1) / 6
    out[0, 1:N] = 2 * (-1)**middle_range / (1 - x[1:N])
    out[0, N] = 0.5 * (-1)**N

    # 0 < i < N cases
    out[middle_range, 0] = -0.5 * (-1)**middle_range / (1 - x[middle_range])
    out[middle_range, middle_range] = -x[middle_range] / (2 * (1 - x[middle_range]**2))
    out[middle_range, N] = 0.5 * (-1)**(N + middle_range) / (1 + x[middle_range])

    # i == N cases
    out[N, 0] = -0.5 * (-1)**N
    out[N, 1:N] = -2 * (-1)**(N + middle_range) / (1 + x[1:N])
    out[N, N] = -(2 * N**2 + 1) / 6

    # Other cases
    I, J = torch.meshgrid(middle_range, middle_range)
    mask = I != J
    out[I[mask], J[mask]] = (-1)**(I + J)[mask] / (x[I] - x[J])[mask]

    return out
