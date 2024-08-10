import torch
import matplotlib.pyplot as plt
from matplotlib.pyplot import cm
import yaml
from data_loader import get_test_data
import numpy as np


def make_freq_noisy(data, amp=0.3, noise_amplitude=2) :
    freq=noise_amplitude*np.pi
    x = torch.linspace(0,1, data.size()[-1])
    y = torch.linspace(0,1, data.size()[-1])
    xx,yy = torch.meshgrid(x,y)
    return data + (amp*torch.sin(freq*xx+freq*yy)).to(data.device)

def main():
    with open('test_config.yaml', 'r') as file:
        config = yaml.safe_load(file)
        
    test_idx = config['test_idx']
    s = config['s']
    load_noise_amplitude = config['load_noise_amplitude']
    save_noise_amplitude = config['save_noise_amplitude']
    
    load_path_template = config['load_path']
    save_path_template = config['save_path']
    
    load_path = load_path_template.format(s, load_noise_amplitude)
    save_path = save_path_template.format(test_idx, s, save_noise_amplitude)
    
    device = torch.device(f"cuda:{config['device_num']}" if torch.cuda.is_available() else "cpu")
    
    test_data = get_test_data()
    errs, model = torch.load(load_path, map_location=device)

    # Plot
    fig = plt.figure(figsize=(13,3))
    sample = test_data[test_idx][0].to(device)
    noisy = make_freq_noisy(sample, noise_amplitude=save_noise_amplitude)
    output = model(noisy.unsqueeze(0).to(device))

    gs = fig.add_gridspec(1,3)

    ax1 = fig.add_subplot(gs[0, 0])
    ax1.imshow(sample.squeeze().cpu().detach().numpy(), cmap='gray', aspect="auto")
    ax1.set_title('original')

    ax1 = fig.add_subplot(gs[0, 1])
    ax1.imshow(noisy.squeeze().cpu().detach().numpy(), cmap='gray', aspect="auto")
    ax1.set_title('noisy')
    
    ax3 = fig.add_subplot(gs[0, 2])
    ax3.imshow(output.squeeze().cpu().detach().numpy(), cmap='gray', aspect="auto")
    ax3.set_title('Reconstructed')

    plt.tight_layout()
    
    plt.savefig(save_path)

    return

if __name__ == "__main__":
    main()