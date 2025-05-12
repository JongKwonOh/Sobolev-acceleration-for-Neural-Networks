import os
import yaml
import torch
import matplotlib.pyplot as plt
from denoising_diffusion_pytorch import Unet, GaussianDiffusion
from utils import *

with open("celeba_config.yaml", "r") as f:
    config = yaml.safe_load(f)

model = Unet(
    dim=config['model']['hidden_dim'],
    dim_mults=(1, 2, 4),
    channels=3,
).to(device)

diffusion_model = GaussianDiffusion(
    model,
    image_size=config['data']['image_size'],
    objective=config['model']['objective'],
    timesteps=config['model']['timesteps'],
    sampling_timesteps=config['model']['sampling_timesteps'],
    loss_type=config['model']['loss_type']
).to(device)

checkpoint_path = f"./{config['folder_path']}/{config['model']['loss_type']}_diffusion_model_and_metrics_last.pt"
checkpoint = torch.load(checkpoint_path, map_location=device)
model.load_state_dict(checkpoint['model_state_dict'])
model.eval()

num_samples = 8
ncols = 4

with torch.no_grad():
    sampled_images = diffusion_model.sample(batch_size=num_samples)
    sampled_images = sampled_images.clamp(0, 1).cpu()

os.makedirs('./save', exist_ok=True)

plot_images(sampled_images, ncols=ncols, save_name=f'{config["model"]["loss_type"]}_sampling.png')
