import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
import torchvision
from torchvision import datasets, transforms
import matplotlib.pyplot as plt
import numpy as np
import random
import tqdm
import os
from utils import *
from PIL import Image
from torchvision.datasets.folder import default_loader
from torch.utils.data import Dataset

from denoising_diffusion_pytorch import Unet, GaussianDiffusion
# from torchmetrics.image.inception import InceptionScore
# from torchmetrics.image.fid import FrechetInceptionDistance
from torch_fidelity import calculate_metrics

import yaml

with open("celeba_config.yaml", "r") as f:
    config = yaml.safe_load(f)

if not os.path.exists(config['folder_path']):
    os.makedirs(config['folder_path'])
    print(f"Created folder: {config['folder_path']}")
else:
    print(f"Folder exists: {config['folder_path']}")

loss_type=config['loss_type']
save_path = f"./{config['folder_path']}/{loss_type}_diffusion_model_and_metrics"
# device = torch.device(f"cuda:{config['device_num']}" if torch.cuda.is_available() else "cpu")
if not isinstance(config['device_num'], str):
    config['device_num'] = str(config['device_num'])

os.environ['CUDA_VISIBLE_DEVICES'] = config['device_num']

BATCH_SIZE = config['data']['BATCH_SIZE']
WORKERS = 8
img_size = config['data']['image_size']
transform = transforms.Compose([
    transforms.Resize((img_size, img_size)),
    transforms.ToTensor(),
])

print('Loading data...')
download_path = config['data']['download_path']

class CelebAHQDataset(Dataset):
    def __init__(self, root_dir, transform=None):
        self.root_dir = root_dir
        self.transform = transform
        self.image_files = sorted([
            file for file in os.listdir(root_dir)
            if file.endswith(('.jpg', '.png'))
        ])

    def __len__(self):
        return len(self.image_files)

    def __getitem__(self, idx):
        img_path = os.path.join(self.root_dir, self.image_files[idx])
        image = Image.open(img_path).convert("RGB")
        if self.transform:
            image = self.transform(image)
        return image

class WrappedModel(nn.Module):
    def __init__(self, model):
        super().__init__()
        self.model = model.cuda()
        # DataParallel이면 module을, 아니면 그대로
        self.channels = model.module.channels if hasattr(model, 'module') else model.channels
        self.out_dim = model.module.out_dim if hasattr(model, 'module') else model.out_dim

    def forward(self, *args, **kwargs):
        return self.model(*args, **kwargs)

# 4. Dataset 및 DataLoader 구성
dataset = CelebAHQDataset(root_dir=download_path, transform=transform)
dataloader = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=True, num_workers=4)
for batch in dataloader:
    print("Batch shape:", batch.shape)  # [128, 3, 256, 256]
    break

raw_model = Unet(
    dim=config['model']['hidden_dim'],                   
    dim_mults=(1, 2, 4),     
    channels=3,
).cuda()

n_gpus = len(os.environ['CUDA_VISIBLE_DEVICES'].split(','))
if n_gpus > 1:
    model = nn.parallel.DataParallel(raw_model)
    model = WrappedModel(model)
else:
    model=raw_model

model = model.cuda()

# Gaussian Diffusion 모듈 생성
diffusion = GaussianDiffusion(
    raw_model,
    image_size=img_size,  
    objective = config['model']['objective'],
    timesteps=config['model']['timesteps'],
    sampling_timesteps = config['model']['sampling_timesteps'],
    loss_type=config['loss_type']
).cuda()

optimizer = torch.optim.Adam(model.parameters(), lr=config['lr'])

# test_idx = torch.arange(BATCH_SIZE, device=device)
# test_images = torch.stack([test_dataset[i][0] for i in test_idx]).to(device)

epochs = config['epochs'] 
val_every = config['metrics']['val_every']# 1
val_sample_count = config['metrics']['val_sample_count'] # 1000 
val_batch = config['metrics']['val_batch']# 100 

gen_eval_path = os.path.join(config['data']['evaluation_path'], config['loss_type']) # "./generated_eval"
real_path = config['data']['download_path'] # '../data/celeba_hq_128'
# os.makedirs(gen_eval_path, exist_ok=True)
os.makedirs(gen_eval_path,exist_ok=True)
FID_list = []

for epoch in tqdm.tqdm_notebook(range(epochs)):
    for step, images in enumerate(dataloader):
        model.train()
        images = images.cuda()#.to(device)
        loss = diffusion(images)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        torch.cuda.empty_cache()

    print(f"Epoch: {epoch}  |  Loss: {loss.item():.4f}")

    if epoch % val_every == 0:
        print(f"[Epoch {epoch}] Calculating FID...")

        model.eval()
        with torch.no_grad():
            for f in os.listdir(gen_eval_path):
                os.remove(os.path.join(gen_eval_path, f))

            num_collected = 0
            global_id = 0
            while num_collected < val_sample_count:
                batch_size = min(val_batch, val_sample_count - num_collected)
                sampled = diffusion.sample(batch_size=batch_size).clamp(0, 1).cpu()

                save_generated_images(sampled, gen_eval_path, start_idx=global_id, prefix='eval')
                global_id += batch_size
                num_collected += batch_size

        # FID 계산
        metrics = calculate_metrics(
            input1=gen_eval_path,
            input2=real_path,
            fid=True,
            batch_size=64,
            cache=True,
            device='cuda' if torch.cuda.is_available() else 'cpu'
        )
        fid_score = metrics['frechet_inception_distance']
        print(f"[Epoch {epoch}] FID: {fid_score:.4f}")
        FID_list.append(fid_score)

    # ------------------ 모델 저장 ------------------
    if epoch % config['metrics']['save_every'] == 0:
        torch.save({
            'model_state_dict': model.state_dict(),
            'FID_list': FID_list
        }, save_path + f"{epoch}.pt")

# 마지막 모델 저장
save_path = f"./{config['folder_path']}/{loss_type}_diffusion_model_and_metrics_last.pt"
torch.save({
    'model_state_dict': model.state_dict(),
    'FID_list': FID_list
}, save_path)