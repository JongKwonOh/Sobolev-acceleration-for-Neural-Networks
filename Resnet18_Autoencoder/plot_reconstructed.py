import torch
import matplotlib.pyplot as plt
import numpy as np
from scripts.data_loading import get_imagenet_data_loaders
from scripts import utils 
from classes.resnet_autoencoder import AE
import argparse
import os
from tqdm import tqdm

parser = argparse.ArgumentParser()
parser.add_argument('--gpu', default=0)
args = parser.parse_args()

if torch.cuda.is_available():
    device = f'cuda:{args.gpu}' 

download_path = "your_data_path"
train_loader, test_loader, train_dataset, test_dataset = get_imagenet_data_loaders(download_path=download_path, batch_size=1)

l2_AE = AE('light')
l2_load_path = "your_model_path"
l2_AE, _, _ = utils.resume(l2_AE, l2_load_path)

h1_AE = AE('light')
h1_load_path = "your_model_path"
h1_AE, _, _ = utils.resume(h1_AE, h1_load_path)

l2_AE.to(device).eval()
h1_AE.to(device).eval()

rand_idx = np.random.randint(len(test_dataset))
img, _ = test_dataset[rand_idx]
img_input = img.unsqueeze(0).to(device)

with torch.no_grad():
    l2_rec_img, _ = l2_AE(img_input)
    h1_rec_img, _ = h1_AE(img_input)

img = img.cpu().permute(1, 2, 0).numpy()
l2_rec_img = l2_rec_img.cpu().squeeze().permute(1, 2, 0).numpy()
h1_rec_img = h1_rec_img.cpu().squeeze().permute(1, 2, 0).numpy()

fig, axs = plt.subplots(1, 3, figsize=(15, 5))

axs[0].imshow(img)
axs[0].set_title('Original Image', fontsize=21)
axs[0].axis("off")

axs[1].imshow(l2_rec_img)
axs[1].set_title('$L_2$ Reconstructed', fontsize=21)
axs[1].axis("off")

axs[2].imshow(h1_rec_img)
axs[2].set_title('$H^1$ Reconstructed', fontsize=21)
axs[2].axis("off")

save_folder = "comparison_single"
os.makedirs(save_folder, exist_ok=True)
save_path = os.path.join(save_folder, f"rand_{rand_idx}_comparison.png")
plt.savefig(save_path, bbox_inches="tight")
plt.show()

print(f"✅ Saved comparison at '{save_path}'")