# Sobolev acceleration for Neural Networks

Official code for the paper "Sobolev acceleration for neural networks" submitted to NeurIPS 2025.

## Architecture
You can reproduce the results in Figure 6 of the manuscript by run the following command.<br>
<br>
`python fusion_optim_train.py`<br>

## Denoising Autoencoder

You can reproduce the results in Figure 7 of the manuscript by run the following command.<br>
`python train.py` command.<br>
The saved model and results can be ploted using the `python test.py` command.

## Differentiation

You can reproduce the results in Figure 9 of the manuscript by run the following command.<br>
<br>
`python train.py`<br>
You can change arguments in config.yaml.

## Resnet18 Autoencoder

To reproduce the results, please first download the Mini-ImageNet dataset from the following link:

[Mini-ImageNet on Hugging Face](https://huggingface.co/datasets/timm/mini-imagenet)

After downloading, make sure the dataset is placed in the appropriate directory as expected by the code (you can configure this path in `scripts/config.py`).

This code is built on the [ResNet-18 Autoencoder](https://github.com/eleannavali/resnet-18-autoencoder) implementation.  
We adapted and extended the original repository to support both $L_2$ and $H^1$ training for qualitative comparison in Figure 9 of the manuscript.

You can reproduce the results shown in Figure 9 of the manuscript by running the following command:

<br>
`python main.py`<br>

The reconstructed images and the validation loss graph can be plotted using the following commands, respectively:

<br>
`python plot_reconstructed.py`
`python plot_val_loss.py`<br>

## Diffusion Models

### Dataset Preparation
Download the CelebA-HQ (256x256) dataset from the following link:

[https://www.kaggle.com/datasets/badasstechie/celebahq-resized-256x256](https://www.kaggle.com/datasets/badasstechie/celebahq-resized-256x256)

Ensure the dataset is placed in the appropriate directory, as expected by the training script.

### Code Base
This implementation is based on the [DDPM (NeurIPS 2020)](https://arxiv.org/abs/2006.11239) paper and uses the following PyTorch package:

[https://github.com/lucidrains/denoising-diffusion-pytorch](https://github.com/lucidrains/denoising-diffusion-pytorch)

You can install the required package using:

<br>
pip install denoising-diffusion-pytorch<br>

### Modifications
We made the following modifications inside `denoising_diffusion_pytorch.py` to support $H^1$-based loss:

#### 1. Add `cheb_loss2` function
Add the `cheb_loss2` function inside your `utils.py` file.

#### 2. Modify `GaussianDiffusion` class

##### Modify the `__init__` method as follows:
```python
self.loss_type = loss_type
if 'H' in self.loss_type:
    self.power = float(self.loss_type.split('H')[-1])
    self.loss_func = cheb_loss2
else:
    self.loss_func = F.mse_loss
```    

##### Modify the `p_losses` method:
Replace the existing code:
```python
loss = F.mse_loss(model_out, target, reduction = 'none')
loss = reduce(loss, 'b ... -> b', 'mean')

loss = loss * extract(self.loss_weight, t, loss.shape)
return loss.mean()
```

With the following:
```python
if 'H' not in self.loss_type:
    loss = self.loss_func(model_out, target, reduction='none')
    loss = reduce(loss, 'b ... -> b', 'mean')

    loss = loss * extract(self.loss_weight, t, loss.shape)
    return loss.mean()
else:
    return self.loss_func(out=model_out - target, power=self.power)
```


### Configuration
You can modify the training arguments by editing the `celeba_config.yaml` file.

### Training
To start training, run the following command:

```bash
python celeba_train.py
```

### Sampling
To start sampling, run the following command:
```bash
python sampling_images.py
```