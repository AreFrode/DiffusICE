# DiffusICE
A work in project implementation of Score SDE Diffusion following the Karras EDM framework, intended for use internally at the Norwegian Meteorological Institute. The software is provided as is.

# Model architecture
DiffusICE is provided as a set of diffusion model backbones, and wrappers that perform basic EDM functionality (noise sampling, preconditioning, denoising) and Inference sampling (Heuns stochastic 2nd order). The provided functions are intended  for constructing [Pytorch Lightning](https://lightning.ai/docs/pytorch/stable/index.html) modules, but this is not strictly required. Example LightningModules are provided in the deterministic variants of the underlying architectures. A Diffusion example is provided in the **examples/** directory.

## diffusice
*edm.py* contains functionality for preconditioning, sigma sampling, loss weighting and denoising, required by diffusion model training and sampling.

*samplers.py* contains an implementation of [Heuns 2nd order stochastic sampler](https://arxiv.org/abs/2206.00364), written for a compatible LightningModule function.

### backbone
*unet.py* is a basic [UNet](https://arxiv.org/abs/1505.04597), broken down into basic blocks performing DownSampling, UpSampling, ResNet. 

*unet_backbone_sde.py* extends upon *unet.py* with the necesarry time-embedding steps required for diffusion modelling. 

*uvit.py* is a basic [UVit](https://arxiv.org/abs/2301.11093), extending upon *unet.py* with Vision transformer functionality. 

*uvit_backbone_sde.py* extends upon *uvit.py* with the necessary time-embedding steps required for diffusion modelling.

## examples
The examples directory contains an example where a simple dataloader is created, *uvit_backbone_sde.py* is used to construct a simple diffusion model as a LightningModule, a unconditional diffusion model is trained and subsequently used to generate new images. The example at present is based on the [ImageWoof](https://github.com/fastai/imagenette) ImageNet subset, containing images of 10 labelled dog-breeds.