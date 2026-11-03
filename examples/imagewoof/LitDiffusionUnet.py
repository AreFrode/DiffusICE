import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))

import torch
import lightning.pytorch as L

from torch.nn import functional as F
from diffusice.samplers import HeunStochasticSampler
from diffusice.backbone.unet_backbone_sde import UNETbackbone
from diffusice.edm import sample_sigma, precondition, loss_weight, denoise

class LitDiffusionEDM(L.LightningModule):
    def __init__(self, config):
        super().__init__()
        self.config = config

        self.backbone = UNETbackbone(
            in_fields = 3,
            channels = config['channels'],
            n_embedding = config['embedding_dim'],
            n_time_hidden = config['embedding_hidden_dim'],
            encoder_multiples = config['encoder_block_multiples'],
            decoder_multiples = config['decoder_block_multiples'],
            n_bottleneck = config['num_bottleneck_blocks'],
            out_fields = 3,
            max_drop_path_rate = config['max_drop_path_rate'],
            padding_mode = config['padding_mode']
        )

        self.learning_rate = config['learning_rate']
        self.warmup_epochs = config['warmup_epochs']
        self.epochs = config['epochs']
        self.weight_decay = config['weight_decay']
        self.min_lr = config['min_lr']
        self.P_mean = -1.2
        self.P_std = 1.2

        self.sigma_data = 0.5
        self.ema = torch.optim.swa_utils.AveragedModel(self.backbone, multi_avg_fn=torch.optim.swa_utils.get_ema_multi_avg_fn(0.9999))

    def configure_optimizers(self):
        optimizer = torch.optim.AdamW(self.parameters(),
                                      lr = self.learning_rate,
                                      weight_decay = self.weight_decay
        )

        # Warmup scheduler: Linear increase
        warmup_scheduler = torch.optim.lr_scheduler.LinearLR(
            optimizer,
            start_factor = 0.01,
            end_factor = 1.0,
            total_iters = self.warmup_epochs
        )

        # Cosine annealing scheduler
        cosine_scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            optimizer,
            T_max = self.epochs - self.warmup_epochs,
            eta_min = self.min_lr
        )

        # Combine schedulers
        scheduler = torch.optim.lr_scheduler.SequentialLR(
            optimizer,
            schedulers = [warmup_scheduler, cosine_scheduler],
            milestones = [self.warmup_epochs]
        )

        lr_scheduler_config = {
            "scheduler": scheduler,
            "interval": "epoch",
            "frequency": 1,
            "name": "CosineWarmup"
        }

        return {"optimizer": optimizer, "lr_scheduler": lr_scheduler_config}

    def training_step(self, train_batch):
        y, _ = train_batch

        sigma, lnsigma = sample_sigma(self.P_mean, self.P_std, y.size(0), y.device)
        c_skip, c_out, c_in, c_noise = precondition(sigma, lnsigma, self.sigma_data)

        noisy_y = y + torch.randn_like(y) * sigma.view(-1, *([1]*(y.ndim - 1)))
        y_in = c_in.view(-1, *([1]*(y.ndim - 1))) * noisy_y

        D = denoise(self.backbone, y_in, noisy_y, c_noise, c_skip, c_out, y.ndim - 1)
        loss = (loss_weight(sigma, self.sigma_data, y.ndim - 1) * F.mse_loss(D, y, reduction = 'none')).mean()

        self.log("train_loss", loss, prog_bar = True, sync_dist = True)

        return loss

    def on_train_batch_end(self, outputs, batch, batch_idx):
        self.ema.update_parameters(self.backbone)


    def _make_denoiser(self):
        def denoiser_fn(z, t):
            c_skip, c_out, c_in, c_noise = precondition(t, torch.log(t).unsqueeze(0), self.sigma_data)

            z_in = c_in.view(-1, *([1]*(z.ndim-1))) * z
            D = denoise(self.ema, z_in, z, c_noise, c_skip, c_out, z.ndim - 1)

            return D

        return denoiser_fn
    

    def generate_samples(self, batch_size, device, n_inference_steps = 40):  
        sampler = HeunStochasticSampler()

        denoiser_fn = self._make_denoiser()

        time_steps, gamma = sampler.setup_schedule(n_inference_steps, device)

        z = torch.randn(batch_size, 3, 160, 160, device = device) * time_steps[0]

        return sampler.step(z, denoiser_fn, time_steps, gamma, n_inference_steps)
    
