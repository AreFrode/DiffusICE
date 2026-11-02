import torch

# import pytorch_lightning as L
import lightning.pytorch as L

from torch import nn
from torch.nn import functional as F
from torchvision.ops import stochastic_depth
from einops import rearrange

def construct_multiples(multiples, comparisson):
    if isinstance(multiples, int):
        multiples = len(comparisson) * [multiples]

    return multiples

class DownSample(nn.Module):
    def __init__(self, in_channels, out_channels, padding_mode = 'zeros'):
        super().__init__()
        self.norm = nn.GroupNorm(32, in_channels) if in_channels >= 32 else nn.GroupNorm(in_channels, in_channels)
        self.down_layer = nn.Conv2d(in_channels,
                                    out_channels,
                                    kernel_size=2,
                                    stride=2,
                                    padding_mode=padding_mode)

    def forward(self, x):
        x = self.norm(x)
        return self.down_layer(x)


class UpSample(nn.Module):
    def __init__(self, in_channels, out_channels, padding_mode = 'zeros'):
        super().__init__()
        self.norm = nn.GroupNorm(32, in_channels) if in_channels >= 32 else nn.GroupNorm(in_channels, in_channels)
        self.upsampling = nn.Upsample(scale_factor = 2, mode = 'bilinear', align_corners = False)
        self.conv = nn.Conv2d(in_channels + out_channels, out_channels, kernel_size=3, padding=1, padding_mode=padding_mode)

    def forward(self, x, x_skip):
        x = self.norm(x)
        x = self.upsampling(x)
        x = torch.cat([x_skip, x], dim = 1)
        return self.conv(x)

class ConvNeXtBlock(nn.Module):
    def __init__(self, in_channels, widening_factor = 4):
        super().__init__()
        self.hidden_channels_bottleneck = int(widening_factor * in_channels)

        self.depthwise_convolution = nn.Conv2d(
            in_channels,
            in_channels,
            kernel_size=7,
            groups=in_channels,
            padding='same')
        self.layer_norm = nn.GroupNorm(1, in_channels)

        self.expanding_path = nn.Linear(in_channels, self.hidden_channels_bottleneck)

        self.act = nn.GELU()

        self.contracting_path = nn.Linear(self.hidden_channels_bottleneck, in_channels)

    def forward(self, x):
        b, c, height, width = x.shape
        h = self.depthwise_convolution(x)
        h = self.layer_norm(h)
        h = rearrange(h,
                      "b c h w -> b (h w) c",
                      b=b, c=c, h=height, w=width)

        h = self.expanding_path(h)
        h = self.act(h)
        h = self.contracting_path(h)

        return x + rearrange(h,
                             "b (h w) c -> b c h w",
                             b=b, c=c, h=height, w=width)

class ResNetBlock(nn.Module):
    def __init__(self, in_channels, drop_path_rate = 0., padding_mode = 'zeros'):
        super().__init__()
        self.drop_path_rate = drop_path_rate
        self.first_block = nn.Sequential(
            nn.GroupNorm(32, in_channels) if in_channels >= 32 else nn.GroupNorm(in_channels, in_channels),
            nn.SiLU(),
            nn.Conv2d(in_channels = in_channels,
                               out_channels = in_channels,
                               kernel_size = 3,
                               padding = 1,
                               padding_mode = padding_mode,
                               bias = False)
        )

        self.norm = nn.GroupNorm(32, in_channels) if in_channels >= 32 else nn.GroupNorm(in_channels, in_channels)
        self.act = nn.SiLU()
        self.conv = nn.Conv2d(in_channels = in_channels,
                      out_channels = in_channels,
                      kernel_size = 3,
                      padding = 1,
                      padding_mode = padding_mode,
                      bias = False)

        # Initialize second conv as zeros, start out as identity mapping
        nn.init.zeros_(self.conv.weight)

    def forward(self, x):
        h = self.first_block(x)
        h = self.norm(h)
        h = self.act(h)
        h = self.conv(h)
        return x + stochastic_depth(
            h,
            p = self.drop_path_rate,
            mode = 'batch',
            training = self.training
        )


class Encoder(nn.Module):
    def __init__(self, channels, multiples, drop_path_rates, padding_mode = 'zeros'):
        super().__init__()
        self.encoder_blocks = nn.ModuleList()

        block_idx = 0
        for in_channels, num_blocks in zip(channels[:-1], multiples):
            self.encoder_blocks.append(nn.Module())
            self.encoder_blocks[-1].blocks = nn.ModuleList([
                ResNetBlock(in_channels, drop_path_rate=drop_path_rates[block_idx + i], padding_mode=padding_mode) for i in range(num_blocks)
            ])
            block_idx += num_blocks


        self.downsamples = nn.ModuleList([
            DownSample(in_channels=channels[i],
                       out_channels=channels[i+1]
                       ) for i in range(len(channels) - 1)
        ])

    def forward(self, x):
        feature_maps = []
        for conv, downsample in zip(self.encoder_blocks, self.downsamples):
            for block in conv.blocks:
                x = block(x)

            feature_maps.append(x)
            x = downsample(x)

        return x, feature_maps

class ConvBottleneck(nn.Module):
    def __init__(self, in_channels, n_blocks, drop_path_rates):
        super().__init__()
        self.blocks = nn.ModuleList([
            ResNetBlock(in_channels, drop_path_rate = drop_path_rates[i])
            for i in range(n_blocks)
        ])

    def forward(self, x):
        for block in self.blocks:
            x = block(x)
        return x


class Decoder(nn.Module):
    def __init__(self, channels, multiples, drop_path_rates, padding_mode = 'zeros'):
        super().__init__()
        self.upsamples = nn.ModuleList([
            UpSample(channels[i], channels[i+1], padding_mode=padding_mode) for i in range(len(channels) - 1)
        ])

        self.decoder_blocks = nn.ModuleList()
        block_idx = 0
        for in_channels, num_blocks in zip(channels[1:], multiples):
            self.decoder_blocks.append(nn.Module())
            self.decoder_blocks[-1].blocks = nn.ModuleList([
                ResNetBlock(in_channels, drop_path_rate=drop_path_rates[block_idx + i], padding_mode=padding_mode) for i in range(num_blocks)
            ])
            block_idx += num_blocks


    def forward(self, x, encoder_features):
        for upsample, skip_connection, conv in zip(self.upsamples, encoder_features, self.decoder_blocks):
            x = upsample(x, skip_connection)

            for block in conv.blocks:
                x = block(x)

        return x

class UNET(nn.Module):
    def __init__(self, in_fields, channels, encoder_multiples, decoder_multiples, n_bottleneck, max_drop_path_rate = 0.2):
        super().__init__()
        self.in_fields = in_fields
        self.n_bottleneck = n_bottleneck
        self.EmbedInput = nn.Conv2d(in_fields, channels[0], 1)

        # Build Encoder/decoder
        encoder_multiples = construct_multiples(encoder_multiples, channels[:-1])
        decoder_multiples = construct_multiples(decoder_multiples, channels[:-1])

        # Calculate total number of blocks
        total_encoder_blocks = sum(encoder_multiples)
        total_decoder_blocks = sum(decoder_multiples)
        total_blocks = total_encoder_blocks + n_bottleneck + total_decoder_blocks

        # Create linearly increasing drop path rates
        drop_path_rates = [i * max_drop_path_rate / (total_blocks - 1) for i in range(total_blocks)]

        # Split rates for Encoder, Bottleneck and Decoder
        encoder_rates = drop_path_rates[:total_encoder_blocks]
        bottleneck_rates = drop_path_rates[total_encoder_blocks:total_encoder_blocks+n_bottleneck]
        decoder_rates = drop_path_rates[total_encoder_blocks+n_bottleneck:]

        self.encoder = Encoder(channels, encoder_multiples, encoder_rates)

        if n_bottleneck > 0:
            self.bottleneck = ConvBottleneck(channels[-1], n_bottleneck, bottleneck_rates)
        else:
            self.bottleneck = nn.Identity()

        self.decoder = Decoder(channels[::-1], decoder_multiples, decoder_rates)

        self.out_norm = nn.GroupNorm(1, channels[0])
        self.out_act = nn.ReLU()
        self.output_layer = nn.Conv2d(in_channels = channels[0], out_channels=1, kernel_size=1)

    def forward(self, x):
        x = self.EmbedInput(x)

        # print('Encoder')
        last_h, encoder_features = self.encoder(x)

        encoder_features_inverse = encoder_features[::-1]

        # print('transformer')
        bottleneck_out = self.bottleneck(last_h)

        # print('decoder')
        out = self.decoder(bottleneck_out, encoder_features_inverse)

        out = self.out_norm(out)

        out = self.out_act(out)

        return self.output_layer(out)

class LitUNET(L.LightningModule):
    def __init__(self, config, output_path = None):
        super().__init__()
        self.config = config

        self.model = UNET(
            in_fields = len(config['fields']),
            channels = config['channels'],
            encoder_multiples=config['encoder_block_multiples'],
            decoder_multiples=config['decoder_block_multiples'],
            n_bottleneck=config['num_bottleneck_blocks'],
            max_drop_path_rate=config['max_drop_path_rate'],
        )

        self.output_path = output_path
        self.learning_rate = config['learning_rate']
        self.warmup_epochs = config['warmup_epochs']
        self.epochs = config['epochs']
        self.weight_decay = config['weight_decay']
        self.min_lr = config['min_lr']

    def forward(self, x):
        pred = self.model(x)
        return pred

    def configure_optimizers(self):
        optimizer = torch.optim.AdamW(self.parameters(),
                                      lr = self.learning_rate,
                                      weight_decay = self.weight_decay
        )

        # Warmup scheduler: Linear increase
        warmup_scheduler = torch.optim.lr_scheduler.LinearLR(
            optimizer,
            start_factor=0.01,
            end_factor = 1.0,
            total_iters=self.warmup_epochs
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
            milestones=[self.warmup_epochs]
        )

        # scheduler = torch.optim.lr_scheduler.ExponentialLR(optimizer, gamma = 0.97)
        # scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max = 10)
        lr_scheduler_config = {
            "scheduler": scheduler,
            "interval": "epoch",    # Update per epoch
            "frequency": 1,
            "name": "CosineWarmup"
        }

        return {"optimizer": optimizer, "lr_scheduler": lr_scheduler_config}

    def training_step(self, train_batch):
        _, x, y = train_batch

        x = x[:, 0]

        y_pred = self.model(x)
        loss = F.mse_loss(y_pred, y)

        # log learning rate
        current_lr = self.trainer.optimizers[0].param_groups[0]['lr']
        self.log('lr', current_lr, logger=True, prog_bar=True, sync_dist = True)
        self.log('train_loss', loss, logger=True, prog_bar=True, sync_dist = True)

        return loss

    def validation_step(self, val_batch):
        _, x, y = val_batch

        x = x[:, 0]

        y_pred = self.model(x)
        loss = F.mse_loss(y_pred, y)
        self.log('val_loss', loss, logger = True, prog_bar=True, sync_dist = True)

    def predict_step(self, batch):
        sample_date, x, y = batch
        y_pred = torch.clip(self.model(x), min = 0, max = 1)

        x = x.cpu().detach().numpy()
        y = y.cpu().detach().numpy()
        y_pred = y_pred.float().cpu().detach().numpy()
        sample_date = str(sample_date[0].item())

    def predict_single_timestep(self,
                                x):

        return self.model(x.unsqueeze(0))