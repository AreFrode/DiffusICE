import torch

# import pytorch_lightning as L
import lightning.pytorch as L

from torch import nn
from torch.nn import functional as F
from torchvision.ops import stochastic_depth
from einops import rearrange

from .unet import ResNetBlock, DownSample, Encoder, Decoder, UpSample, construct_multiples


class MLPBlock(nn.Module):
    def __init__(self, in_channels, expansion_factor = 4, dropout_rate = .1, drop_path_rate = 0.):
        super().__init__()
        self.drop_path_rate = drop_path_rate
        self.hidden_channels = in_channels * expansion_factor
        self.norm = nn.LayerNorm(in_channels)
        self.mlp = nn.Sequential(
            nn.Linear(in_channels, self.hidden_channels),
            nn.SiLU(),
            nn.Dropout(p = dropout_rate),
            nn.Linear(self.hidden_channels, in_channels)
        )

    def forward(self, x):
        h = self.norm(x)
        h = self.mlp(h)
        # Apply stochastic depth to the residual connection
        return x + stochastic_depth(
            h,
            p = self.drop_path_rate,
            mode = 'row',
            training = self.training
        )


class AttnetionBlock(nn.Module):
    def __init__(self, in_channels, n_heads = 8, dropout_rate = 0.1, drop_path_rate = 0.):
        super().__init__()
        self.drop_path_rate = drop_path_rate
        self.norm = nn.LayerNorm(in_channels)
        self.qkv = nn.Linear(in_channels, in_channels * 3)
        self.out_layer = nn.Linear(in_channels, in_channels)
        self.dropout = nn.Dropout(p = dropout_rate)

        self.n_heads = n_heads

    def forward(self, x):   # x: (B, L, C)
        h = self.norm(x)

        qkv = self.qkv(h)   # (B, L, 3*C)

        b, l, c3 = qkv.shape

        # Multi-Head Self-Attention
        head_dim = c3 // 3 // self.n_heads

        qkv = rearrange(qkv,
                        "b l (k h c) -> k b h l c",
                        b=b, l=l, h=self.n_heads, c=head_dim, k=3)

        attn_out = F.scaled_dot_product_attention(
            query=qkv[0], key=qkv[1], value=qkv[2]
        )

        attn_out = rearrange(attn_out,
                             "b h l c -> b l (h c)",
                             b=b, l=l, h=self.n_heads, c=head_dim)

        # Apply dropout then stochastic depth
        attn_out = self.out_layer(attn_out)
        attn_out = self.dropout(attn_out)

        return x + stochastic_depth(
            attn_out,
            p = self.drop_path_rate,
            mode = 'row',
            training = self.training
        )


class Transformer(nn.Module):
    """
    Note that, following U-ViT conventions, the input is not patched into NxN embeddings
    """
    def __init__(self, in_channels, n_bottleneck, latent_dims, drop_path_rates, attn_dropout = 0.1, mlp_dropout = 0.1):
        super().__init__()
        self.pos_emb = nn.Parameter(torch.empty(1, latent_dims[0] * latent_dims[1], in_channels))
        nn.init.normal_(self.pos_emb, mean = 0.0, std = 0.01)

        self.self_attention = nn.ModuleList([
            AttnetionBlock(in_channels, dropout_rate = attn_dropout, drop_path_rate=drop_path_rates[i]) for i in range(n_bottleneck)
        ])

        self.mlp_block = nn.ModuleList([
            MLPBlock(in_channels, dropout_rate = mlp_dropout, drop_path_rate=drop_path_rates[i]) for i in range(n_bottleneck)
        ])

    def forward(self, x):
        b, c, height, width = x.shape

        h = rearrange(x,
                      "b c h w -> b (h w) c",
                      b=b, c=c, h=height, w=width)

        h = h + self.pos_emb

        for self_attention, mlp in zip(self.self_attention, self.mlp_block):
            h = self_attention(h)
            h = mlp(h)
        return rearrange(h,
                         "b (h w) c -> b c h w",
                         b=b, c=c, h=height, w=width)
    

class UVIT(nn.Module):
    def __init__(self,
                 in_fields,
                 channels,
                 encoder_multiples,
                 decoder_multiples,
                 n_bottleneck,
                 latent_dims, 
                 max_drop_path_rate = 0.2,
                 attn_dropout = 0.1,
                 mlp_dropout = 0.1
        ):

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
            self.bottleneck = Transformer(channels[-1], n_bottleneck, latent_dims, bottleneck_rates, attn_dropout, mlp_dropout)
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


class MultiResUVIT(nn.Module):
    """Specific implementation of a U-ViT with two sequential input channels
    """
    def __init__(self,
                 in_fields_lr,
                 in_fields_hr,
                 channels,
                 hr_channels,
                 encoder_multiples,
                 decoder_multiples,
                 n_bottleneck,
                 latent_dims,
                 out_fields_lr,
                 out_fields_hr,
                 max_drop_path_rate=0.2,
                 attn_dropout = 0.1,
                 mlp_dropout = 0.1,
                 padding_mode = 'reflect'
    ):
        super().__init__()

        enc_mult = construct_multiples(encoder_multiples, channels[:-1])
        dec_mult = construct_multiples(decoder_multiples, channels[:-1])
        hr_enc_mult = construct_multiples(decoder_multiples, [hr_channels])
        hr_dec_mult = construct_multiples(decoder_multiples, [hr_channels])

        total = sum(hr_enc_mult) + sum(enc_mult) + n_bottleneck + sum(dec_mult) + sum(hr_dec_mult)
        drop_path_rates = [i * max_drop_path_rate / max(total - 1, 1) for i in range(total)]

        hr_enc_dpr = drop_path_rates[:sum(hr_enc_mult)]
        enc_dpr = drop_path_rates[sum(hr_enc_mult):sum(hr_enc_mult)+sum(enc_mult)]
        bn_dpr = drop_path_rates[sum(hr_enc_mult)+sum(enc_mult):sum(hr_enc_mult)+sum(enc_mult)+n_bottleneck]
        dec_dpr = drop_path_rates[sum(hr_enc_mult)+sum(enc_mult)+n_bottleneck:sum(hr_enc_mult)+sum(enc_mult)+n_bottleneck+sum(dec_mult)]
        hr_dec_dpr = drop_path_rates[sum(hr_enc_mult)+sum(enc_mult)+n_bottleneck+sum(dec_mult):]

        # HR Entry stage
        self.hr_input_proj = nn.Conv2d(in_fields_hr, hr_channels, 1)
        self.hr_encoder_blocks = nn.ModuleList([
            ResNetBlock(hr_channels,
                        drop_path_rate=hr_enc_dpr[i],
                        padding_mode=padding_mode)
            for i in range(sum(hr_enc_mult))
        ])
        self.hr_downsample = DownSample(hr_channels, hr_channels)

        # Main Path, lr entry point
        self.lr_input_proj = nn.Conv2d(in_fields_lr, hr_channels, 1)
        self.EmbedInput = nn.Conv2d(2 * hr_channels, channels[0], 1)

        self.encoder = Encoder(
            channels,
            enc_mult,
            enc_dpr,
            padding_mode=padding_mode
        )

        if n_bottleneck > 0:
            self.bottleneck = Transformer(
                channels[-1],
                n_bottleneck,
                latent_dims,
                bn_dpr,
                attn_dropout,
                mlp_dropout
            )

        else:
            self.bottleneck = nn.Identity()

        self.decoder = Decoder(
            channels[::-1],
            dec_mult,
            dec_dpr,
            padding_mode=padding_mode
        )

        self.out_norm = nn.GroupNorm(1, channels[0])
        self.out_act = nn.ReLU()

        # LR output
        self.lr_output_layer = nn.Conv2d(channels[0], out_fields_lr, 1)

        self.hr_upsample = UpSample(channels[0], hr_channels, padding_mode)
        self.hr_decoder_blocks = nn.ModuleList([
            ResNetBlock(hr_channels,
                        drop_path_rate=hr_dec_dpr[i],
                        padding_mode=padding_mode)
            for i in range(sum(hr_dec_mult))
        ])

        self.hr_out_norm = nn.GroupNorm(1, hr_channels)
        self.hr_out_act = nn.ReLU()
        self.hr_output_layer = nn.Conv2d(hr_channels, out_fields_hr, 1)

    def forward(self, x_lr, x_hr):
        # HR encoder
        hr = self.hr_input_proj(x_hr)
        for block in self.hr_encoder_blocks:
            hr = block(hr)

        hr_skip = hr
        hr_down = self.hr_downsample(hr)

        # Main network
        lr = self.lr_input_proj(x_lr)
        x = self.EmbedInput(torch.cat([lr, hr_down], dim = 1))

        last_h, encoder_features = self.encoder(x)
        bottleneck_out = self.bottleneck(last_h)
        dec_out = self.decoder(bottleneck_out, encoder_features[::-1])

        out_lr = self.lr_output_layer(self.out_act(self.out_norm(dec_out)))

        # HR decoder
        hr_out = self.hr_upsample(dec_out, hr_skip)
        for block in self.hr_decoder_blocks:
            hr_out = block(hr_out)

        out_hr = self.hr_output_layer(self.hr_out_act(self.hr_out_norm(hr_out)))

        return out_lr, out_hr


class LitUVIT(L.LightningModule):
    def __init__(self, config, output_path = None):
        super().__init__()
        self.config = config

        self.model = UVIT(
            in_fields = len(config['fields']),
            channels = config['channels'],
            encoder_multiples=config['encoder_block_multiples'],
            decoder_multiples=config['decoder_block_multiples'],
            n_bottleneck=config['num_transformer_blocks'],
            latent_dims=[config['latent_y_dim'], config['latent_x_dim']],
            max_drop_path_rate=config['max_drop_path_rate'],
            attn_dropout=config['attn_dropout'],
            mlp_dropout=config['mlp_dropout']
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
