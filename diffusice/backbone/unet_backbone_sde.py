import torch

import math
from torch import nn
from torchvision.ops import stochastic_depth

from .unet import DownSample as UvitDownSample, UpSample as UvitUpSample, ResNetBlock as UvitResNetBlock, construct_multiples

class SinusoidalPositionalEmbedding(nn.Module):
    def __init__(self, embed_dim):
        super().__init__()
        div = torch.exp(torch.arange(0, embed_dim, 2).float() * -(math.log(10000.0) / embed_dim))
        self.register_buffer('div', div)

    def forward(self, timesteps):
        embedding = timesteps.unsqueeze(1) * self.div
        embedding = torch.cat([embedding.sin(), embedding.cos()], dim = -1)
        return embedding

class TimeEmbedding(nn.Module):
    def __init__(self, embedding_dim, n_time_hidden):
        super().__init__()
        self.sinusoidal = SinusoidalPositionalEmbedding(n_time_hidden)

        # Process time embedding with MLP
        self.mlp = nn.Sequential(
            nn.Linear(n_time_hidden, embedding_dim),
            nn.SiLU(),
            nn.Linear(embedding_dim, embedding_dim)
        )

    def forward(self, t):
        emb = self.sinusoidal(t)
        return self.mlp(emb)

class DownSample(UvitDownSample):
    """Downsample with AdaGN time conditioning"""
    def __init__(self, in_channels, out_channels, n_embedding, padding_mode='reflect'):

        super().__init__(in_channels, out_channels, padding_mode)

        # Time conditioning
        self.time_mlp = nn.Sequential(
            nn.SiLU(),
            nn.Linear(n_embedding, 2 * in_channels)
        )

    def forward(self, x, embedding):
        # x: (B, C, H, w)
        # (time_)embedding: (B, D)
        # Apply time conditioning before downsampling

        scale, shift = self.time_mlp(embedding).chunk(2, dim = 1)
        scale = scale[:, :, None, None]
        shift = shift[:, :, None, None]
        x = self.norm(x) * (1 + scale) + shift

        return self.down_layer(x)

class UpSample(UvitUpSample):
    """Upsample with skip-connection and AdaGN time conditioning"""
    def __init__(self, in_channels, out_channels, n_embedding, padding_mode = 'zeros'):
        super().__init__(in_channels, out_channels, padding_mode)

        # Time conditioning
        self.time_mlp = nn.Sequential(
            nn.SiLU(),
            nn.Linear(n_embedding, 2 * in_channels)
        )

    def forward(self, x, x_skip, embedding):
        # x: (B, C, H, W)
        # (time_)embedding: (B, D)
        # Apply time conditioning before upsampling

        scale, shift = self.time_mlp(embedding).chunk(2, dim = 1)
        scale = scale[:, :, None, None]
        shift = shift[:, :, None, None]

        x = self.norm(x) * (1 + scale) + shift
        x = self.upsampling(x)
        x = torch.cat([x_skip, x], dim = 1)
        return self.conv(x)

class ResNetBlock(UvitResNetBlock):
    """ResNet block with AdaGN time conditioning"""
    def __init__(self, in_channels, embedding_dim, drop_path_rate = 0., padding_mode = 'zeros'):
        super().__init__(in_channels, drop_path_rate, padding_mode)

        # Time conditioning
        self.time_mlp = nn.Sequential(
            nn.SiLU(),
            nn.Linear(embedding_dim, 2 * in_channels)
        )

    def forward(self, x, embedding):
        # x: (B, C, H, W)
        # (time_)embedding: (B, D)

        h = self.first_block(x)

        # Apply time conditioning
        scale, shift = self.time_mlp(embedding).chunk(2, dim = 1)
        # Broadcast to "match" spatial dims
        scale = scale[:, :, None, None]
        shift = shift[:, :, None, None]

        h = self.norm(h) * (1 + scale) + shift
        h = self.act(h)
        h = self.conv(h)

        return x + stochastic_depth(
            h,
            p = self.drop_path_rate,
            mode = 'batch',
            training = self.training
        )

class ConvBottleneck(nn.Module):
    def __init__(self, in_channels, embedding_dim, n_blocks, drop_path_rates, padding_mode = 'zeros'):
        super().__init__()
        self.blocks = nn.ModuleList([
            ResNetBlock(in_channels, embedding_dim, drop_path_rate = drop_path_rates[i], padding_mode = padding_mode)
            for i in range(n_blocks)
        ])

    def forward(self, x, time_emb):
        for block in self.blocks:
            x = block(x, time_emb)

        return x

class IdentityBottleneck(nn.Module):
    def forward(self, x, time_emb):
        return x

class Encoder(nn.Module):
    def __init__(self, channels, embedding_dim, multiples, drop_path_rates, padding_mode = 'zeros'):
        super().__init__()
        self.encoder_blocks = nn.ModuleList()

        block_idx = 0
        for in_channels, num_blocks in zip(channels[:-1], multiples):
            self.encoder_blocks.append(nn.Module())
            self.encoder_blocks[-1].blocks = nn.ModuleList([
            ResNetBlock(in_channels,
                        embedding_dim=embedding_dim,
                        drop_path_rate = drop_path_rates[block_idx + i],
                        padding_mode = padding_mode
                        ) for i in range(num_blocks)
            ])
            block_idx += num_blocks


        self.downsamples = nn.ModuleList([
            DownSample(in_channels=channels[i],
                       out_channels = channels[i+1], n_embedding=embedding_dim
                       ) for i in range(len(channels) - 1)
        ])

    def forward(self, x, t):
        # x: (B, C, H, W)
        # t: (B, D)
        feature_maps = []

        for conv, downsample in zip(self.encoder_blocks, self.downsamples):
            for block in conv.blocks:
                x = block(x, t)

            feature_maps.append(x)
            x = downsample(x, t)

        return x, feature_maps

class Decoder(nn.Module):
    def __init__(self, channels, embedding_dim, multiples, drop_path_rates, padding_mode = 'zeros'):
        super().__init__()
        self.upsamples = nn.ModuleList([
            UpSample(channels[i],
                     channels[i+1],
                     embedding_dim,
                     padding_mode = padding_mode
                     ) for i in range(len(channels) - 1)
        ])

        self.decoder_blocks = nn.ModuleList()
        block_idx = 0
        for in_channels, num_blocks in zip(channels[1:], multiples):
            self.decoder_blocks.append(nn.Module())
            self.decoder_blocks[-1].blocks = nn.ModuleList([
                ResNetBlock(in_channels,
                            embedding_dim = embedding_dim,
                            drop_path_rate=drop_path_rates[block_idx + i],
                            padding_mode = padding_mode
                            ) for i in range(num_blocks)
            ])
            block_idx += num_blocks

    def forward(self, x, encoder_features, t):
        for upsample, skip_connection, conv in zip(self.upsamples, encoder_features, self.decoder_blocks):
            x = upsample(x, skip_connection, t)

            for block in conv.blocks:
                x = block(x, t)

        return x

class UNETbackbone(nn.Module):
    def __init__(self,
                 in_fields,
                 channels,
                 n_embedding,
                 n_time_hidden,
                 encoder_multiples,
                 decoder_multiples,
                 n_bottleneck,
                 out_fields = 1,
                 max_drop_path_rate = 0.2,
                 padding_mode = 'zeros'
        ):

        super().__init__()
        self.in_fields = in_fields
        self.n_embedding = n_embedding

        # Time embedding: timestep -> vector
        self.n_time_hidden = n_time_hidden

        # Input projection
        self.EmbedInput = nn.Conv2d(in_fields, channels[0], 1)

        # Build encoder/decoder
        encoder_multiples = construct_multiples(encoder_multiples, channels[:-1])
        decoder_multiples = construct_multiples(decoder_multiples, channels[:-1])

        # Calculate total number of blocks
        total_encoder_blocks = sum(encoder_multiples)
        total_decoder_blocks = sum(decoder_multiples)
        total_blocks = total_encoder_blocks + n_bottleneck + total_decoder_blocks

        self.bottleneck_stride = len(channels) - 1
        self.EmbedTime = TimeEmbedding(n_embedding, n_time_hidden)
        # self.EmbeddingActivation = nn.GELU()

        # Create linearly increasing drop path rates
        drop_path_rates = [i * max_drop_path_rate / (total_blocks - 1) for i in range(total_blocks)]

        # Split rates for Encoder, Bottleneck and Decoder
        encoder_rates = drop_path_rates[:total_encoder_blocks]
        bottleneck_rates = drop_path_rates[total_encoder_blocks:total_encoder_blocks+n_bottleneck]
        decoder_rates = drop_path_rates[total_encoder_blocks+n_bottleneck:]

        self.encoder = Encoder(channels,
                               n_embedding,
                               encoder_multiples,
                               encoder_rates,
                               padding_mode
                            )

        if n_bottleneck > 0:
            self.bottleneck = ConvBottleneck(channels[-1],
                                        n_embedding,
                                        n_bottleneck,
                                        bottleneck_rates,
                                        padding_mode
                                        )
        else:
            self.bottleneck = IdentityBottleneck()

        self.decoder = Decoder(channels[::-1],
                               n_embedding,
                               decoder_multiples,
                               decoder_rates,
                               padding_mode
                            )

        # Output layers
        self.out_norm = nn.GroupNorm(1, channels[0])
        self.out_act = nn.SiLU()
        self.output_layer = nn.Conv2d(in_channels = channels[0], out_channels=out_fields, kernel_size=1)

    def forward(self, x, t):
        # x: (B, C, H, W) - noisy image + conditioning
        # t: (B,) - timeste indices

        # AdaGN time embedding
        # embedding = torch.zeros(
            # x.size(0), self.n_embedding, *x.shape[-2:], device = x.device, dtype = x.dtype
        # )

        # embedding.add_(self.EmbedTime(t))
        time_embedding = self.EmbedTime(t)

        # DEBUG: Is the time embedding reasonable
        # print(f"Time embedding: [{time_embedding.min():.3f}, {time_embedding.max():.3f}], mean = {time_embedding.mean()}, std = {time_embedding.std()}")
        # embedding = self.EmbeddingActivation(embedding)

        # Process through U-Net
        x = self.EmbedInput(x)

        last_h, encoder_features = self.encoder(x, time_embedding)
        # print(f"{last_h.size()=}")
        encoder_features_inverse = encoder_features[::-1]

        # print(f"{last_h.size()=}")

        bottleneck_out = self.bottleneck(last_h, time_embedding)
        # bottleneck_out = self.bottleneck(last_h, embedding[..., ::2**self.bottleneck_stride, ::2**self.bottleneck_stride])

        out = self.decoder(bottleneck_out, encoder_features_inverse, time_embedding)

        out = self.out_norm(out)
        out = self.out_act(out)
        return self.output_layer(out)