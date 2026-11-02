import torch

from torch import nn
from torch.nn import functional as F
from einops import rearrange
from torchvision.ops import stochastic_depth

from .unet import construct_multiples
from .uvit import MLPBlock as UvitMLPBlock, AttnetionBlock as UvitAttentionBlock
from .unet_backbone_sde import TimeEmbedding, Encoder, Decoder, ResNetBlock, UpSample, DownSample, IdentityBottleneck


class MLPBlock(UvitMLPBlock):
    """MLP Block with AdaLN-Zero time conditioning"""
    def __init__(self,
                 in_channels,
                 embedding_dim,
                 expansion_factor = 4,
                 dropout_rate = .1,
                 drop_path_rate = 0.,
                 adaln_zero = False
                ):

        super().__init__(in_channels, expansion_factor, dropout_rate, drop_path_rate)

        if adaln_zero:
            # Time conditioning
            self.time_mlp = nn.Sequential(
                nn.SiLU(),
                nn.Linear(embedding_dim, 3 * in_channels)
            )

            # AdaLN-Zero: Initialize the final third of time_mlp's linear layer
            with torch.no_grad():
                self.time_mlp[-1].weight[2 * in_channels:].zero_()
                self.time_mlp[-1].bias[2 * in_channels:].zero_()

            self.forward_function = self.forward_with_gate

        else:
            self.time_mlp = nn.Sequential(
                nn.SiLU(),
                nn.Linear(embedding_dim, 2 * in_channels)
            )

            self.forward_function = self.forward_without_gate

    def forward_with_gate(self, x, embedding):
        # x: (B, L, C) - sequence of tokens
        # (time_)embedding: (B, D)

        # AdaLN-Zero, Each (B, C)
        scale, shift, gate = self.time_mlp(embedding).chunk(3, dim = -1)

        h = self.norm(x) * (1 + scale.unsqueeze(1)) + shift.unsqueeze(1) # unsqueeze for (B, L, C)
        h = self.mlp(h)

        # Apply gate (alpha) before resiudual connection
        h = gate.unsqueeze(1) * h

        return x + stochastic_depth(
            h,
            p = self.drop_path_rate,
            mode = 'row',
            training = self.training
        )

    def forward_without_gate(self, x, embedding):
        # x: (B, L, C) - sequence of tokens
        # (time_)embedding: (B, D)

        # AdaLN, Each (B, C)
        scale, shift = self.time_mlp(embedding).chunk(2, dim = -1)

        h = self.norm(x) * (1 + scale.unsqueeze(1)) + shift.unsqueeze(1) # unsqueeze for (B, L, C)
        h = self.mlp(h)

        return x + stochastic_depth(
            h,
            p = self.drop_path_rate,
            mode = 'row',
            training = self.training
        )

    def forward(self, x, embedding):
        return self.forward_function(x, embedding)

class AttentionBlock(UvitAttentionBlock):
    """Attention block with AdaLN-Zero time conditioning"""
    def __init__(self,
                 in_channels,
                 n_embedding,
                 n_heads = 8,
                 dropout_rate = 0.1,
                 drop_path_rate = 0.,
                 adaln_zero = False
                ):

        super().__init__(in_channels, n_heads, dropout_rate, drop_path_rate)

        if adaln_zero:
            self.time_mlp = nn.Sequential(
                nn.SiLU(),
                nn.Linear(n_embedding, 3 * in_channels)
            )

            # AdaLN-Zero: Initialize the final third of time_mlp's linear layer
            with torch.no_grad():
                self.time_mlp[-1].weight[2 * in_channels:].zero_()
                self.time_mlp[-1].bias[2 * in_channels:].zero_()

            self.forward_function = self.forward_with_gate

        else:
            self.time_mlp = nn.Sequential(
                nn.SiLU(),
                nn.Linear(n_embedding, 2 * in_channels)
            )

            self.forward_function = self.forward_without_gate

    def forward_attention(self, h_out):
        # Multi-Head Self-Attention
        qkv = self.qkv(h_out)
        b, l, c3 = qkv.shape

        # head_dim = int(x.size(-1) / self.n_heads)
        head_dim = c3 // 3 // self.n_heads

        qkv = rearrange(qkv,
                        "b l (k h c) -> k b h l c",
                        b=b, l=l, h=self.n_heads, c=head_dim, k=3
        )

        attn_out = F.scaled_dot_product_attention(
            query = qkv[0], key = qkv[1], value = qkv[2]
        )

        attn_out = rearrange(attn_out,
                             "b h l c -> b l (h c)",
                             b=b, l=l, h=self.n_heads, c=head_dim
        )
        attn_out = self.out_layer(attn_out)
        attn_out = self.dropout(attn_out)

        return attn_out


    def forward_with_gate(self, x, embedding):
        # x: (B, L, C) - sequence of tokens
        # (time_)embedding: (B, D)

        # AdaLN-Zero, Each (B, C)
        scale, shift, gate = self.time_mlp(embedding).chunk(3, dim = 1)

        h_out = self.norm(x) * (1 + scale.unsqueeze(1)) + shift.unsqueeze(1) # unsqueeze for (B, L, C)

        # Multi-Head Self-Attention
        # Apply gate (alpha) before residual connection
        attn_out = gate.unsqueeze(1) * self.forward_attention(h_out)

        return x + stochastic_depth(
            attn_out,
            p = self.drop_path_rate,
            mode = 'row',
            training = self.training
        )

    def forward_without_gate(self, x, embedding):
        # x: (B, L, C) - sequence of tokens
        # (time_)embedding: (B, D)

        # AdaLN, Each (B, C)
        scale, shift = self.time_mlp(embedding).chunk(2, dim = 1)

        h_out = self.norm(x) * (1 + scale.unsqueeze(1)) + shift.unsqueeze(1) # unsqueeze for (B, L, C)

        # Multi-head self attention
        attn_out = self.forward_attention(h_out)

        return x + stochastic_depth(
            attn_out,
            p = self.drop_path_rate,
            mode = 'row',
            training = self.training
        )

    def forward(self, x, embedding):
        return self.forward_function(x, embedding)


class Transformer(nn.Module):
    def __init__(self,
                 in_channels,
                 n_embedding,
                 n_bottleneck,
                 latent_dims,
                 drop_path_rates,
                 attn_dropout = 0.1,
                 mlp_dropout = 0.1,
                 adaln_zero = False
                ):

        super().__init__()
        self.pos_emb = nn.Parameter(torch.empty(1, latent_dims[0] * latent_dims[1], in_channels))
        nn.init.normal_(self.pos_emb, mean = 0.0, std = 0.01)

        self.self_attention = nn.ModuleList([
            AttentionBlock(
                in_channels,
                n_embedding,
                dropout_rate = attn_dropout,
                drop_path_rate = drop_path_rates[i],
                adaln_zero = adaln_zero
                ) for i in range(n_bottleneck)
        ])

        self.mlp_block = nn.ModuleList([
            MLPBlock(in_channels,
                     n_embedding,
                     dropout_rate = mlp_dropout,
                     drop_path_rate = drop_path_rates[i],
                     adaln_zero = adaln_zero,
                     ) for i in range(n_bottleneck)
        ])


    def forward(self, x, time_emb):
        b, c, h, w, = x.shape
        h_curr = rearrange(x,
                      "b c h w -> b (h w) c",
                      b=b, c=c, h=h, w=w)
        h_curr = h_curr + self.pos_emb

        for self_attention, mlp in zip(self.self_attention, self.mlp_block):
            h_curr = self_attention(h_curr, time_emb)
            h_curr = mlp(h_curr, time_emb)

        return rearrange(h_curr,
                         "b (h w) c -> b c h w",
                         b=b, c=c, h=h, w=w)


class UVITbackbone(nn.Module):
    def __init__(self,
                 in_fields,
                 channels,
                 n_embedding,
                 n_time_hidden,
                 encoder_multiples,
                 decoder_multiples,
                 n_bottleneck,
                 latent_dims,
                 out_fields = 1,
                 max_drop_path_rate = 0.2,
                 attn_dropout = 0.1,
                 mlp_dropout = 0.1,
                 adaln_zero = False,
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
            self.bottleneck = Transformer(channels[-1],
                                        n_embedding,
                                        n_bottleneck,
                                        latent_dims,
                                        bottleneck_rates,
                                        attn_dropout,
                                        mlp_dropout,
                                        adaln_zero
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

    
class MultiResUVITbackbone(nn.Module):
    def __init__(self,
                 in_fields_lr,
                 in_fields_hr,
                 channels,
                 hr_channels,
                 n_embedding,
                 n_time_hidden,
                 encoder_multiples,
                 decoder_multiples,
                 n_bottleneck,
                 latent_dims,
                 out_fields_lr,
                 out_fields_hr,
                 max_drop_path_rate=0.2,
                 attn_dropout=0.1,
                 mlp_dropout=0.1,
                 adaln_zero = False,
                 padding_mode = 'reflect'
    ):
        super().__init__()

        enc_mult = construct_multiples(encoder_multiples, channels[:-1])
        dec_mult = construct_multiples(decoder_multiples, channels[:-1])
        hr_enc_mult = construct_multiples(encoder_multiples, [hr_channels])
        hr_dec_mult = construct_multiples(decoder_multiples, [hr_channels])

        total = sum(hr_enc_mult) + sum(enc_mult) + n_bottleneck + sum(dec_mult) + sum(hr_dec_mult)
        drop_path_rates = [i * max_drop_path_rate / max(total - 1, 1) for i in range(total)]
        
        hr_enc_dpr = drop_path_rates[:sum(hr_enc_mult)]
        enc_dpr = drop_path_rates[sum(hr_enc_mult):sum(hr_enc_mult)+sum(enc_mult)]
        bn_dpr = drop_path_rates[sum(hr_enc_mult)+sum(enc_mult):sum(hr_enc_mult)+sum(enc_mult)+n_bottleneck]
        dec_dpr = drop_path_rates[sum(hr_enc_mult)+sum(enc_mult)+n_bottleneck:sum(hr_enc_mult)+sum(enc_mult)+n_bottleneck+sum(dec_mult)]
        hr_dec_dpr = drop_path_rates[sum(hr_enc_mult)+sum(enc_mult)+n_bottleneck+sum(dec_mult):]

        self.EmbedTime = TimeEmbedding(n_embedding, n_time_hidden)

        # HR entry stage
        self.hr_input_proj = nn.Conv2d(in_fields_hr, hr_channels, 1)
        self.hr_encoder_blocks = nn.ModuleList([
            ResNetBlock(hr_channels,
                        n_embedding,
                        drop_path_rate=hr_enc_dpr[i],
                        padding_mode=padding_mode)
            for i in range(sum(hr_enc_mult))
        ])
        self.hr_downsample = DownSample(hr_channels, hr_channels, n_embedding, padding_mode)

        # Main Path, this is where the HR is concatenated with LR
        self.lr_input_proj = nn.Conv2d(in_fields_lr, hr_channels, 1)
        self.EmbedInput = nn.Conv2d(2 * hr_channels, channels[0], 1)

        self.encoder = Encoder(
                        channels,
                        n_embedding,
                        enc_mult,
                        enc_dpr,
                        padding_mode
                    )

        self.bottleneck = (
            Transformer(
                channels[-1],
                n_embedding,
                n_bottleneck,
                latent_dims,
                bn_dpr,
                attn_dropout,
                mlp_dropout,
                adaln_zero
            ) if n_bottleneck > 0 else IdentityBottleneck()
        )

        self.decoder = Decoder(
                        channels[::-1],
                        n_embedding,
                        dec_mult,
                        dec_dpr,
                        padding_mode
        )

        self.out_norm = nn.GroupNorm(1, channels[0])
        self.out_act = nn.SiLU()

        # LR output
        self.lr_output_layer = nn.Conv2d(channels[0], out_fields_lr, 1)

        # HR decoder stage
        self.hr_upsample = UpSample(channels[0], hr_channels, n_embedding, padding_mode)
        self.hr_decoder_blocks = nn.ModuleList([
            ResNetBlock(hr_channels,
                        n_embedding,
                        drop_path_rate=hr_dec_dpr[i],
                        padding_mode=padding_mode)
            for i in range(sum(hr_dec_mult))
        ])

        self.hr_out_norm = nn.GroupNorm(1, hr_channels)
        self.hr_out_act = nn.SiLU()
        self.hr_output_layer = nn.Conv2d(hr_channels, out_fields_hr, 1)

    def forward(self, x_lr, x_hr, t):
        time_emb = self.EmbedTime(t)

        # HR encoder
        hr = self.hr_input_proj(x_hr)
        for block in self.hr_encoder_blocks:
            hr = block(hr, time_emb)
        hr_skip = hr
        hr_down = self.hr_downsample(hr, time_emb)

        # Main network
        lr = self.lr_input_proj(x_lr)
        x = self.EmbedInput(torch.cat([lr, hr_down], dim = 1))

        last_h, enc_features = self.encoder(x, time_emb)
        bn_out = self.bottleneck(last_h, time_emb)
        dec_out = self.decoder(bn_out, enc_features[::-1], time_emb)

        out_lr = self.lr_output_layer(self.out_act(self.out_norm(dec_out)))

        # HR decoder
        hr_out = self.hr_upsample(dec_out, hr_skip, time_emb)
        for block in self.hr_decoder_blocks:
            hr_out = block(hr_out, time_emb)
        out_hr = self.hr_output_layer(self.hr_out_act(self.hr_out_norm(hr_out)))

        return out_lr, out_hr

