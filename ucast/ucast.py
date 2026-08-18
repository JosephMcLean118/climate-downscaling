import torch
from torch import nn
import torch.nn.functional as F
from .utils import Conv2d, _weight_init, GroupNorm, AttentionOp, compute_area_weights, choose_block_size, block_attention, mosaic_attn_func, cSwiGLU
from mosaic.primitives import MosaicAttention
from einops import rearrange
import math

_silu = torch.nn.functional.silu

def run_mosaic_attn(q, k, v, x, to_strategy, num_heads, tokens, rearrange_pattern):
    sparse_block_size = min(16, tokens)
    sparse_block_count = min(3, tokens // sparse_block_size)
    block_size = choose_block_size(tokens, target_size=100)
    strategy_logits = rearrange(to_strategy(x), rearrange_pattern, s=3, h=num_heads)
    weights = torch.softmax(strategy_logits.float(), dim=0).type_as(x).unsqueeze(-1)

    return mosaic_attn_func(
        q=q, k=k, v=v, weight_ba_cmp_slc=weights,
        block_attn_size=block_size, sparse_block_size=sparse_block_size,
        sparse_block_count=sparse_block_count, block_attn_only=False
    )

class UNetBlock(nn.Module):
    def __init__(
        self,
        in_channels,
        out_channels,
        up=False,
        down=False,
        attention=False,
        block_sparse_attention=False,
        num_heads=None,
        channels_per_head=64,
        dropout=0.1,
        eps=1e-5,
    ):
        super().__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.noise_dim=64
        self.swiglu = cSwiGLU(
            dim=out_channels,
            hidden_dim=out_channels * 4,
            noise_dim=self.noise_dim,
        
        )
        self.num_heads = (
            0 if not attention else (num_heads if num_heads is not None else out_channels // channels_per_head)
        )

        self.norm0 = GroupNorm(num_channels=in_channels, eps=eps)
        self.conv0 = Conv2d(in_channels=in_channels, out_channels=out_channels, kernel=3, up=up, down=down)
        self.norm1 = GroupNorm(num_channels=out_channels, eps=eps)
        self.conv1 = Conv2d(in_channels=out_channels, out_channels=out_channels, kernel=3)
        self.dropout = nn.Dropout(p=dropout)
        self.block_sparse_attention = block_sparse_attention

        self.skip = None
        if out_channels != in_channels or up or down:
            kernel = 1 if out_channels != in_channels else 0
            self.skip = Conv2d(in_channels=in_channels, out_channels=out_channels, kernel=kernel, up=up, down=down)

        # Self-attention
        if self.num_heads:
            self.norm2 = GroupNorm(num_channels=out_channels, eps=eps)
            self.qkv = Conv2d(in_channels=out_channels, out_channels=out_channels * 3, kernel=1)
            self.proj = Conv2d(in_channels=out_channels, out_channels=out_channels, kernel=1)
            self.to_strategy = Conv2d(in_channels=self.out_channels, out_channels=3 * self.num_heads, kernel=1)

    def forward(self, x, z=None):
        orig = x
        x = self.conv0(_silu(self.norm0(x)))
        x = _silu(self.norm1(x))
        x = self.conv1(self.dropout(x))
        x = x.add_(self.skip(orig) if self.skip is not None else orig)
        
        # Functional perturbation
        if z is not None and x.shape[-1] < 128:
            b, c, h, w = x.shape
            tokens_x = rearrange(x,"b c h w -> b (h w) c")
            swiglu_out = self.swiglu(tokens_x,z)
            tokens_x = tokens_x + swiglu_out
            x = rearrange(tokens_x,"b (h w) c -> b c h w",h=h,w=w)

        if self.num_heads:
            b, c, h, w = x.shape
            tokens = h*w
            nh = self.num_heads
            B2, C2 = b * nh, c // nh
            normed = self.norm2(x)
            q, k, v = self.qkv(normed).reshape(B2, C2, 3, -1).unbind(2)

            if self.block_sparse_attention and tokens>=16:
                # Rearrange tensors
                block_size = choose_block_size(tokens, target_size=100)
                q, k, v = [rearrange(x, "(b h) d t -> b t h d", b=b, h=nh) for x in (q, k, v)]
                a = run_mosaic_attn(q=q, k=k, v=v, x=x, to_strategy=self.to_strategy, num_heads=nh, 
                                tokens=tokens, rearrange_pattern="b (s h) H W -> s b (H W) h")
                a = rearrange(a, "b (h w) nh c -> b (nh c) h w", h=h, w=w)
            else:
                # Dense attention
                attn_w = AttentionOp.apply(q, k)
                a = torch.einsum("nqk,nck->ncq", attn_w, v)
                a = rearrange(a, "(b nh) c (h w) -> b (nh c) h w", b=b, nh=nh, h=h, w=w)
            x = self.proj(a).add_(x)
            
        return x


# Transformer architecture

class TransformerUNetBlock(nn.Module):
    """
    Transormer style block to be used in UNet in place of convolutional style blocks.
    
    Rather than have attention at certain layers and use convolutions for feature
    extraction and resampling, use self attention at all layers and MLP based feature
    processing.
    """
    def __init__(
        self,
        in_channels,
        out_channels,
        up=False,
        down=False,
        num_heads=None,
        dropout=0.1,
        mlp_ratio=4.0,
        eps=1e-5,
    ):
        super().__init__()

        self.in_channels = in_channels
        self.out_channels = out_channels
        self.up = up
        self.down = down
        self.noise_dim=64
        self.swiglu = cSwiGLU(
            dim=out_channels,
            hidden_dim=out_channels * 4,
            noise_dim=self.noise_dim,
        
        )

        # Spatial resampling
        self.upsample = (nn.Upsample(scale_factor=2, mode="bilinear", align_corners=False) if up else None)
        self.downsample = (nn.AvgPool2d(kernel_size=2,stride=2)if down else None)
        
        self.input_proj = nn.Linear(in_channels, out_channels)

        if in_channels != out_channels:
            self.skip_proj = nn.Linear(in_channels, out_channels)
        else:
            self.skip_proj = nn.Identity()

        self.num_heads = (num_heads if num_heads is not None else max(1, out_channels // 64))
        self.to_strategy = nn.Linear(out_channels,3 * self.num_heads)

        assert out_channels % self.num_heads == 0

        self.norm1 = nn.LayerNorm(out_channels, eps=eps)
        self.qkv = nn.Linear(out_channels, out_channels * 3)
        self.proj = nn.Linear(out_channels,out_channels)

        # MLP
        self.norm2 = nn.LayerNorm(out_channels, eps=eps)

        hidden_channels = int(out_channels * mlp_ratio)

        self.mlp = nn.Sequential(
            nn.Linear(out_channels, hidden_channels),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_channels, out_channels),
            nn.Dropout(dropout),
        )

    def forward(self, x, z=None):
        residual = x
        nh = self.num_heads
        
        # Resampling
        if self.up:
            x = self.upsample(x)
            residual = self.upsample(residual)
    
        if self.down:
            x = self.downsample(x)
            residual = self.downsample(residual)

        B, C, H, W = x.shape
        tokens = H * W

        # Image to tokens
        x = rearrange(x, "b c h w -> b (h w) c")
        residual = rearrange(residual, "b c h w -> b (h w) c")
        sparse_block_size = min(16, tokens)
        nb = tokens // sparse_block_size

        # Input projection
        x = self.input_proj(x)

        # Residual projection
        residual = self.skip_proj(residual)

        # Self attention
        x_norm = self.norm1(x)
        qkv = self.qkv(x_norm)
        q, k, v = qkv.chunk(3, dim=-1)
        q = rearrange(q, "b n (h d) -> b n h d", h=nh)
        k = rearrange(k, "b n (h d) -> b n h d", h=nh)
        v = rearrange(v, "b n (h d) -> b n h d", h=nh)

        # Block Sparse Attention
        out = run_mosaic_attn(q=q, k=k, v=v, x=x_norm, to_strategy=self.to_strategy, num_heads=nh, 
                    tokens=tokens, rearrange_pattern="b n (s h) -> s b n h")

        # Heads to channels
        out = rearrange(out, "b n h d -> b n (h d)")
        out = self.proj(out)

        # Attention residual
        x = residual + out

        # MLP residual
        x = x + self.mlp(x)
        x = rearrange(x, "b (h w) c -> b c h w", h=H, w=W)

        return x   


class DhariwalUNet(nn.Module):
    """ADM U-Net for weather forecasting that has been adapted for
       climate downscaling on CORDEX-ML Bench."""

    def __init__(
        self,
        in_channels,
        out_channels,
        model_channels=128,
        channel_mult=(1, 2, 3, 4),
        num_blocks=3,
        attn_levels=(2, 3),
        block_sparse_attention=False,
        base="convolutional", 
        channels_per_head=64,
        dropout=0.1,
    ):            
        super().__init__()
        block_kwargs = dict(channels_per_head=channels_per_head, dropout=dropout, block_sparse_attention=block_sparse_attention)
        img_resolution = 16 
        self.noise_dim = 64
        # ── Encoder ──
        self.enc = nn.ModuleDict()
        for level, mult in enumerate(channel_mult):
            res = img_resolution >> level
            use_attn = level in attn_levels
            level_channels = int(model_channels * mult)
            if level == 0:
                cout = level_channels
                self.enc[f"{res}x{res}_conv"] = Conv2d(in_channels=in_channels, out_channels=cout, kernel=3)
            else:
                # Down block preserves channel count from the previous level
                # If not conv_base, use transformer base
                if base == "convolutional":
                    self.enc[f"{res}x{res}_down"] = UNetBlock(
                        in_channels=cout, out_channels=cout, down=True, **block_kwargs
                    )
                elif base == "transformer":
                    self.enc[f"{res}x{res}_down"] = TransformerUNetBlock(
                        in_channels=cout, out_channels=cout, down=True)
                else:
                    raise Exception("Invalid base chosen. Must be either convolution blocks or trasnformer blocks")
            for idx in range(num_blocks):
                cin = cout
                cout = level_channels
                if base == "convolutional":
                    self.enc[f"{res}x{res}_block{idx}"] = UNetBlock(
                        in_channels=cin, out_channels=cout, attention=use_attn, **block_kwargs
                    )
                else:
                    self.enc[f"{res}x{res}_block{idx}"] = TransformerUNetBlock(
                        in_channels=cin, out_channels=cout
                    )

        skips = [block.out_channels for block in self.enc.values()]
        channel_mult_dec = list(reversed(channel_mult))
        
        # ── Decoder ──
        self.dec = nn.ModuleDict()
        for dec_block_i, mult in enumerate(channel_mult_dec):
            level = len(channel_mult_dec) - 1 - dec_block_i  
            res = img_resolution >> level
            use_attn = level in attn_levels
            level_channels = int(model_channels * mult)
            if level == len(channel_mult) - 1:
                # Bottleneck: two blocks that preserve channel count before upsampling begins
                if base == "convolutional":
                    self.dec[f"{res}x{res}_in0"] = UNetBlock(
                        in_channels=cout, out_channels=cout, attention=True, **block_kwargs
                    )
                    self.dec[f"{res}x{res}_in1"] = UNetBlock(in_channels=cout, out_channels=cout, **block_kwargs)
                else:
                    self.dec[f"{res}x{res}_in0"] = TransformerUNetBlock(
                        in_channels=cout, out_channels=cout)
                    self.dec[f"{res}x{res}_in1"] = TransformerUNetBlock(in_channels=cout, out_channels=cout)
            else:
                if base == "convolutional":
                    # Up block preserves channel count from the previous decoder level
                    self.dec[f"{res}x{res}_up"] = UNetBlock(in_channels=cout, out_channels=cout, up=True, **block_kwargs)
                else:
                    self.dec[f"{res}x{res}_up"] = TransformerUNetBlock(in_channels=cout, out_channels=cout, up=True)
                    
            for idx in range(num_blocks + 1):
                cin = cout + skips.pop()
                cout = level_channels
                if base == "convolutional":
                    self.dec[f"{res}x{res}_block{idx}"] = UNetBlock(
                        in_channels=cin, out_channels=cout, attention=use_attn, **block_kwargs
                    )
                else:
                    self.dec[f"{res}x{res}_block{idx}"] = TransformerUNetBlock(
                        in_channels=cin, out_channels=cout)

        # Upsample to 128x128 for CORDEX  target resolution (3 Upsamples)

        self.superres = nn.ModuleList([
            UNetBlock(cout, cout, up=True),
            UNetBlock(cout, cout),
            UNetBlock(cout, cout),
            
            UNetBlock(cout, cout, up=True),
            UNetBlock(cout, cout),
            UNetBlock(cout, cout),
            
            UNetBlock(cout, cout, up=True),
            UNetBlock(cout, cout),
            UNetBlock(cout, cout),
        ])
        self.out_norm = GroupNorm(num_channels=cout)
        self.out_conv = Conv2d(in_channels=cout, out_channels=out_channels, kernel=3)

    def forward(self, inputs, stochasticity="dropout", dynamical_condition=None, static_condition=None):
        # Concatenate conditional channels
        parts = [inputs]
        if dynamical_condition is not None:
            parts.append(dynamical_condition)
        if static_condition is not None:
            parts.append(static_condition)
        x = torch.cat(parts, dim=1) if len(parts) > 1 else inputs

        if stochasticity == "perturbation":
            z = torch.randn(x.shape[0], self.noise_dim, device=x.device, dtype=x.dtype)
        else:
            z = None

        # Encoder
        skips = []
        for block in self.enc.values():
            if isinstance(block, Conv2d):
                x = block(x)
            elif stochasticity == "perturbation":
                x = block(x, z)
            else:
                x = block(x)
            skips.append(x)

        # Decoder
        for block in self.dec.values():
            if x.shape[1] != block.in_channels:
                skip = skips.pop()
                # Handle mismatched spatial dims via bilinear interpolation
                if skip.shape[-2:] != x.shape[-2:]:
                    x = F.interpolate(x, size=skip.shape[-2:], mode="bilinear")
                x = torch.cat([x, skip], dim=1)
            if stochasticity == "perturbation":
                x = block(x, z)
            else:
                x = block(x)

        # Upsample to 128x128
        for block in self.superres:
            if stochasticity == "perturbation":
                x = block(x, z)
            else:
                x = block(x)

        x = self.out_conv(_silu(self.out_norm(x)))
        return x