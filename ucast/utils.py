import numpy as np
import torch
import torch.nn.functional as F
from torch import nn
from .muon import Muon, get_muon_momentum, muon_update, zeropower_via_newtonschulz5
import math
from torch.nn import RMSNorm
from torch.nn.functional import scaled_dot_product_attention
from einops import rearrange, reduce, repeat
import triton
import triton.language as tl



def flash_attn_func(q, k, v):
    # q,k,v: (batch, tokens, heads, dim)
    q = q.transpose(1, 2)
    k = k.transpose(1, 2)
    v = v.transpose(1, 2)

    out = scaled_dot_product_attention(q, k, v)

    return out.transpose(1, 2)

def _weight_init(shape, fan_in):
    return np.sqrt(3 / fan_in) * (torch.rand(*shape) * 2 - 1)


class Conv2d(nn.Module):
    def __init__(self, in_channels, out_channels, kernel, up=False, down=False):
        super().__init__()
        assert not (up and down)
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.up = up
        self.down = down
        self.padding = 0

        if kernel == 0:
            self.weight = None
            self.bias = None
        else:
            fan_in = in_channels * kernel * kernel
            self.weight = nn.Parameter(_weight_init((out_channels, in_channels, kernel, kernel), fan_in))
            self.bias = nn.Parameter(torch.zeros(out_channels))  # zero-init bias is also standard practice
            self.padding = kernel // 2

    def forward(self, x):
        if self.up:
            x = F.interpolate(x, scale_factor=2, mode="bilinear")
        if self.down:
            x = F.avg_pool2d(x, kernel_size=2)

        if self.weight is not None:
            x = F.conv2d(x, self.weight, padding=self.padding)
        if self.bias is not None:
            x = x.add_(self.bias.reshape(1, -1, 1, 1))
        return x


class GroupNorm(nn.Module):
    """Group norm that automatically picks num_groups based on min_channels_per_group.

    Stores weight/bias directly (not wrapped in nn.GroupNorm) so that the parameter
    names match the checkpoint exactly (e.g. ``norm0.weight`` not ``norm0.gn.weight``).
    """

    def __init__(self, num_channels, eps=1e-5, min_channels_per_group=4):
        super().__init__()
        num_groups = 32
        while num_channels % num_groups != 0 or num_channels // num_groups < min_channels_per_group:
            num_groups //= 2
        self.num_groups = num_groups
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(num_channels))
        self.bias = nn.Parameter(torch.zeros(num_channels))

    def forward(self, x):
        return F.group_norm(
            x, num_groups=self.num_groups, weight=self.weight.to(x.dtype), bias=self.bias.to(x.dtype), eps=self.eps
        )


class AttentionOp(torch.autograd.Function):
    @staticmethod
    def forward(ctx, q, k):
        w = (
            torch.einsum("ncq,nck->nqk", q.to(torch.float32), (k / math.sqrt(q.shape[1])).to(torch.float32))
            .softmax(dim=2)
            .to(q.dtype)
        )
        ctx.save_for_backward(q, k, w)
        return w

    
    @staticmethod
    def backward(ctx, dw):
        q, k, w = ctx.saved_tensors
        db = torch._softmax_backward_data(
            grad_output=dw.to(torch.float32), output=w.to(torch.float32), dim=2, input_dtype=torch.float32
        )
        dq = torch.einsum("nck,nqk->ncq", k.to(torch.float32), db).to(q.dtype) / np.sqrt(k.shape[1])
        dk = torch.einsum("ncq,nqk->nck", q.to(torch.float32), db).to(k.dtype) / np.sqrt(k.shape[1])
        return dq, dk


def choose_block_size(num_tokens, target_size=100):
    divisors = [
        i for i in range(1, num_tokens + 1)
        if num_tokens % i == 0
    ]

    return min(divisors, key=lambda x: abs(x - target_size))

# local block attn
def block_attention(q: torch.Tensor, k: torch.Tensor, v: torch.Tensor, block_size: int):
    batch_size = q.shape[0]
    q, k, v = map(lambda x: rearrange(x, 'b (nb bs) h d -> (b nb) bs h d', bs=block_size), (q, k, v))
    o_ba = flash_attn_func(q, k, v)
    return rearrange(o_ba, '(b nb) bs h d -> b (nb bs) h d', b=batch_size)

#picks which blocks branch 3 should look at
@torch.no_grad()
def attn_topk(q: torch.Tensor, k: torch.Tensor, block_count: int):
    Hq, Hk = q.shape[2], k.shape[2]
    G = Hq // Hk
    k = k.repeat_interleave(G, dim=2)

    scores = torch.matmul(
        rearrange(q, 'b t h d -> b h t d'),
        rearrange(k, 'b t h d -> b h d t')
    )

    if Hq != Hk:
        scores = reduce(scores, 'b (g h) t k -> b h t k', 'mean', g=G)

    scores = rearrange(scores, 'b h t k -> b t h k')
    top_indices = scores.topk(k=block_count, dim=-1, largest=True)[1]
    return top_indices   

@triton.jit
def mosaic_attn_fwd_kernel(
    q_ptr, k_ptr, v_ptr, output_ptr, lse_ptr, block_indices_ptr,
    softmax_scale: tl.constexpr,
    seq_len: tl.constexpr,
    num_kv_heads: tl.constexpr,
    num_q_heads: tl.constexpr,
    q_heads_per_kv_head: tl.constexpr,
    feature_dim: tl.constexpr,
    kv_block_size: tl.constexpr,
    num_kv_blocks_per_q_block: tl.constexpr,
    q_tile_size: tl.constexpr,
):
    """
    Sparse attention forward kernel:
        for each query tile (i.e. block chunk), for each query head, attend to a subset of key/value blocks.
    """
    LOG2_E: tl.constexpr = 1.44269504089

    q_tile_id = tl.program_id(0)
    q_head_id = tl.program_id(1)
    batch_kv_head_id = tl.program_id(2)

    batch_idx = batch_kv_head_id // num_kv_heads
    kv_head_idx = batch_kv_head_id % num_kv_heads
    q_head_idx = kv_head_idx * q_heads_per_kv_head + q_head_id

    batch_offset = batch_idx * seq_len
    q_tile_start = q_tile_id * q_tile_size
    num_blocks_in_seq = seq_len // kv_block_size
    tiles_per_block = kv_block_size // q_tile_size
    q_block_id = q_tile_id // tiles_per_block

    block_indices_offset = (
        batch_idx * num_blocks_in_seq * num_kv_heads * num_kv_blocks_per_q_block +
        q_block_id * num_kv_heads * num_kv_blocks_per_q_block +
        kv_head_idx * num_kv_blocks_per_q_block
    )

    q_base_ptr = q_ptr + batch_offset * num_q_heads  * feature_dim + q_head_idx  * feature_dim
    k_base_ptr = k_ptr + batch_offset * num_kv_heads * feature_dim + kv_head_idx * feature_dim
    v_base_ptr = v_ptr + batch_offset * num_kv_heads * feature_dim + kv_head_idx * feature_dim

    q_tile_ptr = tl.make_block_ptr(
        base=q_base_ptr,
        shape=(seq_len, feature_dim),
        strides=(num_q_heads * feature_dim, 1),
        offsets=(q_tile_start, 0),
        block_shape=(q_tile_size, feature_dim),
        order=(1, 0)
    )

    output_tile_ptr = tl.make_block_ptr(
        base=output_ptr + batch_offset * num_q_heads * feature_dim + q_head_idx * feature_dim,
        shape=(seq_len, feature_dim),
        strides=(num_q_heads * feature_dim, 1),
        offsets=(q_tile_start, 0),
        block_shape=(q_tile_size, feature_dim),
        order=(1, 0)
    )

    lse_base_ptr = lse_ptr + (batch_offset + q_tile_start) * num_q_heads + tl.arange(0, q_tile_size) * num_q_heads + q_head_idx

    output_accum = tl.zeros([q_tile_size, feature_dim], dtype=tl.float32)
    max_scores = tl.full([q_tile_size], float('-inf'), dtype=tl.float32)
    sum_exp_scores = tl.zeros([q_tile_size], dtype=tl.float32)

    q_tile = tl.load(q_tile_ptr)
    q_tile = (q_tile * softmax_scale * LOG2_E).to(tl.float16)

    for i in range(num_kv_blocks_per_q_block):
        kv_block_start = kv_block_size * tl.load(block_indices_ptr + block_indices_offset + i).to(tl.int32)

        k_block_ptr = tl.make_block_ptr(
            base=k_base_ptr,
            shape=(feature_dim, seq_len),
            strides=(1, num_kv_heads * feature_dim),
            offsets=(0, kv_block_start),
            block_shape=(feature_dim, kv_block_size),
            order=(1, 0)
        )

        v_block_ptr = tl.make_block_ptr(
            base=v_base_ptr,
            shape=(seq_len, feature_dim),
            strides=(num_kv_heads * feature_dim, 1),
            offsets=(kv_block_start, 0),
            block_shape=(kv_block_size, feature_dim),
            order=(1, 0)
        )

        k_block = tl.load(k_block_ptr).to(tl.float16)
        v_block = tl.load(v_block_ptr).to(tl.float16)

        attention_scores = tl.dot(q_tile, k_block, allow_tf32=False)

        new_max = tl.max(attention_scores, axis=1)
        old_max = max_scores
        max_scores = tl.maximum(max_scores, new_max)
        rescale = tl.exp2(old_max - max_scores)
        attention_probs = tl.exp2(attention_scores - max_scores[:, None])
        sum_exp_scores = sum_exp_scores * rescale + tl.sum(attention_probs, axis=1)

        output_accum = output_accum * rescale[:, None]
        output_accum += tl.dot(attention_probs.to(tl.float16), v_block)

    final_output = output_accum / sum_exp_scores[:, None]
    log_sum_exp = (max_scores + tl.log2(sum_exp_scores))

    tl.store(output_tile_ptr, final_output.to(q_ptr.dtype.element_ty))
    tl.store(lse_base_ptr, log_sum_exp)

    
def mosaic_attn_fwd(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    block_indices: torch.LongTensor,
    block_size: int,
    softmax_scale: float,
):
    batch_size, seq_len, num_kv_heads, feature_dim = k.shape
    num_q_heads = q.shape[2]
    num_kv_blocks_per_q_block = block_indices.shape[-1]
    q_heads_per_kv_head = num_q_heads // num_kv_heads

    output = torch.empty(batch_size, seq_len, num_q_heads, feature_dim, dtype=v.dtype, device=q.device)
    lse = torch.empty(batch_size, seq_len, num_q_heads, dtype=torch.float32, device=q.device)

    grid = lambda META: (
        triton.cdiv(seq_len, META['q_tile_size']),
        q_heads_per_kv_head,
        batch_size * num_kv_heads
    )

    mosaic_attn_fwd_kernel[grid](
        q_ptr = q,
        k_ptr = k,
        v_ptr = v,
        output_ptr = output,
        lse_ptr = lse,
        block_indices_ptr = block_indices,
        softmax_scale = softmax_scale,
        seq_len = seq_len,
        num_kv_heads = num_kv_heads,
        num_q_heads = num_q_heads,
        q_heads_per_kv_head = q_heads_per_kv_head,
        feature_dim = feature_dim,
        kv_block_size = block_size,
        num_kv_blocks_per_q_block = num_kv_blocks_per_q_block,
        q_tile_size=16
    )

    return output, lse


#combines all 3 branches — THE core function
def mosaic_attn_func(
    q, k, v,
    weight_ba_cmp_slc,
    block_attn_size, sparse_block_size, sparse_block_count,
    block_attn_only, no_compression=False,
):

    # Local block attention
    o_ba = block_attention(q, k, v, block_attn_size)

    if block_attn_only:
        return o_ba

    q_cmp = reduce(q, 'b (nb bs) h d -> b nb h d', 'mean', bs=sparse_block_size)
    k_cmp = reduce(k, 'b (nb bs) h d -> b nb h d', 'mean', bs=sparse_block_size)

    if no_compression:
        block_indices = attn_topk(q_cmp, k_cmp, sparse_block_count)
        o_slc = mosaic_sparse_attn(q, k, v, block_indices, sparse_block_size)
        w_ba = weight_ba_cmp_slc[0]
        w_slc = weight_ba_cmp_slc[2]
        w_sum = w_ba + w_slc + 1e-8
        return o_ba * (w_ba / w_sum) + o_slc * (w_slc / w_sum)

    # Compressed attention
    v_cmp = reduce(v, 'b (nb bs) h d -> b nb h d', 'mean', bs=sparse_block_size)
    o_cmp = flash_attn_func(q_cmp, k_cmp, v_cmp)
    o_cmp = o_cmp.repeat_interleave(sparse_block_size, dim=1)

    if sparse_block_count == 0:
        w_ba = weight_ba_cmp_slc[0]
        w_cmp = weight_ba_cmp_slc[1]
        w_sum = w_ba + w_cmp + 1e-8
        return o_ba * (w_ba / w_sum) + o_cmp * (w_cmp / w_sum)

    # Selective attention
    block_indices = attn_topk(q_cmp, k_cmp, sparse_block_count)
    o_slc = mosaic_sparse_attn(q, k, v, block_indices, sparse_block_size)

    # Combine compressed, selective and local to get block sparse attention
    return o_ba * weight_ba_cmp_slc[0] + o_cmp * weight_ba_cmp_slc[1] + o_slc * weight_ba_cmp_slc[2]


class MosaicAttnFunction(torch.autograd.Function):

    @staticmethod
    @torch.amp.custom_fwd(device_type='cuda')
    def forward(
        ctx: torch.autograd.function.FunctionCtx,
        q: torch.Tensor,
        k: torch.Tensor,
        v: torch.Tensor,
        block_indices: torch.Tensor,
        block_size: int,
        softmax_scale: float
    ):
        q, k, v, block_indices = map(lambda x: x.contiguous(), (q, k, v, block_indices))

        ctx.dtype = q.dtype

        output, lse = mosaic_attn_fwd(
            q=q, k=k, v=v,
            block_indices=block_indices,
            block_size=block_size,
            softmax_scale=softmax_scale,
        )

        ctx.save_for_backward(q, k, v, output, lse, block_indices)
        ctx.block_size = block_size
        ctx.softmax_scale = softmax_scale

        return output.to(q.dtype)

    @staticmethod
    @torch.amp.custom_bwd(device_type='cuda')
    def backward(
        ctx: torch.autograd.function.FunctionCtx,
        grad_o: torch.Tensor
    ):
        q, k, v, output, lse, block_indices = ctx.saved_tensors
        grad_o = grad_o.contiguous()
        grad_q, grad_k, grad_v = mosaic_attn_bwd(
            q=q, k=k, v=v, output=output, lse=lse, grad_o=grad_o,
            softmax_scale=ctx.softmax_scale,
            block_indices=block_indices,
            block_size=ctx.block_size,
        )
        return grad_q, grad_k, grad_v, None, None, None

def mosaic_sparse_attn(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    block_indices: torch.LongTensor,
    block_size: int,
    softmax_scale: float = None,
):
    #softmax_scale = q.shape[-1] ** -0.5 if softmax_scale is None else softmax_scale
    #return MosaicAttnFunction.apply(q, k, v, block_indices, block_size, softmax_scale)
    return flash_attn_func(q, k, v)


class MosaicAttention(nn.Module): # nn.Module wrapper: does QKV projection. Generates the gate weights, calls mosaic_attn_func
    def __init__(self, config, block_attn_only: bool, no_compression: bool = False):
        super().__init__()
        self.block_attn_only = block_attn_only
        self.no_compression = no_compression
        self.block_attn_size = config.block_attn_size
        self.sparse_block_size = config.sparse_block_size
        self.sparse_block_count = config.sparse_block_count

        q_heads = config.num_heads
        gqa_ratio = config.gqa_ratio
        dim = config.dim
        qkv_compress_ratio = config.qkv_compress_ratio
        rope = config.rope
        rope_theta = config.rope_theta

        kv_heads = q_heads // gqa_ratio
        head_dim = int(dim // q_heads // qkv_compress_ratio)

        self.q_heads = q_heads
        self.kv_heads = kv_heads

        self.to_q = nn.Linear(dim, q_heads * head_dim, bias=False)
        self.to_k = nn.Linear(dim, kv_heads * head_dim, bias=False)
        self.to_v = nn.Linear(dim, kv_heads * head_dim, bias=False)
        self.to_o = nn.Linear(q_heads * head_dim, dim, bias=False)

        self.q_rope = RoPE(head_dim, rope_theta) if rope else None
        self.k_rope = RoPE(head_dim, rope_theta) if rope else None

        if block_attn_only:
            self.to_strategy_combine_mlp = None
        else:
            self.to_strategy_combine_mlp = nn.Linear(dim, 3 * q_heads, bias=False)

    def generate_strategy_weights(self, x):
        if self.block_attn_only:
            return [None, None, None]
        strategy_logits = self.to_strategy_combine_mlp(x)
        strategy_logits = rearrange(strategy_logits, 't b (h s) -> s b t h', h=self.q_heads)
        strategy_weights = torch.softmax(strategy_logits.float(), dim=0).type_as(x)
        strategy_weights = strategy_weights.unsqueeze(-1)
        return strategy_weights

    def forward(self, x):
        q = self.to_q(x)
        k = self.to_k(x)
        v = self.to_v(x)

        strategy_weights = self.generate_strategy_weights(x)

        q = rearrange(q, 's b (h d) -> b s h d', h=self.q_heads)
        k = rearrange(k, 's b (h d) -> b s h d', h=self.kv_heads)
        v = rearrange(v, 's b (h d) -> b s h d', h=self.kv_heads)

        if self.q_rope is not None:
            q = self.q_rope(q)
            k = self.k_rope(k)

        output = mosaic_attn_func(
            q=q, k=k, v=v,
            weight_ba_cmp_slc=strategy_weights,
            block_attn_size=self.block_attn_size,
            sparse_block_size=self.sparse_block_size,
            sparse_block_count=self.sparse_block_count,
            block_attn_only=self.block_attn_only,
            no_compression=self.no_compression,
        )

        output = rearrange(output, 'b s h d -> s b (h d)')
        output = self.to_o(output)
        return output                 

def compute_area_weights(latitudes: torch.Tensor) -> torch.Tensor:
    w = torch.cos(torch.deg2rad(latitudes))
    return w / w.mean()

def weighted_mae_loss(pred, target, lat_weights=None):
    err = (pred - target).abs()
    if lat_weights is None:
        return err.mean()
    return (err * lat_weights.view(1, 1, -1, 1)).mean()


class EMA:
    def __init__(self, model, decay=0.9999):
        self.decay = decay
        self.shadow = {k: v.clone().detach() for k, v in model.state_dict().items()}

    @torch.no_grad()
    def update(self, model):
        for k, v in model.state_dict().items():
            if v.dtype.is_floating_point:
                self.shadow[k].mul_(self.decay).add_(v, alpha=1 - self.decay)
            else:
                self.shadow[k] = v.clone()

    def copy_to(self, model):
        model.load_state_dict(self.shadow, strict=True)


def build_optimizer(model, adamw_lr, adamw_wd, use_muon=False, muon_lr=0.003, muon_wd=0.03, muon_momentum=0.95 ):
    if not use_muon:
        return torch.optim.AdamW(model.parameters(), lr=adamw_lr, weight_decay=adamw_wd,
                                  eps=1e-8, betas=(0.9, 0.95))
    else:
        muon_params = []
        for name, p in model.named_parameters():
            if not p.requires_grad:
                continue
        
            if (
                p.ndim >= 2
                and not name.startswith("out_conv")
            ):
                muon_params.append(p)
        return {"muon": Muon(muon_params, muon_lr, muon_wd, muon_momentum),
               "adamw": torch.optim.AdamW(model.parameters(), lr=adamw_lr, weight_decay=adamw_wd,
                                  eps=1e-8, betas=(0.9, 0.95))}
    
class LinearWarmupCosineAnnealingLR(torch.optim.lr_scheduler._LRScheduler):
    def __init__(self, optimizer, warmup_steps, max_steps, warmup_start_lr=1e-8, eta_min=1e-8, last_epoch=-1):
        self.warmup_steps = warmup_steps
        self.max_steps = max_steps
        self.warmup_start_lr = warmup_start_lr
        self.eta_min = eta_min
        super().__init__(optimizer, last_epoch)

    def get_lr(self):
        step = self.last_epoch
        if step < self.warmup_steps:
            frac = step / max(1, self.warmup_steps)
            return [self.warmup_start_lr + frac * (base_lr - self.warmup_start_lr) for base_lr in self.base_lrs]
        progress = (step - self.warmup_steps) / max(1, self.max_steps - self.warmup_steps)
        progress = min(1.0, progress)
        return [self.eta_min + 0.5 * (base_lr - self.eta_min) * (1 + math.cos(math.pi * progress))
                for base_lr in self.base_lrs]

def fair_crps_loss(preds, target, lat_weights=None):
    """
    preds:  (M, B, C, H, W) — M stochastic ensemble members
    target: (B, C, H, W)
    """
    M = preds.shape[0]
    skill = (preds - target.unsqueeze(0)).abs().mean(dim=0) 
    if M > 1:
        diff = preds.unsqueeze(0) - preds.unsqueeze(1)  
        spread = diff.abs().sum(dim=(0, 1)) / (M * (M - 1))
    else:
        term2 = torch.zeros_like(skill)
    crps = skill - 0.5 * spread  
    if lat_weights is not None:
        crps = crps * lat_weights.view(1, 1, -1, 1)
    return crps.mean()

def enable_inference_dropout(model):
    """Keep dropout layers stochastic (MC-Dropout) even if the rest of the model is in eval/train mode."""
    for m in model.modules():
        if isinstance(m, nn.Dropout):
            m.train()
            
def disable_inference_dropout(model):
    """Keep dropout layers deterministic (disabled), even if the rest of the model is in eval/train mode."""
    for m in model.modules():
        if isinstance(m, nn.Dropout):
            m.eval()


def handle_fp16(scaler, loss_batch, model, adamw_opt, muon_opt, scheduler_adamw, scheduler_muon):
    """
    Handle underflow and overflow errors related to fp16 type data
    """
    #- Avoid underflow of gradient
    scaler.scale(loss_batch).backward()
    
    scaler.unscale_(adamw_opt)
    if muon_opt is not None:
        scaler.unscale_(muon_opt)

    #-  Clip gradient
    torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)

    #- Check if any grad is inf/nan (handle overflow)
    scaler.step(adamw_opt)
    if muon_opt is not None:
        scaler.step(muon_opt)

    #- adjust scale factor (shrink if overflown, otherwise grow)
    scaler.update()
    scheduler_adamw.step()
    if scheduler_muon is not None:
        scheduler_muon.step()


@triton.jit
def mosaic_attn_bwd_q_kernel(
    q_ptr, k_ptr, v_ptr, lse_ptr, delta_ptr, grad_o_ptr, grad_q_ptr, block_indices_ptr,
    softmax_scale: tl.constexpr,
    seq_len: tl.constexpr,
    num_kv_heads: tl.constexpr,
    num_q_heads: tl.constexpr,
    q_heads_per_kv_head: tl.constexpr,
    feature_dim: tl.constexpr,
    kv_block_size: tl.constexpr,
    num_kv_blocks_per_q_block: tl.constexpr,
    q_tile_size: tl.constexpr,
):
    LOG2_E: tl.constexpr = 1.44269504089
    LN_2: tl.constexpr = 0.69314718056

    q_tile_id = tl.program_id(0)
    q_head_id = tl.program_id(1)
    batch_kv_head_id = tl.program_id(2)

    batch_idx = batch_kv_head_id // num_kv_heads
    kv_head_idx = batch_kv_head_id % num_kv_heads
    q_head_idx = kv_head_idx * q_heads_per_kv_head + q_head_id

    batch_offset = batch_idx * seq_len
    q_tile_start = q_tile_id * q_tile_size
    tiles_per_block = kv_block_size // q_tile_size
    q_block_id = q_tile_id // tiles_per_block
    num_q_blocks = seq_len // kv_block_size

    block_indices_offset = (
        batch_idx * num_q_blocks * num_kv_heads * num_kv_blocks_per_q_block +
        q_block_id * num_kv_heads * num_kv_blocks_per_q_block +
        kv_head_idx * num_kv_blocks_per_q_block
    )

    q_offsets = (
        tl.arange(0, q_tile_size)[:, None] * num_q_heads * feature_dim +
        q_head_idx * feature_dim +
        tl.arange(0, feature_dim)[None, :]
    )

    lse_offsets = tl.arange(0, q_tile_size) * num_q_heads + q_head_idx

    q_base_ptr = q_ptr + (batch_offset + q_tile_start) * num_q_heads * feature_dim
    grad_o_base_ptr = grad_o_ptr + (batch_offset + q_tile_start) * num_q_heads * feature_dim
    delta_base_ptr = delta_ptr + (batch_offset + q_tile_start) * num_q_heads
    lse_base_ptr = lse_ptr + (batch_offset + q_tile_start) * num_q_heads
    grad_q_base_ptr = grad_q_ptr + (batch_offset + q_tile_start) * num_q_heads * feature_dim

    grad_q_accum = tl.zeros([q_tile_size, feature_dim], dtype=tl.float32)

    q_tile = tl.load(q_base_ptr + q_offsets)
    q_tile = (q_tile * softmax_scale * LOG2_E).to(tl.float16)

    grad_o_tile = tl.load(grad_o_base_ptr + q_offsets).to(tl.float16)
    delta_vals = tl.load(delta_base_ptr + lse_offsets)
    lse_vals = tl.load(lse_base_ptr + lse_offsets).to(tl.float32)

    for i in range(num_kv_blocks_per_q_block):
        kv_block_idx = tl.load(block_indices_ptr + block_indices_offset + i).to(tl.int32)

        k_block_ptr = tl.make_block_ptr(
            base=k_ptr + (batch_offset * num_kv_heads + kv_head_idx) * feature_dim,
            shape=(feature_dim, seq_len),
            strides=(1, num_kv_heads * feature_dim),
            offsets=(0, kv_block_idx * kv_block_size),
            block_shape=(feature_dim, kv_block_size),
            order=(0, 1)
        )

        v_block_ptr = tl.make_block_ptr(
            base=v_ptr + (batch_offset * num_kv_heads + kv_head_idx) * feature_dim,
            shape=(feature_dim, seq_len),
            strides=(1, num_kv_heads * feature_dim),
            offsets=(0, kv_block_idx * kv_block_size),
            block_shape=(feature_dim, kv_block_size),
            order=(0, 1)
        )

        k_block = tl.load(k_block_ptr).to(tl.float16)
        v_block = tl.load(v_block_ptr).to(tl.float16)

        attention_scores = tl.dot(q_tile, k_block)
        attention_probs = tl.exp2(attention_scores - lse_vals[:, None]) * LN_2

        grad_times_v = tl.dot(grad_o_tile, v_block)
        grad_scores = attention_probs * (grad_times_v - delta_vals[:, None])
        grad_q_accum += tl.dot(grad_scores.to(tl.float16), tl.trans(k_block.to(tl.float16)))

    grad_q_accum = grad_q_accum * softmax_scale * LOG2_E
    tl.store(grad_q_base_ptr + q_offsets, grad_q_accum.to(q_ptr.dtype.element_ty))


@torch.compile
@torch.no_grad()
def mosaic_block_mask(
    block_indices: torch.LongTensor,
):
    batch_size, num_blocks, num_heads, _ = block_indices.shape

    block_mask = torch.zeros(
        batch_size, num_blocks, num_heads, num_blocks,
        dtype=torch.bool, device=block_indices.device
    )

    batch_idx = torch.arange(batch_size, device=block_indices.device)[:, None, None, None]
    q_block_idx = torch.arange(num_blocks, device=block_indices.device)[None, :, None, None]
    head_idx = torch.arange(num_heads, device=block_indices.device)[None, None, :, None]

    block_mask[batch_idx, q_block_idx, head_idx, block_indices] = True

    block_mask_transposed = block_mask.permute(0, 2, 3, 1).contiguous()

    return block_mask_transposed



@triton.jit
def mosaic_attn_bwd_kv_kernel(
    q_ptr, k_ptr, v_ptr, lse_ptr, delta_ptr,
    grad_o_ptr, grad_k_ptr, grad_v_ptr,
    block_mask_ptr,
    softmax_scale: tl.constexpr,
    seq_len: tl.constexpr,
    num_kv_heads: tl.constexpr,
    num_q_heads: tl.constexpr,
    q_heads_per_kv_head: tl.constexpr,
    feature_dim: tl.constexpr,
    kv_block_size: tl.constexpr,
    q_tile_size: tl.constexpr,
):
    LOG2_E: tl.constexpr = 1.44269504089
    LN_2: tl.constexpr = 0.69314718056

    kv_block_id = tl.program_id(0)
    batch_kv_head_id = tl.program_id(1)

    batch_idx = batch_kv_head_id // num_kv_heads
    kv_head_idx = batch_kv_head_id % num_kv_heads
    batch_offset = batch_idx * seq_len

    num_blocks_in_seq = seq_len // kv_block_size
    tiles_per_block = kv_block_size // q_tile_size

    fine_mask_start = (
        batch_idx * num_kv_heads * num_blocks_in_seq * num_blocks_in_seq +
        kv_head_idx * num_blocks_in_seq * num_blocks_in_seq +
        kv_block_id * num_blocks_in_seq
    )

    k_block_ptr = tl.make_block_ptr(
        k_ptr + (batch_offset * num_kv_heads + kv_head_idx) * feature_dim,
        (seq_len, feature_dim), (num_kv_heads * feature_dim, 1),
        (kv_block_id * kv_block_size, 0), (kv_block_size, feature_dim), (1, 0)
    )

    v_block_ptr = tl.make_block_ptr(
        v_ptr + (batch_offset * num_kv_heads + kv_head_idx) * feature_dim,
        (seq_len, feature_dim), (num_kv_heads * feature_dim, 1),
        (kv_block_id * kv_block_size, 0), (kv_block_size, feature_dim), (1, 0)
    )

    grad_k_ptr = tl.make_block_ptr(
        grad_k_ptr + (batch_offset * num_kv_heads + kv_head_idx) * feature_dim,
        (seq_len, feature_dim), (num_kv_heads * feature_dim, 1),
        (kv_block_id * kv_block_size, 0), (kv_block_size, feature_dim), (1, 0)
    )

    grad_v_ptr = tl.make_block_ptr(
        grad_v_ptr + (batch_offset * num_kv_heads + kv_head_idx) * feature_dim,
        (seq_len, feature_dim), (num_kv_heads * feature_dim, 1),
        (kv_block_id * kv_block_size, 0), (kv_block_size, feature_dim), (1, 0)
    )

    k_block = tl.load(k_block_ptr).to(tl.float16)
    v_block = tl.load(v_block_ptr).to(tl.float16)

    grad_k_accum = tl.zeros([kv_block_size, feature_dim], dtype=tl.float32)
    grad_v_accum = tl.zeros([kv_block_size, feature_dim], dtype=tl.float32)

    for q_block_id in range(num_blocks_in_seq):
        is_connected = tl.load(block_mask_ptr + fine_mask_start + q_block_id)

        if is_connected:
            for tile_in_block in range(tiles_per_block):
                tile_idx = q_block_id * tiles_per_block + tile_in_block
                q_tile_start = tile_idx * q_tile_size

                q_tile_ptr = tl.make_block_ptr(
                    base=q_ptr + (batch_offset + q_tile_start) * num_q_heads * feature_dim,
                    shape=(q_tile_size, num_q_heads, feature_dim),
                    strides=(num_q_heads * feature_dim, feature_dim, 1),
                    offsets=(0, kv_head_idx * q_heads_per_kv_head, 0),
                    block_shape=(q_tile_size, q_heads_per_kv_head, feature_dim),
                    order=(0, 1, 2),
                )

                grad_o_tile_ptr = tl.make_block_ptr(
                    base=grad_o_ptr + (batch_offset + q_tile_start) * num_q_heads * feature_dim,
                    shape=(q_tile_size, num_q_heads, feature_dim),
                    strides=(num_q_heads * feature_dim, feature_dim, 1),
                    offsets=(0, kv_head_idx * q_heads_per_kv_head, 0),
                    block_shape=(q_tile_size, q_heads_per_kv_head, feature_dim),
                    order=(0, 1, 2),
                )

                lse_tile_ptr = tl.make_block_ptr(
                    base=lse_ptr + (batch_offset + q_tile_start) * num_q_heads,
                    shape=(q_tile_size, num_q_heads),
                    strides=(num_q_heads, 1),
                    offsets=(0, kv_head_idx * q_heads_per_kv_head),
                    block_shape=(q_tile_size, q_heads_per_kv_head),
                    order=(1, 0),
                )

                delta_tile_ptr = tl.make_block_ptr(
                    base=delta_ptr + (batch_offset + q_tile_start) * num_q_heads,
                    shape=(q_tile_size, num_q_heads),
                    strides=(num_q_heads, 1),
                    offsets=(0, kv_head_idx * q_heads_per_kv_head),
                    block_shape=(q_tile_size, q_heads_per_kv_head),
                    order=(1, 0),
                )

                q_tile = tl.load(q_tile_ptr) * softmax_scale * LOG2_E
                q_tile = tl.reshape(q_tile, (q_tile_size * q_heads_per_kv_head, feature_dim))
                q_tile = q_tile.to(tl.float16)

                grad_o_block = tl.load(grad_o_tile_ptr)
                grad_o_block = tl.reshape(grad_o_block, (q_tile_size * q_heads_per_kv_head, feature_dim))
                grad_o_block = grad_o_block.to(tl.float16)

                lse_vals = tl.load(lse_tile_ptr)
                lse_vals = tl.reshape(lse_vals, (q_tile_size * q_heads_per_kv_head,))

                delta_vals = tl.load(delta_tile_ptr)
                delta_vals = tl.reshape(delta_vals, (q_tile_size * q_heads_per_kv_head,))

                attention_scores = tl.dot(k_block, tl.trans(q_tile))
                attention_probs = tl.exp2(attention_scores - lse_vals[None, :])
                grad_v_accum += tl.dot(attention_probs.to(tl.float16), grad_o_block)
                grad_times_v = tl.dot(v_block, tl.trans(grad_o_block))
                grad_scores = attention_probs * (grad_times_v - delta_vals[None, :]) * LN_2
                grad_k_accum += tl.dot(grad_scores.to(tl.float16), q_tile)

    tl.store(grad_k_ptr, grad_k_accum.to(grad_k_ptr.dtype.element_ty))
    tl.store(grad_v_ptr, grad_v_accum.to(grad_v_ptr.dtype.element_ty))


def mosaic_attn_bwd(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    output: torch.Tensor,
    lse: torch.Tensor,
    grad_o: torch.Tensor,
    softmax_scale: float,
    block_indices: torch.LongTensor,
    block_size: int,
):
    batch_size, seq_len, num_kv_heads, feature_dim = k.shape
    num_q_heads = q.shape[2]
    
    num_kv_blocks_per_q_block = block_indices.shape[-1]
    q_heads_per_kv_head = num_q_heads // num_kv_heads
    num_blocks_in_seq = seq_len // block_size

    grad_q = torch.empty_like(q)
    grad_k = torch.empty_like(k)
    grad_v = torch.empty_like(v)

    block_mask = mosaic_block_mask(block_indices)

    delta = (output * grad_o).sum(dim=-1)

    grid_dq = lambda META: (
        triton.cdiv(seq_len, META['q_tile_size']),
        q_heads_per_kv_head,
        batch_size * num_kv_heads
    )

    mosaic_attn_bwd_q_kernel[grid_dq](
        q_ptr=q,
        k_ptr=k,
        v_ptr=v,
        lse_ptr=lse,
        delta_ptr=delta,
        grad_o_ptr=grad_o,
        grad_q_ptr=grad_q,
        block_indices_ptr=block_indices,
        softmax_scale=softmax_scale,
        seq_len=seq_len,
        num_kv_heads=num_kv_heads,
        num_q_heads=num_q_heads,
        q_heads_per_kv_head=q_heads_per_kv_head,
        feature_dim=feature_dim,
        kv_block_size=block_size,
        num_kv_blocks_per_q_block=num_kv_blocks_per_q_block,
        q_tile_size=16
    )

    grid_dkv = (num_blocks_in_seq, batch_size * num_kv_heads)

    mosaic_attn_bwd_kv_kernel[grid_dkv](
        q_ptr=q,
        k_ptr=k,
        v_ptr=v,
        lse_ptr=lse,
        delta_ptr=delta,
        grad_o_ptr=grad_o,
        grad_k_ptr=grad_k,
        grad_v_ptr=grad_v,
        block_mask_ptr=block_mask,
        softmax_scale=softmax_scale,
        seq_len=seq_len,
        num_kv_heads=num_kv_heads,
        num_q_heads=num_q_heads,
        q_heads_per_kv_head=q_heads_per_kv_head,
        feature_dim=feature_dim,
        kv_block_size=block_size,
        q_tile_size=16
    )

    return grad_q, grad_k, grad_v

class MosaicBlock(nn.Module):
    def __init__(self, config, block_attn_only: bool, no_compression: bool = False):
        super().__init__()
        dim = config.dim
        noise_dim = config.noise_dim
        mlp_ratio = config.mlp_ratio

        self.attention = MosaicAttention(config, block_attn_only, no_compression)
        self.norm1 = RMSNorm(dim, elementwise_affine=config.rmsnorm_elementwise_affine)
        self.norm2 = RMSNorm(dim, elementwise_affine=config.rmsnorm_elementwise_affine)
        self.ffn = cSwiGLU(dim, int(dim * mlp_ratio), noise_dim)

    def forward(self, x: torch.Tensor, z: torch.Tensor = None):
        x = x + self.attention(self.norm1(x))
        x = x + self.ffn(self.norm2(x), z)
        return x
        
class cSwiGLU(nn.Module):
    def __init__(self, dim: int, hidden_dim: int, noise_dim: int):
        super().__init__()
        self.w13 = nn.Linear(dim, 2 * hidden_dim, bias=False)
        self.w2 = nn.Linear(hidden_dim, dim, bias=False)
        self.act_fn = nn.SiLU()
        self.norm = nn.LayerNorm(dim)
        self.perturb_scale = 0.1

        if noise_dim > 0:
            self.noise_bias = nn.Linear(noise_dim, hidden_dim, bias=False)

        with torch.no_grad():
            self.w2.weight.mul_(0.1)
            if noise_dim > 0:
                self.noise_bias.weight.mul_(0.1)


    def forward(self, x: torch.Tensor, z: torch.Tensor = None):
        noise = self.noise_bias(z).unsqueeze(1) if z is not None else 0
    
        x1, x3 = self.w13(x).chunk(2, dim=-1)
        out = self.w2(self.act_fn(x1 + noise) * x3)
        out = self.norm(out)
        out = self.perturb_scale * out
        
        return out