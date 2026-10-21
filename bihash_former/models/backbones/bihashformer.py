# Copyright (c) 2026 BiHashFormer authors.
# Swin patch embedding/merging and initialization: Copyright (c) OpenMMLab.
# SPDX-License-Identifier: Apache-2.0
"""BiHashFormer for high-resolution, wide-field object detection.

See docs/implementation.md for module definitions and training conventions.
"""
import math

import torch
from torch import nn
from torch.nn import functional as F
from torch.utils.checkpoint import checkpoint

from mmdet.models.backbones.swin import SwinTransformer
from mmdet.registry import MODELS


def binary_code_ste(z):
    """Hard 0/1 codes, including at zero; dz uses the paper's 1/2 STE."""
    hard = (z > 0).to(z.dtype)
    return hard + 0.5 * (z - z.detach())


def pairwise_hamming(a, b):
    """Packed 32-bit XOR/popcount forward, with a continuous STE backward.

    Leading batch dimensions are broadcast. The final two dimensions are
    (tokens, bits). Multiple words support the 64-bit ablation as well.
    """
    count = None
    for start in range(0, a.shape[-1], 32):
        size = min(32, a.shape[-1] - start)
        shifts = torch.arange(size, device=a.device, dtype=torch.int64)
        aa = (a[..., start:start + size].detach().long() << shifts).sum(-1)
        bb = (b[..., start:start + size].detach().long() << shifts).sum(-1)
        x = aa.unsqueeze(-1) ^ bb.unsqueeze(-2)
        x = x - ((x >> 1) & 0x55555555)
        x = (x & 0x33333333) + ((x >> 2) & 0x33333333)
        x = (x + (x >> 4)) & 0x0F0F0F0F
        x = x + (x >> 8)
        x = x + (x >> 16)
        word_count = x & 0x3F
        count = word_count if count is None else count + word_count
    hard = count.to(torch.float32)
    if torch.is_grad_enabled() and (a.requires_grad or b.requires_grad):
        # Exact on bits; this branch supplies the backward surrogate only.
        with torch.autocast(device_type=a.device.type, enabled=False):
            af, bf = a.float(), b.float()
            soft = af.sum(-1, keepdim=True) + bf.sum(-1).unsqueeze(-2) - 2 * (af @ bf.transpose(-2, -1))
        return hard + soft - soft.detach()
    return hard


def partition(x, w):
    b, h, width, c = x.shape
    return x.reshape(b, h // w, w, width // w, w, c).permute(0, 1, 3, 2, 4, 5).reshape(b, -1, w * w, c)


def reverse(windows, h, width, w):
    b, _, _, c = windows.shape
    return windows.reshape(b, h // w, width // w, w, w, c).permute(0, 1, 3, 2, 4, 5).reshape(b, h, width, c)


class PrototypeCodes(nn.Module):
    def __init__(self, num_bits=32, num_prototypes=3, radius=4):
        super().__init__()
        if not 1 <= num_prototypes <= num_bits:
            raise ValueError('Require 1 <= num_prototypes <= num_bits')
        self.num_bits, self.radius = num_bits, radius
        # Latents must be +/-1: zeros would binarize to 0.5 with sign(0).
        latents = -torch.ones(num_prototypes, num_bits)
        for i in range(num_prototypes):
            latents[i, i * num_bits // num_prototypes:(i + 1) * num_bits // num_prototypes] = 1
        self.latents = nn.Parameter(latents)

    def bits(self):
        return binary_code_ste(self.latents)

    def loss(self, key_bits):
        centers = self.bits()
        if key_bits.numel() == 0:
            intra = self.latents.sum() * 0
        else:
            intra = pairwise_hamming(key_bits.reshape(-1, self.num_bits), centers).min(-1).values.mean() / self.num_bits
        if centers.shape[0] == 1:
            cross = centers.sum() * 0
        else:
            distances = pairwise_hamming(centers, centers)
            mask = ~torch.eye(len(centers), dtype=torch.bool, device=centers.device)
            cross = F.relu(2 * self.radius + 1 - distances[mask]).mean()
        return intra + cross

    def retrieve(self, key_bits):
        # Nearest center: union of radius-r balls, as in the rebuttal/code.
        return pairwise_hamming(key_bits, self.bits().detach()).min(-1).values <= self.radius


class HashAttention(nn.Module):
    """Per-query nearest-half routing, attention in continuous feature space.

    The 32-bit routing code is formed from each complete token's Q/K vector;
    its selected key indices are shared by attention heads.
    """
    def __init__(self, dim, num_heads, window_size=21, num_bits=32,
                 kv_ratio=0.5, num_prototypes=3, radius=4,
                 attn_drop=0., query_chunk_size=64, routing_ste=True):
        super().__init__()
        if dim % num_heads or not 0 < kv_ratio <= 1:
            raise ValueError('Invalid head dimensions or KV ratio')
        self.dim, self.num_heads = dim, num_heads
        self.num_bits, self.kv_ratio = num_bits, kv_ratio
        self.scale = (dim // num_heads) ** -0.5
        self.query_chunk_size = query_chunk_size
        self.routing_ste = routing_ste
        self.qkv = nn.Linear(dim, 3 * dim)
        self.query_projector = nn.Linear(dim, num_bits, bias=False)
        self.key_projector = nn.Linear(dim, num_bits, bias=False)
        self.prototypes = PrototypeCodes(num_bits, num_prototypes, radius)
        self.proj = nn.Linear(dim, dim)
        self.dropout = nn.Dropout(attn_drop)
        w = window_size
        coords = torch.stack(torch.meshgrid(torch.arange(w), torch.arange(w), indexing='ij')).flatten(1)
        relative = coords[:, :, None] - coords[:, None, :]
        index = (relative[0] + w - 1) * (2 * w - 1) + relative[1] + w - 1
        self.register_buffer('relative_index', index, persistent=False)
        self.relative_bias = nn.Parameter(torch.zeros((2 * w - 1) ** 2, num_heads))

    def route(self, q_bits, k_bits, valid, regions):
        distance = pairwise_hamming(q_bits, k_bits)
        allowed = valid[:, None, :] & (regions[:, :, None] == regions[:, None, :])
        distance = distance.masked_fill(~allowed, self.num_bits + 1e4)
        topk = max(1, int(q_bits.shape[-2] * self.kv_ratio))
        return distance.topk(topk, dim=-1, largest=False), allowed

    def forward(self, x, valid, regions):
        b, n, c = x.shape
        q, k, v = self.qkv(x).chunk(3, -1)
        qb = binary_code_ste(self.query_projector(q))
        kb = binary_code_ste(self.key_projector(k))
        (selected_distances, indices), allowed = self.route(qb, kb, valid, regions)
        qh, kh, vh = [t.reshape(b, n, self.num_heads, c // self.num_heads).transpose(1, 2) for t in (q, k, v)]
        bias = self.relative_bias[self.relative_index].permute(2, 0, 1)
        batch = torch.arange(b, device=x.device)[:, None, None, None]
        head = torch.arange(self.num_heads, device=x.device)[None, :, None, None]
        chunks = []
        for start in range(0, n, self.query_chunk_size):
            end = min(start + self.query_chunk_size, n)
            idx = indices[:, start:end]
            selected_k = kh[batch, head, idx[:, None]]
            selected_v = vh[batch, head, idx[:, None]]
            logits = (qh[:, :, start:end, None] * selected_k).sum(-1) * self.scale
            local_bias = bias[:, start:end].unsqueeze(0).expand(b, -1, -1, -1)
            logits = logits + local_bias.gather(-1, idx[:, None].expand(-1, self.num_heads, -1, -1))
            if self.training and self.routing_ste:
                # Forward remains Eq. (12). Hard top-k indices alone have no
                # derivative; use selected Hamming affinity as an STE surrogate.
                affinity = -selected_distances[:, start:end] / self.num_bits
                logits = logits + (affinity - affinity.detach())[:, None]
            permitted = allowed[:, start:end].gather(-1, idx)
            logits = logits.masked_fill(~permitted[:, None], torch.finfo(logits.dtype).min)
            weights = self.dropout(logits.softmax(-1))
            chunks.append((weights[..., None] * selected_v).sum(-2))
        out = torch.cat(chunks, dim=2).transpose(1, 2).reshape(b, n, c)
        out = self.proj(out) * valid[..., None]
        if self.training:
            seen = torch.zeros(b, n, dtype=torch.bool, device=x.device)
            seen.scatter_(1, indices.reshape(b, -1), True)
            loss = self.prototypes.loss(kb[seen & valid])
        else:
            loss = x.new_zeros(())
        return out, loss


class HashMiner(nn.Module):
    """Radius-based non-RoI retrieval followed by attention on retained tokens."""
    def __init__(self, dim, num_heads, mlp_ratio=4, attn_drop=0.):
        super().__init__()
        self.norm1, self.norm2 = nn.LayerNorm(dim), nn.LayerNorm(dim)
        self.attn = nn.MultiheadAttention(dim, num_heads, dropout=attn_drop, batch_first=True)
        self.ffn = nn.Sequential(nn.Linear(dim, int(dim * mlp_ratio)), nn.GELU(), nn.Linear(int(dim * mlp_ratio), dim))

    def forward(self, x, valid, regions, compression, compression_norm):
        with torch.no_grad():
            normalized = compression_norm(x)
            c = compression.dim
            # Reuse the compression key transform/projector without updating
            # them from the compensation branch. Compression still trains them.
            keys = F.linear(normalized, compression.qkv.weight[c:2 * c], compression.qkv.bias[c:2 * c])
            bits = binary_code_ste(compression.key_projector(keys))
            selected = compression.prototypes.retrieve(bits) & valid
        out = x.clone()
        counts = selected.sum(-1)
        for count in counts.unique().tolist():
            if count == 0:
                continue
            windows = (counts == count).nonzero(as_tuple=True)[0]
            positions = selected[windows].nonzero(as_tuple=False)[:, 1].reshape(-1, count)
            tokens = x[windows[:, None], positions]
            labels = regions[windows[:, None], positions]
            blocked = labels[:, :, None] != labels[:, None, :]
            blocked = blocked[:, None].expand(-1, self.attn.num_heads, -1, -1).reshape(-1, count, count)
            normed = self.norm1(tokens)
            update, _ = self.attn(normed, normed, normed, attn_mask=blocked, need_weights=False)
            tokens = tokens + update
            tokens = tokens + self.ffn(self.norm2(tokens))
            out[windows[:, None], positions] = tokens
        return out, selected


class HashBlock(nn.Module):
    def __init__(self, dim, num_heads, window_size=21, shift=False,
                 mlp_ratio=4, num_bits=32, kv_ratio=.5, num_prototypes=3,
                 radius=4, attn_drop=0., drop_path=0., window_chunk_size=4,
                 query_chunk_size=64, routing_ste=True, with_cp=False):
        super().__init__()
        from mmcv.cnn.bricks.drop import DropPath
        self.w, self.shift = window_size, window_size // 2 if shift else 0
        self.score_net = nn.Linear(window_size * window_size * dim, 1)
        self.norm1, self.norm2 = nn.LayerNorm(dim), nn.LayerNorm(dim)
        self.compression = HashAttention(dim, num_heads, window_size, num_bits, kv_ratio, num_prototypes, radius, attn_drop, query_chunk_size, routing_ste)
        self.compensation = HashMiner(dim, num_heads, mlp_ratio, attn_drop)
        self.ffn = nn.Sequential(nn.Linear(dim, int(dim * mlp_ratio)), nn.GELU(), nn.Linear(int(dim * mlp_ratio), dim))
        self.drop_path = DropPath(drop_path) if drop_path else nn.Identity()
        self.window_chunk_size, self.with_cp = window_chunk_size, with_cp
        self.last_proto_loss = None

    def forward(self, x, hw_shape, ratios):
        b, _, c = x.shape
        h, width = hw_shape
        w, shift = self.w, self.shift
        padded = F.pad(x.reshape(b, h, width, c), (0, 0, 0, (-width) % w, 0, (-h) % w))
        hp, wp = padded.shape[1:3]
        valid = torch.zeros(b, hp, wp, 1, dtype=torch.bool, device=x.device)
        valid[:, :h, :width] = True
        regions = torch.zeros(1, hp, wp, 1, device=x.device, dtype=torch.long)
        if shift:
            label = 0
            for hs in (slice(0, -w), slice(-w, -shift), slice(-shift, None)):
                for ws in (slice(0, -w), slice(-w, -shift), slice(-shift, None)):
                    regions[:, hs, ws] = label
                    label += 1
            padded = torch.roll(padded, (-shift, -shift), (1, 2))
            valid = torch.roll(valid, (-shift, -shift), (1, 2))
        windows = partition(padded, w)
        masks = partition(valid, w).squeeze(-1)
        labels = partition(regions, w).squeeze(-1).expand(b, -1, -1)
        pooled = F.avg_pool2d(padded.permute(0, 3, 1, 2), w, w)
        mean = F.interpolate(pooled, size=(hp, wp), mode='bilinear', align_corners=False).permute(0, 2, 3, 1)
        residual = partition(padded - mean, w) * masks[..., None]
        scores = self.score_net(residual.flatten(-2)).squeeze(-1).softmax(-1)
        out = windows.clone()
        proto_losses = []
        for image in range(b):
            count = min(windows.shape[1], max(1, math.ceil(float(ratios[image].detach()) * windows.shape[1])))
            roi = scores[image].topk(count).indices
            roi_mask = torch.zeros(windows.shape[1], dtype=torch.bool, device=x.device)
            roi_mask[roi] = True
            for chunk in roi.split(self.window_chunk_size):
                source = windows[image, chunk]
                mask, region = masks[image, chunk], labels[image, chunk]
                def attend(tokens, token_mask, token_region):
                    update, aux = self.compression(self.norm1(tokens), token_mask, token_region)
                    tokens = tokens + self.drop_path(update)
                    return tokens + self.drop_path(self.ffn(self.norm2(tokens))), aux
                if self.with_cp and self.training:
                    result, loss = checkpoint(attend, source, mask, region, use_reentrant=False)
                else:
                    result, loss = attend(source, mask, region)
                # SparseFormer's window-score STE: identity in the forward pass.
                gain = scores[image, chunk, None, None]
                out[image, chunk] = source + (result - source) * (1 + gain - gain.detach())
                proto_losses.append((loss, len(chunk)))
            other = (~roi_mask).nonzero(as_tuple=True)[0]
            if len(other):
                result, _ = self.compensation(windows[image, other], masks[image, other], labels[image, other], self.compression, self.norm1)
                out[image, other] = result
        total = sum(n for _, n in proto_losses)
        self.last_proto_loss = sum(loss * n for loss, n in proto_losses) / total
        spatial = reverse(out, hp, wp, w)
        if shift:
            spatial = torch.roll(spatial, (shift, shift), (1, 2))
        return spatial[:, :h, :width].reshape(b, h * width, c)


class BiHashStage(nn.Module):
    def __init__(self, dim, num_heads, depth, downsample, pool_size, roi_hidden,
                 block_options, drop_paths):
        super().__init__()
        self.pool = nn.AdaptiveAvgPool2d(pool_size)
        self.ratio_predictor = nn.Sequential(nn.Linear(dim * pool_size ** 2, roi_hidden), nn.ReLU(), nn.Linear(roi_hidden, 1), nn.Sigmoid())
        self.blocks = nn.ModuleList([HashBlock(dim, num_heads, shift=(i % 2 == 1), drop_path=drop_paths[i], **block_options) for i in range(depth)])
        self.downsample, self.last_ratios = downsample, None

    def forward(self, x, hw_shape):
        b, _, c = x.shape
        self.last_ratios = self.ratio_predictor(self.pool(x.transpose(1, 2).reshape(b, c, *hw_shape)).flatten(1)).squeeze(-1)
        for block in self.blocks:
            x = block(x, hw_shape, self.last_ratios)
        if self.downsample is None:
            return x, hw_shape, x, hw_shape
        down, down_hw = self.downsample(x, hw_shape)
        return down, down_hw, x, hw_shape


@MODELS.register_module()
class BiHashFormer(SwinTransformer):
    """Four-stage BiHashFormer, retaining MMDetection's Swin feature interface."""
    def __init__(self, embed_dims=96, depths=(2, 2, 6, 2),
                 num_heads=(3, 6, 12, 24), window_size=21, mlp_ratio=4,
                 num_bits=32, kv_ratio=.5, num_prototypes=3, radius=4,
                 roi_pool_sizes=(32, 16, 8, 4), roi_hidden=128,
                 select_gamma=.1, attn_drop_rate=0., drop_path_rate=.1,
                 window_chunk_size=4, query_chunk_size=64,
                 routing_ste=True, with_cp=False, **kwargs):
        super().__init__(embed_dims=embed_dims, depths=depths, num_heads=num_heads,
                         window_size=window_size, mlp_ratio=mlp_ratio,
                         attn_drop_rate=attn_drop_rate, drop_path_rate=drop_path_rate,
                         with_cp=with_cp, **kwargs)
        if len(roi_pool_sizes) != len(depths):
            raise ValueError('One RoI pooling size is required per stage')
        drops = torch.linspace(0, drop_path_rate, sum(depths)).tolist()
        options = dict(window_size=window_size, mlp_ratio=mlp_ratio,
                       num_bits=num_bits, kv_ratio=kv_ratio,
                       num_prototypes=num_prototypes, radius=radius,
                       attn_drop=attn_drop_rate, window_chunk_size=window_chunk_size,
                       query_chunk_size=query_chunk_size, routing_ste=routing_ste,
                       with_cp=with_cp)
        old_stages = self.stages
        stages, offset = [], 0
        for i, depth in enumerate(depths):
            stages.append(BiHashStage(embed_dims * 2 ** i, num_heads[i], depth,
                                      old_stages[i].downsample, roi_pool_sizes[i],
                                      roi_hidden, options, drops[offset:offset + depth]))
            offset += depth
        self.stages = nn.ModuleList(stages)
        self.select_gamma = select_gamma

    def auxiliary_losses(self, foreground_ratios):
        selection = torch.stack([((s.last_ratios - foreground_ratios) ** 2 + self.select_gamma * s.last_ratios).mean() for s in self.stages]).mean()
        prototype = torch.stack([block.last_proto_loss for stage in self.stages for block in stage.blocks]).mean()
        return dict(loss_select=selection, loss_proto=prototype)
