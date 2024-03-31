import math
from typing import Tuple

import torch
import torch.nn as nn
from torch.nn import functional as F


# Taken from facebookresearch/llama/model.py
def reshape_for_broadcast(freqs_cis: torch.Tensor, x: torch.Tensor) -> torch.Tensor:
    ndim = x.ndim
    assert 0 <= 1 < ndim
    assert freqs_cis.shape == (x.shape[1], x.shape[-1])
    shape = [d if i == 1 or i == ndim - 1 else 1 for i, d in enumerate(x.shape)]
    return freqs_cis.view(*shape)


# Taken from facebookresearch/llama/model.py
def apply_rotary_emb(
    xq: torch.Tensor,
    xk: torch.Tensor,
    freqs_cis: torch.Tensor,
) -> Tuple[torch.Tensor, torch.Tensor]:
    xq_ = torch.view_as_complex(xq.float().reshape(*xq.shape[:-1], -1, 2))
    xk_ = torch.view_as_complex(xk.float().reshape(*xk.shape[:-1], -1, 2))

    T_q = xq.shape[1]
    freqs_cis_q = reshape_for_broadcast(freqs_cis[:T_q], xq_)
    xq_out = torch.view_as_real(xq_ * freqs_cis_q).flatten(3)

    T_k = xk.shape[1]
    freqs_cis_k = reshape_for_broadcast(freqs_cis[:T_k], xk_)
    xk_out = torch.view_as_real(xk_ * freqs_cis_k).flatten(3)

    return xq_out.type_as(xq), xk_out.type_as(xk)


# Taken from facebookresearch/llama/model.py
def precompute_freqs_cis(dim: int, end: int, theta: float = 10000.0) -> torch.Tensor:
    freqs = 1.0 / (theta ** (torch.arange(0, dim, 2)[: (dim // 2)].float() / dim))
    t = torch.arange(end, device=freqs.device)
    freqs = torch.outer(t, freqs).float()
    freqs_cis = torch.polar(torch.ones_like(freqs), freqs)  # complex64

    return freqs_cis


class LayerNorm(nn.Module):
    """LayerNorm but with an optional bias. PyTorch doesn't support simply bias=False"""

    def __init__(self, ndim: int, bias: bool):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(ndim))
        self.bias = nn.Parameter(torch.zeros(ndim)) if bias else None

    def forward(self, input: torch.Tensor) -> torch.Tensor:
        return F.layer_norm(input, self.weight.shape, self.weight, self.bias, 1e-5)


class MultiheadAttention(nn.Module):
    def __init__(
        self,
        embed_dim: int,
        n_head: int,
        bias: bool,
        dropout: float,
        block_size: int,
        causal: bool,
    ):
        super().__init__()
        assert embed_dim % n_head == 0
        self.n_head = n_head
        self.n_embd = embed_dim
        self.dropout = dropout
        self.causal = causal

        # key, query, value projections for all heads, but in a batch
        self.q_proj = nn.Linear(embed_dim, embed_dim, bias=bias)
        self.k_proj = nn.Linear(embed_dim, embed_dim, bias=bias)
        self.v_proj = nn.Linear(embed_dim, embed_dim, bias=bias)
        # output projection
        self.output_proj = nn.Linear(embed_dim, embed_dim, bias=bias)
        # regularization
        self.attn_dropout = nn.Dropout(dropout)
        self.resid_dropout = nn.Dropout(dropout)
        # flash attention make GPU go brrrrr but support is only in PyTorch >= 2.0
        self.flash = hasattr(torch.nn.functional, "scaled_dot_product_attention")
        if not self.flash:
            print(
                "WARNING: using slow attention. Flash Attention requires PyTorch >= 2.0"
            )
            if self.causal:
                self.register_buffer(
                    "attn_mask",
                    torch.tril(torch.ones(block_size, block_size)).view(
                        1, 1, block_size, block_size
                    ),
                )
            else:
                self.register_buffer(
                    "attn_mask",
                    torch.ones(block_size, block_size).view(
                        1, 1, block_size, block_size
                    ),
                )

    def forward(
        self, x: torch.Tensor, mem: torch.Tensor, freqs_cis: torch.Tensor
    ) -> torch.Tensor:
        B, T_x, C = x.size()
        B, T_mem, C = mem.size()

        q = self.q_proj(x).view(B, T_x, self.n_head, C // self.n_head)
        k = self.k_proj(mem).view(B, T_mem, self.n_head, C // self.n_head)
        v = self.v_proj(mem).view(B, T_mem, self.n_head, C // self.n_head)

        q, k = apply_rotary_emb(q, k, freqs_cis=freqs_cis)

        q = q.transpose(1, 2)
        k = k.transpose(1, 2)
        v = v.transpose(1, 2)

        # Self-attend: (B, nh, T, hs) x (B, nh, hs, T) -> (B, nh, T, T)
        if self.flash:
            # efficient attention using Flash Attention CUDA kernels
            y = torch.nn.functional.scaled_dot_product_attention(
                q,
                k,
                v,
                attn_mask=None,
                dropout_p=self.dropout if self.training else 0,
                is_causal=self.causal,
            )
        else:
            # manual implementation of attention
            att = (q @ k.transpose(-2, -1)) * (1.0 / math.sqrt(k.size(-1)))
            att = att.masked_fill(self.bias[:, :, :T_x, :T_mem] == 0, float("-inf"))
            att = F.softmax(att, dim=-1)
            att = self.attn_dropout(att)
            y = att @ v  # (B, nh, T, T) x (B, nh, T, hs) -> (B, nh, T, hs)
        y = (
            y.transpose(1, 2).contiguous().view(B, T_x, C)
        )  # re-assemble all head outputs side by side

        # output projection
        y = self.resid_dropout(self.output_proj(y))
        return y


class MLP(nn.Module):
    def __init__(
        self,
        embed_dim: int,
        bias: bool,
        dropout: float,
    ):
        super().__init__()
        self.c_fc = nn.Linear(embed_dim, 4 * embed_dim, bias=bias)
        self.gelu = nn.GELU()
        self.c_proj = nn.Linear(4 * embed_dim, embed_dim, bias=bias)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        x = self.c_fc(x)
        x = self.gelu(x)
        x = self.c_proj(x)
        x = self.dropout(x)
        return x


class EncoderBlock(nn.Module):
    def __init__(
        self,
        embed_dim: int,
        n_head: int,
        bias: bool,
        dropout: float,
        block_size: int,
        causal: bool,
    ):
        super().__init__()
        self.ln_1 = LayerNorm(embed_dim, bias=bias)
        self.attn = MultiheadAttention(
            embed_dim=embed_dim,
            n_head=n_head,
            bias=bias,
            dropout=dropout,
            block_size=block_size,
            causal=causal,
        )
        self.ln_2 = LayerNorm(embed_dim, bias=bias)
        self.mlp = MLP(
            embed_dim=embed_dim,
            bias=bias,
            dropout=dropout,
        )

    def forward(self, x: torch.Tensor, freqs_cis: torch.Tensor) -> torch.Tensor:
        x_norm = self.ln_1(x)
        x = x + self.attn(x_norm, x_norm, freqs_cis)
        x = x + self.mlp(self.ln_2(x))
        return x


class Encoder(nn.Module):
    def __init__(
        self,
        n_layers: int,
        embed_dim: int,
        n_head: int,
        bias: bool,
        dropout: float,
        block_size: int,
        causal: bool,
    ):
        super().__init__()
        self.layers = nn.ModuleList(
            [
                EncoderBlock(
                    embed_dim=embed_dim,
                    n_head=n_head,
                    bias=bias,
                    dropout=dropout,
                    block_size=block_size,
                    causal=causal,
                )
                for _ in range(n_layers)
            ]
        )

    def forward(self, x: torch.Tensor, freqs_cis: torch.Tensor) -> torch.Tensor:
        for layer in self.layers:
            x = layer(x, freqs_cis)
        return x


class DecoderBlock(nn.Module):
    def __init__(
        self,
        embed_dim: int,
        n_head: int,
        bias: bool,
        dropout: float,
        block_size: int,
        causal: bool,
    ):
        super().__init__()
        self.ln_1 = LayerNorm(embed_dim, bias=bias)
        self.attn_1 = MultiheadAttention(
            embed_dim=embed_dim,
            n_head=n_head,
            bias=bias,
            dropout=dropout,
            block_size=block_size,
            causal=causal,
        )
        self.ln_2 = LayerNorm(embed_dim, bias=bias)
        self.attn_2 = MultiheadAttention(
            embed_dim=embed_dim,
            n_head=n_head,
            bias=bias,
            dropout=dropout,
            block_size=block_size,
            causal=False,
        )
        self.ln_3 = LayerNorm(embed_dim, bias=bias)
        self.mlp = MLP(
            embed_dim=embed_dim,
            bias=bias,
            dropout=dropout,
        )

    def forward(
        self, x: torch.Tensor, mem: torch.Tensor, freqs_cis: torch.Tensor
    ) -> torch.Tensor:
        x_norm = self.ln_1(x)
        x = x + self.attn_1(x_norm, x_norm, freqs_cis)
        x = x + self.attn_2(self.ln_2(x), mem, freqs_cis)
        x = x + self.mlp(self.ln_3(x))
        return x


class Decoder(nn.Module):
    def __init__(
        self,
        input_dim: int,
        output_dim: int,
        n_layers: int,
        embed_dim: int,
        n_head: int,
        bias: bool,
        dropout: float,
        block_size: int,
        causal: bool,
    ):
        super().__init__()
        self.layers = nn.ModuleList(
            [
                DecoderBlock(
                    embed_dim=embed_dim,
                    n_head=n_head,
                    bias=bias,
                    dropout=dropout,
                    block_size=block_size,
                    causal=causal,
                )
                for _ in range(n_layers)
            ]
        )
        self.input_projection = nn.Linear(input_dim, embed_dim)
        self.final_layer_norm = LayerNorm(embed_dim, bias=bias)
        self.output_projection = nn.Linear(embed_dim, output_dim)

    def forward(
        self, x: torch.Tensor, mem: torch.Tensor, freqs_cis: torch.Tensor
    ) -> torch.Tensor:
        for layer in self.layers:
            x = layer(x, mem, freqs_cis)
        x = self.output_projection(self.final_layer_norm(x))
        return x


class Transformer(nn.Module):
    def __init__(
        self,
        input_dim: int,
        output_dim: int,
        n_encoder_layers: int,
        n_decoder_layers: int,
        embed_dim: int,
        n_head: int,
        bias: bool,
        dropout: float,
        block_size: int,
        causal_encoder: bool,
        causal_decoder: bool,
        max_seq_len: int,
    ):
        super().__init__()

        self.input_dim = input_dim
        self.embed_dim = embed_dim
        self.n_head = n_head
        self.bias = bias
        self.dropout = dropout
        self.block_size = block_size
        self.max_seq_len = max_seq_len

        self.input_projection = nn.Linear(input_dim, embed_dim)

        self.encoder = Encoder(
            n_layers=n_encoder_layers,
            embed_dim=embed_dim,
            n_head=n_head,
            bias=bias,
            dropout=dropout,
            block_size=block_size,
            causal=causal_encoder,
        )

        self.decoder = Decoder(
            input_dim=input_dim,
            output_dim=output_dim,
            n_layers=n_decoder_layers,
            embed_dim=embed_dim,
            n_head=n_head,
            bias=bias,
            dropout=dropout,
            block_size=block_size,
            causal=causal_decoder,
        )
        self.freqs_cis = precompute_freqs_cis(embed_dim // n_head, max_seq_len)

        self.register_parameter(
            "start_token", nn.Parameter(torch.randn(1, 1, embed_dim))
        )

        # Init all weights
        self.apply(self._init_weights)
        # Apply special scaled init to the residual projections, per GPT-2 paper
        for pn, p in self.named_parameters():
            if pn.endswith("output_proj.weight"):
                torch.nn.init.normal_(
                    p,
                    mean=0.0,
                    std=0.02 / math.sqrt(n_encoder_layers + n_decoder_layers),
                )

    @property
    def num_params(self) -> int:
        n_params = sum(p.numel() for p in self.parameters())
        return n_params

    def _init_weights(self, module: nn.Module) -> None:
        if isinstance(module, nn.Linear):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)
            if module.bias is not None:
                torch.nn.init.zeros_(module.bias)

    def right_shift_input(self, input: torch.Tensor) -> torch.Tensor:
        start_token = self.start_token.expand(input.shape[0], -1, -1)
        return torch.concat([start_token, input[:, :-1, :]], dim=1)

    def embed_patch(self, input: torch.Tensor) -> torch.Tensor:
        return self.input_projection(input)

    def forward(
        self, input: torch.Tensor, cond: torch.Tensor, start_pos: int = 0
    ) -> torch.Tensor:
        assert input.shape == cond.shape, "Input and condition must have the same shape"
        _, t, _ = input.shape
        assert t <= self.block_size, (
            f"Cannot forward sequence of length {t}, "
            f"block size is only {self.block_size}"
        )

        freqs_cis = self.freqs_cis.to(input.device)
        freqs_cis = freqs_cis[start_pos : start_pos + t]

        # Encode the conditioning information
        cond_feature = self.encoder(cond, freqs_cis)
        # Get the decoded features
        pred = self.decoder(input, cond_feature, freqs_cis)

        return pred

    def estimate_mfu(self, fwdbwd_per_iter: int, dt: float) -> float:
        """estimate model flops utilization (MFU) in units of A100 bfloat16 peak FLOPS"""
        # first estimate the number of flops we do per iteration.
        # see PaLM paper Appendix B as ref: https://arxiv.org/abs/2204.02311
        N = self.get_num_params()
        L, H, Q, T = (
            self.n_encoder_layers + self.n_decoder_layers,
            self.n_head,
            self.embed_dim // self.n_head,
            self.block_size,
        )
        flops_per_token = 6 * N + 12 * L * H * Q * T
        flops_per_fwdbwd = flops_per_token * T
        flops_per_iter = flops_per_fwdbwd * fwdbwd_per_iter
        # express our flops throughput as ratio of A100 bfloat16 peak flops
        flops_achieved = flops_per_iter * (1.0 / dt)  # per second
        flops_promised = 312e12  # A100 GPU bfloat16 peak flops is 312 TFLOPS
        mfu = flops_achieved / flops_promised
        return mfu

    @torch.no_grad()
    def generate(
        self,
        cond: torch.Tensor,
        window_size: int,
        context_size: int,
        start_pos: int = 0,
    ) -> torch.Tensor:
        T = cond.shape[1]

        freqs_cis = self.freqs_cis.to(cond.device)
        freqs_cis = freqs_cis[start_pos : start_pos + T]

        # Encode the conditioning information
        cond_feature = self.encoder(cond, freqs_cis)

        input_feature = self.start_token.expand(cond.shape[0], -1, -1)
        recons = [torch.zeros(cond.shape[0], 2, window_size, device=cond.device)]

        for t in range(1, T + 1):
            recon = self.decoder(input_feature, cond_feature, freqs_cis)

            # Add the new reconstruction
            recons.append(recon[:, -1, :].reshape(-1, 2, window_size))

            feature_next = torch.concat(
                [recons[-2][:, :, -context_size:], recons[-1]], dim=-1
            ).reshape(cond.shape[0], 1, -1)
            input_feature = torch.concat(
                [input_feature, self.input_projection(feature_next)], dim=1
            )

        return torch.stack(recons[1:], dim=1).flatten(start_dim=2)
