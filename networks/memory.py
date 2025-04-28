from typing import Literal, Optional, Union

import torch
from torch import nn

from .positional_embeddings import PositionEmbeddings

class CompressiveMemory(nn.Module):
    def __init__(self, dim_input: int, dim_key: int, dim_value: int, num_heads: int, segment_len: int,
                 sampling_factor: int = None, update: str = "linear", causal: bool = False,
                 position_embedder: nn.Module = None, init_state_learnable: bool = False):
        super(CompressiveMemory, self).__init__()

        self.device = torch.device("cuda:0")  

        self.num_heads = num_heads
        self.segment_len = segment_len
        self.sampling_factor = sampling_factor
        self.dim_input = dim_input
        self.dim_key = dim_key
        self.dim_value = dim_value
        self.update = update
        self.causal = causal
        self.position_embedder = position_embedder

        self.proj_k = nn.Linear(dim_input, num_heads * dim_key, bias=False).to(self.device)
        self.proj_v = nn.Linear(dim_input, num_heads * dim_value, bias=False).to(self.device)
        self.proj_q = nn.Linear(dim_input, num_heads * dim_key, bias=False).to(self.device)
        self.proj_out = nn.Linear(num_heads * dim_value, dim_input, bias=False).to(self.device)

        self.betas = nn.Parameter(torch.randn(1, num_heads, 1, dim_value, device=self.device))

        if init_state_learnable:
            self.init_mem = nn.Parameter(torch.randn(1, self.num_heads, self.dim_key, self.dim_value, device=self.device))
            self.init_z = nn.Parameter(torch.ones(1, self.num_heads, self.dim_key, 1, device=self.device))
        else:
            self.init_mem = None
            self.init_z = None

    def forward(self, x: torch.Tensor, sample_mask: torch.Tensor = None) -> torch.Tensor:
        x = x.to(self.device)  
        batch_size, seq_len, _ = x.shape

        num_segments = (seq_len + self.segment_len - 1)
        out = []

        mem = self.init_mem if self.init_mem is not None else torch.zeros(1, self.num_heads, self.dim_key, self.dim_value, device=self.device)
        z = self.init_z if self.init_z is not None else torch.ones(batch_size, self.num_heads, self.dim_key, 1, device=self.device) / self.dim_key

        k_full = self.proj_k(x).view(batch_size, self.num_heads, seq_len, self.dim_key)
        v_full = self.proj_v(x).view(batch_size, self.num_heads, seq_len, self.dim_value)
        q_full = self.proj_q(x).view(batch_size, self.num_heads, seq_len, self.dim_key)

        for ix in range(num_segments):
            ix_lo, ix_hi = ix * self.segment_len, min((ix + 1) * self.segment_len, seq_len)
            k, v, q = k_full[:, :, ix_lo:ix_hi, :], v_full[:, :, ix_lo:ix_hi, :], q_full[:, :, ix_lo:ix_hi, :]

            sigma_q = (nn.functional.elu(q) + 1.0)

            scores = q @ k.transpose(-2, -1) / self.dim_key ** 0.5

            if self.causal:
                mask = torch.tril(torch.ones((ix_hi - ix_lo, ix_hi - ix_lo), dtype=torch.bool, device=self.device))
                scores.masked_fill_(~mask, float('-inf'))

            att_dot = nn.functional.softmax(scores, dim=-1) @ v
            att_mem = (sigma_q @ mem) / (sigma_q @ z)

            sigma_k = nn.functional.elu(k) + 1.0
            mem = mem + sigma_k.transpose(-2, -1) @ v if self.update == "linear" else mem + sigma_k.transpose(-2, -1) @ (v - (sigma_k @ mem) / (sigma_k @ z))

            z = z + sigma_k.sum(dim=-2, keepdim=True).transpose(-2, -1)

            att = nn.functional.sigmoid(self.betas) * att_mem + (1 - nn.functional.sigmoid(self.betas)) * att_dot
            att = att.view(batch_size, ix_hi - ix_lo, self.num_heads * self.dim_value)

            out.append(self.proj_out(att))

        return torch.cat(out, dim=1)

def test_memory(
        short_seq_len: bool = False,
        even_seq_len: bool = True,
        causal_masking: bool = False,
        update: str = "linear"
) -> None:
    dim_input = 512
    dim_key = 64
    dim_value = 64
    num_heads = 8
    segment_len = 32
    causal = causal_masking
    batch_size = 4
    if short_seq_len:
        seq_len = 16
    else:
        if even_seq_len:
            seq_len = 128
        else:
            seq_len = 144

    model = CompressiveMemory(
        dim_input, dim_key, dim_value, num_heads, segment_len, update, causal)

    batch = torch.randn(batch_size, seq_len, dim_input)

    model(batch)


if __name__ == "__main__":
    print("Testing with short sequence lengths:")

    short_seq_len = True
    even_seq_len = True

    for causal_masking in [True, False]:
        for update in ["linear", "delta"]:
            print(f"  Testing with causal_masking={causal_masking} and update={update}")
            test_compressive_memory(
                short_seq_len=short_seq_len,
                even_seq_len=even_seq_len,
                causal_masking=causal_masking,
                update=update
            )

    print("Testing with non-short sequence lengths:")

    short_seq_len = False

    for even_seq_len in [True, False]:
        for causal_masking in [True, False]:
            for update in ["linear", "delta"]:
                print(
                    f"  Testing with even_seq_len={even_seq_len}, causal_masking={causal_masking} and update={update}")
                test_compressive_memory(
                    short_seq_len=short_seq_len,
                    even_seq_len=even_seq_len,
                    causal_masking=causal_masking,
                    update=update
                )
