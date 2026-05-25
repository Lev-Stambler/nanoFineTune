"""Triton kernels for saved-activation lowpass compression."""

from __future__ import annotations

from typing import Any

import torch

try:
    import triton
    import triton.language as tl
except Exception:  # pragma: no cover - local dev env may not have Triton.
    triton = None
    tl = None


if triton is not None:

    @triton.jit
    def _dense_int8_pack_kernel(
        flattened,
        chunk_starts,
        basis,
        q_out,
        scale_out,
        hidden: tl.constexpr,
        qmax: tl.constexpr,
        chunk_size: tl.constexpr,
        keep: tl.constexpr,
        block_h: tl.constexpr,
    ):
        chunk_id = tl.program_id(0)
        h_offsets = tl.program_id(1) * block_h + tl.arange(0, block_h)
        token_offsets = tl.arange(0, chunk_size)
        keep_offsets = tl.arange(0, keep)

        chunk_start = tl.load(chunk_starts + chunk_id)
        values = tl.load(
            flattened + (chunk_start + token_offsets[:, None]) * hidden + h_offsets[None, :],
            mask=h_offsets[None, :] < hidden,
            other=0.0,
        )
        basis_values = tl.load(basis + keep_offsets[:, None] * chunk_size + token_offsets[None, :])
        coeff = tl.dot(basis_values, values, out_dtype=tl.float32)

        absmax = tl.max(tl.abs(coeff), axis=0)
        scale = tl.maximum(absmax / qmax, 1.0e-20)
        scaled = coeff / scale[None, :]
        rounded = tl.where(scaled >= 0.0, scaled + 0.5, scaled - 0.5).to(tl.int32)
        clipped = tl.minimum(tl.maximum(rounded, -qmax), qmax)

        tl.store(
            q_out + (chunk_id * keep + keep_offsets[:, None]) * hidden + h_offsets[None, :],
            clipped.to(tl.int8),
            mask=h_offsets[None, :] < hidden,
        )
        tl.store(
            scale_out + chunk_id * hidden + h_offsets,
            scale,
            mask=h_offsets < hidden,
        )

    @triton.jit
    def _dense_int8_unpack_kernel(
        q_values,
        scale_values,
        chunk_starts,
        basis,
        restored,
        hidden: tl.constexpr,
        chunk_size: tl.constexpr,
        keep: tl.constexpr,
        block_h: tl.constexpr,
        block_t: tl.constexpr,
    ):
        chunk_id = tl.program_id(0)
        h_offsets = tl.program_id(1) * block_h + tl.arange(0, block_h)
        token_offsets = tl.program_id(2) * block_t + tl.arange(0, block_t)
        keep_offsets = tl.arange(0, keep)

        q = tl.load(
            q_values + (chunk_id * keep + keep_offsets[:, None]) * hidden + h_offsets[None, :],
            mask=h_offsets[None, :] < hidden,
            other=0,
        ).to(tl.float32)
        scale = tl.load(
            scale_values + chunk_id * hidden + h_offsets,
            mask=h_offsets < hidden,
            other=0.0,
        ).to(tl.float32)

        basis_values = tl.load(
            basis + keep_offsets[:, None] * chunk_size + token_offsets[None, :],
            mask=token_offsets[None, :] < chunk_size,
            other=0.0,
        )
        lowpass = (q * scale[None, :]).to(basis_values.dtype)
        decoded = tl.dot(tl.trans(basis_values), lowpass, out_dtype=tl.float32)

        chunk_start = tl.load(chunk_starts + chunk_id)
        tl.store(
            restored + (chunk_start + token_offsets[:, None]) * hidden + h_offsets[None, :],
            decoded,
            mask=(token_offsets[:, None] < chunk_size) & (h_offsets[None, :] < hidden),
        )


def _require_triton() -> Any:
    if triton is None:
        raise RuntimeError("Triton is required for activation_filter_kernel='triton-dense'")
    return triton


def dense_int8_pack(
    flattened: torch.Tensor,
    chunk_starts: torch.Tensor,
    basis: torch.Tensor,
    *,
    chunk_size: int,
    keep: int,
    block_h: int = 64,
) -> tuple[torch.Tensor, torch.Tensor]:
    triton_mod = _require_triton()
    hidden = int(flattened.shape[1])
    n_chunks = int(chunk_starts.numel())
    q_values = torch.empty((n_chunks, keep, hidden), device=flattened.device, dtype=torch.int8)
    scale = torch.empty((n_chunks, 1, hidden), device=flattened.device, dtype=flattened.dtype)
    grid = (n_chunks, triton_mod.cdiv(hidden, block_h))
    _dense_int8_pack_kernel[grid](
        flattened,
        chunk_starts,
        basis,
        q_values,
        scale,
        hidden,
        127,
        chunk_size,
        keep,
        block_h,
        num_warps=4,
    )
    return q_values, scale


def dense_int8_unpack(
    q_values: torch.Tensor,
    scale: torch.Tensor,
    chunk_starts: torch.Tensor,
    basis: torch.Tensor,
    restored: torch.Tensor,
    *,
    chunk_size: int,
    keep: int,
    block_h: int = 64,
    block_t: int = 16,
) -> None:
    triton_mod = _require_triton()
    hidden = int(restored.shape[1])
    n_chunks = int(chunk_starts.numel())
    grid = (n_chunks, triton_mod.cdiv(hidden, block_h), triton_mod.cdiv(chunk_size, block_t))
    _dense_int8_unpack_kernel[grid](
        q_values,
        scale,
        chunk_starts,
        basis,
        restored,
        hidden,
        chunk_size,
        keep,
        block_h,
        block_t,
        num_warps=4,
    )
