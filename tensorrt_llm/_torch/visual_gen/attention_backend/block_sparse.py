# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Generic compact-BSHD block-sparse attention adapter for VisualGen."""

from dataclasses import dataclass
from inspect import signature
from math import ceil
from typing import Literal

import torch

_flashinfer_block_sparse_import_error: Exception | None = None
_BLOCK_SPARSE_PLAN_SUPPORTS_KV_TILE_SIZE = False
try:
    from flashinfer.attention.prims_ts.block_sparse import (
        BlockSparseTSWrapper as _BlockSparseTSWrapper,
    )

    try:
        plan_parameters = signature(_BlockSparseTSWrapper.plan).parameters
    except (TypeError, ValueError) as error:
        raise ImportError("BlockSparseTSWrapper.plan signature is not inspectable") from error
    required_plan_parameters = {"dynamic_metadata", "kv_valid_bits"}
    missing_plan_parameters = required_plan_parameters.difference(plan_parameters)
    if missing_plan_parameters:
        raise ImportError(
            "BlockSparseTSWrapper.plan does not expose the required capabilities: "
            f"{', '.join(sorted(missing_plan_parameters))}"
        )
    _BLOCK_SPARSE_PLAN_SUPPORTS_KV_TILE_SIZE = "kv_tile_size" in plan_parameters
except (AttributeError, ImportError, OSError) as error:
    _BlockSparseTSWrapper = None
    _flashinfer_block_sparse_import_error = error


_SUPPORTED_DTYPES = (torch.float16, torch.bfloat16)
_SUPPORTED_COMPUTE_CAPABILITIES = ((10, 0), (10, 3))


@dataclass(frozen=True)
class _BlockSparsePlanParams:
    """Static traits that select one reusable PrimTS execution plan."""

    device: torch.device
    batch_size: int
    seq_len_q: int
    seq_len_kv: int
    num_qo_heads: int
    num_kv_heads: int
    head_dim: int
    q_block_size: int
    kv_block_size: int
    kv_tile_size: Literal[128, 256] | None
    row_nnz: int
    mask_type: Literal["dense", "causal"]
    q_dtype: torch.dtype
    kv_dtype: torch.dtype
    output_dtype: torch.dtype
    has_kv_token_mask: bool


@dataclass
class _BlockSparsePlan:
    """Wrapper and stable graph-visible metadata for one static plan."""

    wrapper: object
    block_indptr: torch.Tensor
    block_indices: torch.Tensor
    kv_valid_bits: torch.Tensor | None


def is_flashinfer_block_sparse_available() -> bool:
    """Return whether the PrimTS block-sparse wrapper can be imported."""

    return _BlockSparseTSWrapper is not None


def get_flashinfer_block_sparse_import_error() -> Exception | None:
    """Return the optional import failure for diagnostic logging."""

    return _flashinfer_block_sparse_import_error


def _validate_positive_int(value: int, name: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int):
        raise TypeError(f"{name} must be an integer")
    if value <= 0:
        raise ValueError(f"{name} must be positive")
    return value


def _validate_kv_tile_size(value: int | None) -> Literal[128, 256] | None:
    if value is None:
        return None
    if isinstance(value, bool) or not isinstance(value, int):
        raise TypeError("kv_tile_size must be an integer or None")
    if value not in (128, 256):
        raise ValueError("kv_tile_size must be None, 128, or 256")
    return value


def build_block_sparse_indptr(
    batch_size: int,
    num_heads: int,
    num_q_blocks: int,
    row_nnz: int,
    device: torch.device,
) -> torch.Tensor:
    """Build fixed-row canonical BSR offsets with shape ``[B, H, Q + 1]``."""

    batch_size = _validate_positive_int(batch_size, "batch_size")
    num_heads = _validate_positive_int(num_heads, "num_heads")
    num_q_blocks = _validate_positive_int(num_q_blocks, "num_q_blocks")
    row_nnz = _validate_positive_int(row_nnz, "row_nnz")
    total_entries = batch_size * num_heads * num_q_blocks * row_nnz
    if total_entries > torch.iinfo(torch.int32).max:
        raise ValueError("block-sparse indptr offsets exceed int32 capacity")
    row_offsets = torch.arange(
        num_q_blocks + 1,
        dtype=torch.int32,
        device=device,
    ).reshape(1, 1, -1)
    head_offsets = torch.arange(
        batch_size * num_heads,
        dtype=torch.int32,
        device=device,
    ).reshape(batch_size, num_heads, 1)
    return (head_offsets * (num_q_blocks * row_nnz) + row_offsets * row_nnz).contiguous()


def canonicalize_block_sparse_indices(block_indices: torch.Tensor) -> torch.Tensor:
    """Sort raw ``[B, H, Q, K]`` routes into canonical flat BSR order.

    FlashInfer's ``plan()`` performs the bounds and uniqueness validation in
    the same host-synchronizing inspection used for launch selection. Dynamic
    ``run()`` metadata must retain this per-row ordering contract.
    """

    if not isinstance(block_indices, torch.Tensor) or block_indices.ndim != 4:
        raise ValueError("block_indices must have shape [B, H, Q, K]")
    if block_indices.dtype not in (torch.int32, torch.int64):
        raise TypeError("block_indices must have int32 or int64 dtype")
    if any(dimension <= 0 for dimension in block_indices.shape):
        raise ValueError("block_indices dimensions must be non-empty")
    sorted_indices = torch.sort(block_indices.to(torch.int32), dim=-1).values
    return sorted_indices.reshape(-1).contiguous()


def pack_kv_token_mask(kv_token_mask: torch.Tensor) -> torch.Tensor:
    """Pack a boolean ``[B, Skv]`` token mask into LSB-first uint32 words."""

    if not isinstance(kv_token_mask, torch.Tensor) or kv_token_mask.ndim != 2:
        raise ValueError("kv_token_mask must have shape [B, Skv]")
    if kv_token_mask.dtype != torch.bool:
        raise TypeError("kv_token_mask must have bool dtype")
    batch_size, seq_len_kv = map(int, kv_token_mask.shape)
    if batch_size <= 0 or seq_len_kv <= 0:
        raise ValueError("kv_token_mask dimensions must be non-empty")

    padded_length = ceil(seq_len_kv / 32) * 32
    if padded_length == seq_len_kv:
        padded_mask = kv_token_mask
    else:
        padding = torch.zeros(
            (batch_size, padded_length - seq_len_kv),
            dtype=torch.bool,
            device=kv_token_mask.device,
        )
        padded_mask = torch.cat((kv_token_mask, padding), dim=1)
    bit_weights = torch.bitwise_left_shift(
        torch.ones(32, dtype=torch.int64, device=kv_token_mask.device),
        torch.arange(32, dtype=torch.int64, device=kv_token_mask.device),
    )
    words = (padded_mask.reshape(batch_size, -1, 32).to(torch.int64) * bit_weights).sum(dim=-1)
    return words.to(torch.uint32).contiguous()


class FlashInferBlockSparseAttention:
    """Adapt dynamic fixed-row block routes to FlashInfer PrimTS plan/run.

    The public adapter accepts compact BSHD Q/K/V and an optional boolean KV
    token mask. FlashInfer calls the packed form ``kv_valid_bits`` internally.
    Plans are cached by static tensor geometry. Before ``plan()`` and every
    ``run()``, routes are sorted by block ID to satisfy FlashInfer's canonical
    BSR contract. CUDA Graph replay records that sort and reuses stable
    metadata addresses while observing updated route values.

    Dynamic route values are trusted metadata: every row must remain unique
    and in range, although its ordering may change. VSA's ``topk`` output
    satisfies that contract by construction.

    One adapter may be shared by serialized attention layers on the same CUDA
    stream. Concurrent forwards on different streams require separate adapter
    instances because they would otherwise update the same route buffers.
    """

    def __init__(self) -> None:
        self._plans: dict[_BlockSparsePlanParams, _BlockSparsePlan] = {}
        # Retain the most recently used wrapper for compatibility with
        # diagnostics that inspected the original one-plan adapter.
        self._wrapper: object | None = None
        self._indptr_cache: dict[tuple[int, int, int, int, torch.device], torch.Tensor] = {}

    def clear(self) -> None:
        """Release cached plans after all launches and captured graphs are retired."""

        if torch.cuda.is_available() and torch.cuda.is_current_stream_capturing():
            raise RuntimeError("block-sparse plans cannot be cleared during CUDA Graph capture")
        self._plans.clear()
        self._wrapper = None
        self._indptr_cache.clear()

    @property
    def supports_kv_tile_size(self) -> bool:
        """Return whether FlashInfer accepts an explicit physical KV execution tile."""

        return _BLOCK_SPARSE_PLAN_SUPPORTS_KV_TILE_SIZE

    def is_supported(
        self,
        q: torch.Tensor,
        k: torch.Tensor,
        v: torch.Tensor,
        *,
        q_block_size: int,
        kv_block_size: int,
    ) -> bool:
        """Return whether the current wrapper and tensors meet the kernel envelope."""

        if _BlockSparseTSWrapper is None:
            return False
        if q_block_size <= 0 or q_block_size % 64 != 0:
            return False
        if kv_block_size <= 0 or kv_block_size % 64 != 0:
            return False
        if any(tensor.ndim != 4 or not tensor.is_cuda for tensor in (q, k, v)):
            return False
        if any(not tensor.is_contiguous() for tensor in (q, k, v)):
            return False
        if not (q.dtype == k.dtype == v.dtype and q.dtype in _SUPPORTED_DTYPES):
            return False
        if not (q.device == k.device == v.device):
            return False
        if q.shape[0] != k.shape[0] or tuple(k.shape) != tuple(v.shape):
            return False
        if q.shape[2] != k.shape[2] or q.shape[3] != 128 or k.shape[3] != 128:
            return False
        return torch.cuda.get_device_capability(q.device) in _SUPPORTED_COMPUTE_CAPABILITIES

    def _get_indptr(
        self,
        batch_size: int,
        num_heads: int,
        num_q_blocks: int,
        row_nnz: int,
        device: torch.device,
    ) -> torch.Tensor:
        key = (batch_size, num_heads, num_q_blocks, row_nnz, device)
        indptr = self._indptr_cache.get(key)
        if indptr is None:
            indptr = build_block_sparse_indptr(*key)
            self._indptr_cache[key] = indptr
        return indptr

    def _get_plan_params(
        self,
        q: torch.Tensor,
        k: torch.Tensor,
        *,
        row_nnz: int,
        q_block_size: int,
        kv_block_size: int,
        kv_tile_size: Literal[128, 256] | None,
        mask_type: Literal["dense", "causal"],
        output_dtype: torch.dtype,
        has_kv_token_mask: bool,
    ) -> _BlockSparsePlanParams:
        return _BlockSparsePlanParams(
            device=q.device,
            batch_size=int(q.shape[0]),
            seq_len_q=int(q.shape[1]),
            seq_len_kv=int(k.shape[1]),
            num_qo_heads=int(q.shape[2]),
            num_kv_heads=int(k.shape[2]),
            head_dim=int(q.shape[3]),
            q_block_size=q_block_size,
            kv_block_size=kv_block_size,
            kv_tile_size=kv_tile_size,
            row_nnz=row_nnz,
            mask_type=mask_type,
            q_dtype=q.dtype,
            kv_dtype=k.dtype,
            output_dtype=output_dtype,
            has_kv_token_mask=has_kv_token_mask,
        )

    def _get_packed_kv_mask(
        self,
        kv_token_mask: torch.Tensor | None,
        *,
        batch_size: int,
        seq_len_kv: int,
        device: torch.device,
    ) -> torch.Tensor | None:
        if kv_token_mask is None:
            return None
        if kv_token_mask.ndim == 1:
            if tuple(kv_token_mask.shape) != (seq_len_kv,):
                raise ValueError(
                    f"1D kv_token_mask must have shape [{seq_len_kv}], got "
                    f"{tuple(kv_token_mask.shape)}"
                )
            batched_mask = kv_token_mask.unsqueeze(0).expand(batch_size, -1)
        elif kv_token_mask.ndim == 2:
            if tuple(kv_token_mask.shape) != (batch_size, seq_len_kv):
                raise ValueError(
                    "2D kv_token_mask must have shape "
                    f"[{batch_size}, {seq_len_kv}], got {tuple(kv_token_mask.shape)}"
                )
            batched_mask = kv_token_mask
        else:
            raise ValueError("kv_token_mask must have shape [Skv] or [B, Skv]")
        if batched_mask.device != device:
            raise ValueError("kv_token_mask must be on the same device as Q/K/V")

        # Pack on every call. During CUDA Graph capture this records the
        # bool-to-bits conversion, so replay observes in-place value updates
        # to a stable ``kv_token_mask`` input instead of copying stale cached
        # bits from the eager warmup.
        return pack_kv_token_mask(batched_mask)

    def _create_plan(
        self,
        params: _BlockSparsePlanParams,
        canonical_indices: torch.Tensor,
        kv_valid_bits: torch.Tensor | None,
    ) -> _BlockSparsePlan:
        if _BlockSparseTSWrapper is None:
            raise RuntimeError(
                "FlashInfer PrimTS block-sparse attention is unavailable: "
                f"{_flashinfer_block_sparse_import_error}"
            )
        num_q_blocks = ceil(params.seq_len_q / params.q_block_size)
        block_indptr = self._get_indptr(
            params.batch_size,
            params.num_qo_heads,
            num_q_blocks,
            params.row_nnz,
            params.device,
        )
        stable_indices = canonical_indices.clone()
        stable_valid_bits = kv_valid_bits.clone() if kv_valid_bits is not None else None
        wrapper = _BlockSparseTSWrapper()
        kv_tile_size_kwargs = (
            {"kv_tile_size": params.kv_tile_size}
            if _BLOCK_SPARSE_PLAN_SUPPORTS_KV_TILE_SIZE
            else {}
        )
        wrapper.plan(
            block_indptr,
            stable_indices,
            params.batch_size,
            params.seq_len_q,
            params.seq_len_kv,
            params.num_qo_heads,
            params.num_kv_heads,
            params.head_dim,
            params.q_block_size,
            params.kv_block_size,
            kv_valid_bits=stable_valid_bits,
            mask_type=params.mask_type,
            q_data_type=params.q_dtype,
            kv_data_type=params.kv_dtype,
            o_data_type=params.output_dtype,
            dynamic_metadata=True,
            **kv_tile_size_kwargs,
        )
        return _BlockSparsePlan(
            wrapper=wrapper,
            block_indptr=block_indptr,
            block_indices=stable_indices,
            kv_valid_bits=stable_valid_bits,
        )

    @staticmethod
    def _refresh_plan_metadata(
        plan: _BlockSparsePlan,
        runtime_indices: torch.Tensor,
        kv_valid_bits: torch.Tensor | None,
    ) -> None:
        plan.block_indices.copy_(runtime_indices, non_blocking=True)
        if plan.kv_valid_bits is not None:
            if kv_valid_bits is None:
                raise RuntimeError("cached masked plan requires kv_token_mask")
            plan.kv_valid_bits.copy_(kv_valid_bits, non_blocking=True)

    @torch.compiler.disable
    def forward(
        self,
        q: torch.Tensor,
        k: torch.Tensor,
        v: torch.Tensor,
        *,
        block_indices: torch.Tensor,
        q_block_size: int,
        kv_block_size: int,
        kv_tile_size: Literal[128, 256] | None = None,
        kv_token_mask: torch.Tensor | None = None,
        mask_type: Literal["dense", "causal"] = "dense",
        sm_scale: float | None = None,
        out: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """Plan and run block-sparse attention with dynamic ``[B,H,Q,K]`` routes.

        ``kv_block_size`` is the semantic BSR block size. ``kv_tile_size`` is
        an optional physical execution-tile override understood only by newer
        FlashInfer wrappers.
        """

        kv_tile_size = _validate_kv_tile_size(kv_tile_size)
        if _BlockSparseTSWrapper is None:
            raise RuntimeError(
                "FlashInfer PrimTS block-sparse attention is unavailable: "
                f"{_flashinfer_block_sparse_import_error}"
            )
        if kv_tile_size is not None and not _BLOCK_SPARSE_PLAN_SUPPORTS_KV_TILE_SIZE:
            raise RuntimeError(
                "the installed FlashInfer BlockSparseTSWrapper.plan does not support "
                "the explicit physical kv_tile_size override"
            )
        if mask_type not in ("dense", "causal"):
            raise ValueError("mask_type must be 'dense' or 'causal'")
        if not self.is_supported(
            q,
            k,
            v,
            q_block_size=q_block_size,
            kv_block_size=kv_block_size,
        ):
            raise ValueError(
                "FlashInfer block-sparse requires compact CUDA BSHD FP16/BF16 tensors, "
                "MHA with head_dim=128, block sizes divisible by 64, and SM100/SM103"
            )

        batch_size, seq_len_q, num_heads, _ = map(int, q.shape)
        seq_len_kv = int(k.shape[1])
        num_q_blocks = ceil(seq_len_q / q_block_size)
        expected_prefix = (batch_size, num_heads, num_q_blocks)
        if block_indices.ndim != 4 or tuple(block_indices.shape[:3]) != expected_prefix:
            raise ValueError(
                "block_indices must have shape "
                f"[{batch_size}, {num_heads}, {num_q_blocks}, K], got "
                f"{tuple(block_indices.shape)}"
            )
        row_nnz = int(block_indices.shape[3])
        if row_nnz <= 0:
            raise ValueError("block_indices rows must be non-empty")
        if block_indices.device != q.device:
            raise ValueError("block_indices must be on the same device as Q/K/V")

        if out is not None:
            if tuple(out.shape) != tuple(q.shape):
                raise ValueError(f"out must have shape {tuple(q.shape)}, got {tuple(out.shape)}")
            if out.device != q.device or out.dtype != q.dtype or not out.is_contiguous():
                raise ValueError("out must be a compact Q-shaped tensor on Q's device and dtype")

        output_dtype = q.dtype if out is None else out.dtype
        plan_params = self._get_plan_params(
            q,
            k,
            row_nnz=row_nnz,
            q_block_size=q_block_size,
            kv_block_size=kv_block_size,
            kv_tile_size=kv_tile_size,
            mask_type=mask_type,
            output_dtype=output_dtype,
            has_kv_token_mask=kv_token_mask is not None,
        )
        plan = self._plans.get(plan_params)
        if plan is None and torch.cuda.is_current_stream_capturing():
            raise RuntimeError(
                "FlashInfer block-sparse plan cache miss during CUDA Graph capture. "
                "Run an eager warmup with the same tensor geometry before capture."
            )

        runtime_indices = canonicalize_block_sparse_indices(block_indices)
        kv_valid_bits = self._get_packed_kv_mask(
            kv_token_mask,
            batch_size=batch_size,
            seq_len_kv=seq_len_kv,
            device=q.device,
        )
        if plan is None:
            plan = self._create_plan(plan_params, runtime_indices, kv_valid_bits)
            self._plans[plan_params] = plan
        self._refresh_plan_metadata(plan, runtime_indices, kv_valid_bits)

        self._wrapper = plan.wrapper
        return plan.wrapper.run(q, k, v, sm_scale=sm_scale, out=out)


__all__ = [
    "FlashInferBlockSparseAttention",
    "build_block_sparse_indptr",
    "canonicalize_block_sparse_indices",
    "get_flashinfer_block_sparse_import_error",
    "is_flashinfer_block_sparse_available",
    "pack_kv_token_mask",
]
