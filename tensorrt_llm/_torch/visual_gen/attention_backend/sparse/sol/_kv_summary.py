# Copyright (c) 2026 by FlashInfer team.
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""TMA K/V summaries for the VisualGen SOL predictor."""

from __future__ import annotations

import cutlass
import cutlass.cute as cute
import cutlass.experimental.cuda as cuda
from cutlass import BFloat16, Float32, Int32, Int64
from cutlass.experimental import primitives as prims

try:
    from cuda.bindings import driver as cuda_drv
except ImportError:
    from cuda import cuda as cuda_drv

from tensorrt_llm._torch.attention_backend.prims_ts.kernels.tensor_map import (
    TensorMapFloatOOBFill,
    TensorMapSwizzle,
    create_tensor_map_tiled,
)

BLOCK_SIZE = 64
HEAD_DIM = 128
NUM_THREADS = 128


@cute.kernel
def _kv_summary_tma_kernel(
    k_tma: cutlass.GridConstant[cuda.TensorMap],
    v_tma: cutlass.GridConstant[cuda.TensorMap],
    k_summary: cute.Tensor,
    v_summary: cute.Tensor,
    seq_len: Int32,
) -> None:
    """Produce one valid-token K mean and V sum per K64 block."""

    thread_idx, _, _ = cute.arch.thread_idx()
    block_idx, head_idx, batch_idx = cute.arch.block_idx()
    warp_idx = cute.arch.make_warp_uniform(cute.arch.warp_idx())

    smem_k = cutlass.Array(
        BFloat16,
        BLOCK_SIZE * HEAD_DIM,
        space=cutlass.AddressSpace.smem,
        alignment=1024,
    )
    smem_v = cutlass.Array(
        BFloat16,
        BLOCK_SIZE * HEAD_DIM,
        space=cutlass.AddressSpace.smem,
        alignment=1024,
    )
    ready = cutlass.Array(Int64, 1, space=cutlass.AddressSpace.smem, alignment=8)

    if thread_idx == Int32(0):
        prims.mbarrier_init(ready.data_ptr(), 1)
    prims.fence_mbarrier_init()
    prims.barrier_cta_sync(0)

    if warp_idx == Int32(0):
        if prims.elect_sync():
            prims.prefetch_tensormap(k_tma.get_ptr())
            prims.prefetch_tensormap(v_tma.get_ptr())
            prims.mbarrier_arrive_expect_tx(ready.data_ptr(), 32_768)
            token_origin = Int32(block_idx) * Int32(BLOCK_SIZE)
            coordinates_low = (
                Int32(0),
                Int32(0),
                Int32(head_idx),
                token_origin,
                Int32(batch_idx),
            )
            coordinates_high = (
                Int32(64),
                Int32(0),
                Int32(head_idx),
                token_origin,
                Int32(batch_idx),
            )
            prims.cp_async_bulk_tensor_shared_cta_global(
                smem_k.subview(0), k_tma.get_ptr(), coordinates_low, ready.data_ptr()
            )
            prims.cp_async_bulk_tensor_shared_cta_global(
                smem_k.subview(4096),
                k_tma.get_ptr(),
                coordinates_high,
                ready.data_ptr(),
            )
            prims.cp_async_bulk_tensor_shared_cta_global(
                smem_v.subview(0), v_tma.get_ptr(), coordinates_low, ready.data_ptr()
            )
            prims.cp_async_bulk_tensor_shared_cta_global(
                smem_v.subview(4096),
                v_tma.get_ptr(),
                coordinates_high,
                ready.data_ptr(),
            )

    while not prims.mbarrier_try_wait_parity(ready.data_ptr(), 0):
        pass
    prims.barrier_cta_sync(0)

    dimension_half = thread_idx // Int32(64)
    dimension_in_half = thread_idx % Int32(64)
    half_base = dimension_half * Int32(4096)
    k_sum = Float32(0.0)
    v_sum = Float32(0.0)
    smem_swizzle = cutlass.Swizzle.from_name("s128b")
    for token_idx in cutlass.range(BLOCK_SIZE, unroll=1):
        logical_offset = half_base + Int32(token_idx) * Int32(64) + dimension_in_half
        # CUTLASS 4.7 mis-lowers scalar Pointer.load_swizzled for this shape.
        k_value = cutlass.apply_swizzle(smem_k.data_ptr() + logical_offset, smem_swizzle).load()
        v_value = cutlass.apply_swizzle(smem_v.data_ptr() + logical_offset, smem_swizzle).load()
        k_sum = Float32(k_sum + Float32(k_value))
        v_sum = Float32(v_sum + Float32(v_value))

    valid_tokens = seq_len - Int32(block_idx) * Int32(BLOCK_SIZE)
    if valid_tokens > Int32(BLOCK_SIZE):
        valid_tokens = Int32(BLOCK_SIZE)
    k_summary[batch_idx, block_idx, head_idx, thread_idx] = (k_sum / Float32(valid_tokens)).to(
        BFloat16
    )
    v_summary[batch_idx, block_idx, head_idx, thread_idx] = v_sum.to(BFloat16)


@cute.jit
def launch_kv_summary(
    k: cute.Tensor,
    v: cute.Tensor,
    k_summary: cute.Tensor,
    v_summary: cute.Tensor,
    stream: cuda_drv.CUstream,
    batch_size: cutlass.Constexpr[int],
    seq_len: cutlass.Constexpr[int],
    num_heads: cutlass.Constexpr[int],
    num_blocks: cutlass.Constexpr[int],
) -> None:
    """Build TMA descriptors and launch the statically shaped summary kernel."""

    k_tma = create_tensor_map_tiled(
        global_address=k.iterator.toint(),
        dtype=BFloat16,
        global_dims=(HEAD_DIM, 1, num_heads, seq_len, batch_size),
        global_strides=(
            Int64(16),
            Int64(16),
            Int64(num_heads) * Int64(16),
            Int64(seq_len) * Int64(num_heads) * Int64(16),
        ),
        box_dims=(64, 1, 1, BLOCK_SIZE, 1),
        swizzle=TensorMapSwizzle.s128b,
        oob_fill=TensorMapFloatOOBFill.none,
    )
    v_tma = create_tensor_map_tiled(
        global_address=v.iterator.toint(),
        dtype=BFloat16,
        global_dims=(HEAD_DIM, 1, num_heads, seq_len, batch_size),
        global_strides=(
            Int64(16),
            Int64(16),
            Int64(num_heads) * Int64(16),
            Int64(seq_len) * Int64(num_heads) * Int64(16),
        ),
        box_dims=(64, 1, 1, BLOCK_SIZE, 1),
        swizzle=TensorMapSwizzle.s128b,
        oob_fill=TensorMapFloatOOBFill.none,
    )
    _kv_summary_tma_kernel(
        k_tma,
        v_tma,
        k_summary,
        v_summary,
        Int32(seq_len),
    ).launch(
        grid=(num_blocks, num_heads, batch_size),
        block=(NUM_THREADS, 1, 1),
        stream=stream,
    )


__all__ = ["launch_kv_summary"]
