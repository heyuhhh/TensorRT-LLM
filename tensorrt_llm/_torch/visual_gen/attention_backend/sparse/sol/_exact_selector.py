# Copyright (c) 2026 by FlashInfer team.
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Blackwell TMA/tcgen05 exact-block selector for SOL attention.

``launch_fused_selector`` is the composition interface for a larger CuTe host
callable. It consumes compact BF16 ``q[B, S, H, 128]`` and
``k_summary[B, N, H, 128]``, compact FP32 ``k_mean[B, H, 128]`` and
``k_var_diag[B, H, 128]``, and writes compact UInt32
``exact_bits[B, H, ceil(S / 64), ceil(N / 32)]``. Its scalar order is
``tau, sm_scale`` followed by the CUDA stream and constexpr geometry.

The helper is compiled as part of the predictor's single TVM-FFI host callable,
so runtime invocation performs no allocation, copy, synchronization, or
recompilation.
"""

from __future__ import annotations

import cutlass
import cutlass.cute as cute
import cutlass.experimental.cuda as cutlass_cuda
import cutlass.pipeline as pipeline
from cutlass import BFloat16, Boolean, Float32, Int32, Int64, Uint32
from cutlass.experimental import primitives as prims

try:
    from cuda.bindings import driver as cuda_driver
except ImportError:
    from cuda import cuda as cuda_driver

from tensorrt_llm._torch.attention_backend.prims_ts.kernels.mla_decode.helpers import ops as mla_ops
from tensorrt_llm._torch.attention_backend.prims_ts.kernels.tensor_map import (
    TensorMapFloatOOBFill,
    TensorMapSwizzle,
    create_tensor_map_tiled,
)

SELECTOR_N_TILE = 32
SELECTOR_NUM_THREADS = 128
SELECTOR_NUM_STAGES = 1
SELECTOR_Q_BLOCK = 64
SELECTOR_HEAD_DIM = 128


@cute.kernel
def _exact_selector_kernel(
    q_tma: cutlass.GridConstant[cutlass_cuda.TensorMap],
    k_summary_tma: cutlass.GridConstant[cutlass_cuda.TensorMap],
    k_mean: cute.Tensor,
    k_var_diag: cute.Tensor,
    exact_bits: cute.Tensor,
    tau: Float32,
    sm_scale: Float32,
    seq_len: Int32,
    num_kv_blocks: Int32,
) -> None:
    """Compute and pack every exact-selection word for one Q block."""

    thread_idx, _, _ = cute.arch.thread_idx()
    warp_idx = cute.arch.make_warp_uniform(cute.arch.warp_idx())
    q_block_idx, head_idx, batch_idx = cute.arch.block_idx()

    smem_q = cutlass.Array(
        BFloat16,
        SELECTOR_Q_BLOCK * SELECTOR_HEAD_DIM,
        space=cutlass.AddressSpace.smem,
        alignment=1024,
    )
    smem_k_summary = cutlass.Array(
        BFloat16,
        SELECTOR_NUM_STAGES * SELECTOR_N_TILE * SELECTOR_HEAD_DIM,
        space=cutlass.AddressSpace.smem,
        alignment=1024,
    )
    q_ready = cutlass.Array(Int64, 1, space=cutlass.AddressSpace.smem, alignment=8)
    tmem_token = cutlass.Array(Int32, 1, space=cutlass.AddressSpace.smem, alignment=4)
    score_sums = cutlass.Array(
        Float32,
        SELECTOR_N_TILE * 4,
        space=cutlass.AddressSpace.smem,
        alignment=16,
    )

    # This deferred K-summary init is published by the score pipeline's
    # immediately following non-deferred mbarrier fence and CTA synchronization.
    k_producer, k_consumer = pipeline.PipelineTmaUmma.create(
        num_stages=SELECTOR_NUM_STAGES,
        producer_group=pipeline.CooperativeGroup(pipeline.Agent.Thread, 1),
        consumer_group=pipeline.CooperativeGroup(pipeline.Agent.Thread, 1),
        tx_count=8192,
        barrier_storage=None,
        defer_sync=True,
        name="sol.k_summary",
    ).make_participants()
    score_producer, score_consumer = pipeline.PipelineUmmaAsync.create(
        num_stages=1,
        producer_group=pipeline.CooperativeGroup(pipeline.Agent.Thread, 1),
        consumer_group=pipeline.CooperativeGroup(pipeline.Agent.Thread, SELECTOR_NUM_THREADS),
        barrier_storage=None,
        defer_sync=False,
        name="sol.score",
    ).make_participants()

    if thread_idx == Int32(0):
        prims.mbarrier_init(q_ready.data_ptr(), 1)
    prims.fence_mbarrier_init()
    prims.barrier_cta_sync(0)

    if warp_idx == Int32(0):
        if prims.elect_sync():
            prims.prefetch_tensormap(q_tma.get_ptr())
            prims.prefetch_tensormap(k_summary_tma.get_ptr())
            prims.mbarrier_arrive_expect_tx(q_ready.data_ptr(), 16_384)
            prims.cp_async_bulk_tensor_shared_cta_global(
                smem_q.subview(0),
                q_tma.get_ptr(),
                (
                    Int32(0),
                    Int32(0),
                    Int32(head_idx),
                    Int32(q_block_idx) * Int32(SELECTOR_Q_BLOCK),
                    Int32(batch_idx),
                ),
                q_ready.data_ptr(),
            )
            prims.cp_async_bulk_tensor_shared_cta_global(
                smem_q.subview(4096),
                q_tma.get_ptr(),
                (
                    Int32(64),
                    Int32(0),
                    Int32(head_idx),
                    Int32(q_block_idx) * Int32(SELECTOR_Q_BLOCK),
                    Int32(batch_idx),
                ),
                q_ready.data_ptr(),
            )

    if warp_idx == Int32(1):
        prims.tcgen05_alloc(
            tmem_token.data_ptr(),
            Int32(32),
            group=prims.CTAGroup.CTA_1,
        )
        prims.tcgen05_relinquish_alloc_permit(group=prims.CTAGroup.CTA_1)
    # The allocation token is produced by warp 1 in shared memory. Every
    # consumer observes it only after this CTA-wide visibility point.
    prims.barrier_cta_sync(1)
    tmem_ptr = prims.make_tmem_ptr(tmem_token.data_ptr().load(), Float32)

    while not prims.mbarrier_try_wait_parity(q_ready.data_ptr(), 0):
        pass
    # Every warp reads Q during the fused threshold prelude, so all consumers
    # acquire the TMA generation before following swizzled SMEM pointers.
    prims.barrier_cta_sync(1)

    q_desc_base = Int64(0)
    k_desc_base = Int64(0)
    if warp_idx == Int32(1):
        # Both bases are loop-invariant while the serialized K-summary pipeline
        # has one SMEM stage. Keep valid values private to the MMA warp.
        q_desc_base = mla_ops.freeze_smem_descriptor(
            prims.Tcgen05SmemDesc.build(
                smem_q,
                leading_byte_offset=8192,
                stride_byte_offset=1024,
                layout=prims.Tcgen05SmemSwizzle.SWIZZLE_128B,
            )
        )
        k_desc_base = mla_ops.freeze_smem_descriptor(
            prims.Tcgen05SmemDesc.build(
                smem_k_summary,
                leading_byte_offset=4096,
                stride_byte_offset=1024,
                layout=prims.Tcgen05SmemSwizzle.SWIZZLE_128B,
            )
        )

    lane_idx = thread_idx % Int32(32)
    log2_scale = Float32(sm_scale) * Float32(1.4426950408889634)
    num_words = (num_kv_blocks + Int32(SELECTOR_N_TILE - 1)) // Int32(SELECTOR_N_TILE)
    q_len = seq_len - q_block_idx * Int32(SELECTOR_Q_BLOCK)
    if q_len > Int32(SELECTOR_Q_BLOCK):
        q_len = Int32(SELECTOR_Q_BLOCK)
    inv_q_len = Float32(1.0) / Float32(q_len)

    dimension_half = thread_idx // Int32(64)
    dimension_in_half = thread_idx % Int32(64)
    half_base = dimension_half * Int32(4096)
    q_sum = Float32(0.0)
    smem_swizzle = cutlass.Swizzle.from_name("s128b")
    for token_idx in cutlass.range(SELECTOR_Q_BLOCK, unroll=1):
        logical_offset = half_base + Int32(token_idx) * Int32(64) + dimension_in_half
        # CUTLASS 4.7 mis-lowers scalar Pointer.load_swizzled here. Apply the
        # TMA s128b mapping explicitly to preserve logical D order.
        q_value = cutlass.apply_swizzle(smem_q.data_ptr() + logical_offset, smem_swizzle).load()
        q_sum = Float32(q_sum + Float32(q_value))
    q_centroid = Float32(q_sum * inv_q_len)
    mean_part = Float32(q_centroid * Float32(k_mean[batch_idx, head_idx, thread_idx]))
    var_part = Float32(
        q_centroid * q_centroid * Float32(k_var_diag[batch_idx, head_idx, thread_idx])
    )
    for offset in (16, 8, 4, 2, 1):
        mean_part = Float32(
            mean_part
            + prims.shfl_sync(
                thread_mask=0xFFFFFFFF,
                val=mean_part,
                offset=offset,
                mask_and_clamp=0x1F,
                kind=prims.Shfl.BFLY,
            )
        )
        var_part = Float32(
            var_part
            + prims.shfl_sync(
                thread_mask=0xFFFFFFFF,
                val=var_part,
                offset=offset,
                mask_and_clamp=0x1F,
                kind=prims.Shfl.BFLY,
            )
        )
    if lane_idx == Int32(0):
        score_sums[warp_idx] = mean_part
        score_sums[Int32(4) + warp_idx] = var_part

    # Publish eight warp partials. The next barrier generation closes their
    # lifetime before the word loop reuses the same scratch array.
    prims.barrier_cta_sync(1)
    route_threshold = Float32(0.0)
    if warp_idx == Int32(0):
        mean_core = Float32(0.0)
        var_core = Float32(0.0)
        if lane_idx < Int32(4):
            mean_core = Float32(score_sums[lane_idx])
            var_core = Float32(score_sums[Int32(4) + lane_idx])
        for offset in (16, 8, 4, 2, 1):
            mean_core = Float32(
                mean_core
                + prims.shfl_sync(
                    thread_mask=0xFFFFFFFF,
                    val=mean_core,
                    offset=offset,
                    mask_and_clamp=0x1F,
                    kind=prims.Shfl.BFLY,
                )
            )
            var_core = Float32(
                var_core
                + prims.shfl_sync(
                    thread_mask=0xFFFFFFFF,
                    val=var_core,
                    offset=offset,
                    mask_and_clamp=0x1F,
                    kind=prims.Shfl.BFLY,
                )
            )
        scaled_variance = mla_ops.fmax_f32(
            Float32(var_core * log2_scale * log2_scale), Float32(0.0)
        )
        route_threshold = Float32(
            mean_core * log2_scale
            + Float32(tau)
            * Float32(
                cute.math.sqrt(
                    Float32(scaled_variance + Float32(1.0e-6)),
                    fastmath=False,
                )
            )
        )
    prims.barrier_cta_sync(1)

    for word_idx in cutlass.range(num_words, unroll=1):
        if warp_idx == Int32(0):
            k_write_handle = k_producer.acquire_and_advance()
            if prims.elect_sync():
                key_block_origin = word_idx * Int32(SELECTOR_N_TILE)
                prims.cp_async_bulk_tensor_shared_cta_global(
                    smem_k_summary.subview(0),
                    k_summary_tma.get_ptr(),
                    (
                        Int32(0),
                        key_block_origin,
                        Int32(head_idx),
                        Int32(batch_idx),
                    ),
                    k_write_handle.barrier,
                )
                prims.cp_async_bulk_tensor_shared_cta_global(
                    smem_k_summary.subview(2048),
                    k_summary_tma.get_ptr(),
                    (
                        Int32(64),
                        key_block_origin,
                        Int32(head_idx),
                        Int32(batch_idx),
                    ),
                    k_write_handle.barrier,
                )
        if warp_idx == Int32(1):
            score_write_handle = score_producer.acquire_and_advance()
            k_read_handle = k_consumer.wait_and_advance()
            q_desc = q_desc_base
            k_desc = k_desc_base
            instruction_desc = prims.Tcgen05InstrDesc.build(
                c_dtype=Float32,
                a_dtype=BFloat16,
                b_dtype=BFloat16,
                m_dim=64,
                n_dim=32,
            )
            if prims.elect_sync():
                for k_iter in cutlass.range_constexpr(8):
                    prims.tcgen05_mma(
                        prims.Tcgen05MMAKind.F16,
                        prims.CTAGroup.CTA_1,
                        tmem_ptr,
                        q_desc,
                        k_desc,
                        instruction_desc,
                        Boolean(k_iter != 0),
                    )
                    if cutlass.const_expr(k_iter + 1 < 8):
                        if cutlass.const_expr(k_iter == 3):
                            q_desc = q_desc + Int32(506)
                            k_desc = k_desc + Int32(250)
                        else:
                            q_desc = q_desc + Int32(2)
                            k_desc = k_desc + Int32(2)
            k_read_handle.release()
            score_write_handle.commit()

        score_read_handle = score_consumer.wait_and_advance()
        prims.tcgen05_fence(prims.Tcgen05Fence.AFTER_THREAD_SYNC)
        loaded_scores = mla_ops.tcgen05_ld_16x32bx2_f32(
            tmem_ptr,
            num=16,
            offset=Int32(16),
        )
        prims.tcgen05_wait(kind=prims.Tcgen05Wait.LOAD)
        cute.arch.fence_view_async_tmem_load()
        prims.tcgen05_fence(prims.Tcgen05Fence.BEFORE_THREAD_SYNC)
        score_read_handle.release()

        for reg_idx in cutlass.range_constexpr(16):
            partial = Float32(loaded_scores[reg_idx]) * log2_scale
            for offset in (8, 4, 2, 1):
                partial = Float32(
                    partial
                    + prims.shfl_sync(
                        thread_mask=0xFFFFFFFF,
                        val=partial,
                        offset=offset,
                        mask_and_clamp=0x100F,
                        kind=prims.Shfl.BFLY,
                    )
                )
            if lane_idx == Int32(0) or lane_idx == Int32(16):
                column_idx = (lane_idx // Int32(16)) * Int32(16) + Int32(reg_idx)
                score_sums[column_idx * Int32(4) + warp_idx] = partial

        # Publish all four Q16 partial sums before warp zero consumes them.
        prims.barrier_cta_sync(1)

        if warp_idx == Int32(0):
            scratch_idx = lane_idx * Int32(4)
            pair_even = Float32(score_sums[scratch_idx]) + Float32(
                score_sums[scratch_idx + Int32(2)]
            )
            pair_odd = Float32(score_sums[scratch_idx + Int32(1)]) + Float32(
                score_sums[scratch_idx + Int32(3)]
            )
            column_mean = Float32(pair_even + pair_odd) * inv_q_len
            key_block_idx = word_idx * Int32(SELECTOR_N_TILE) + lane_idx
            delta = q_block_idx - key_block_idx
            is_local = delta >= Int32(-1) and delta <= Int32(1)
            is_valid = key_block_idx < num_kv_blocks
            is_exact = Boolean(is_valid and (column_mean > route_threshold or is_local))
            exact_word = cute.arch.vote_ballot_sync(is_exact, mask=0xFFFFFFFF).bitcast(Uint32)
            if lane_idx == Int32(0):
                exact_bits[batch_idx, head_idx, q_block_idx, word_idx] = exact_word

        # Protect scratch reuse and the next iteration's TMEM overwrite.
        prims.barrier_cta_sync(1)

    if warp_idx == Int32(0):
        k_producer.tail()
    if warp_idx == Int32(1):
        score_producer.tail()
        prims.tcgen05_dealloc(
            tmem_ptr,
            Int32(32),
            group=prims.CTAGroup.CTA_1,
        )


@cute.jit
def launch_fused_selector(
    q: cute.Tensor,
    k_summary: cute.Tensor,
    k_mean: cute.Tensor,
    k_var_diag: cute.Tensor,
    exact_bits: cute.Tensor,
    tau: Float32,
    sm_scale: Float32,
    stream: cuda_driver.CUstream,
    batch_size: cutlass.Constexpr[int],
    seq_len: cutlass.Constexpr[int],
    num_heads: cutlass.Constexpr[int],
    num_kv_blocks: cutlass.Constexpr[int],
) -> None:
    """Construct TMA descriptors and launch the fused-threshold selector.

    This helper is intended to be called from another ``@cute.jit`` host
    function. The first seven arguments match the standalone runtime ABI; the
    stream and geometry remain compile-time host-launch details.
    """

    q_tma = create_tensor_map_tiled(
        global_address=q.iterator.toint(),
        dtype=BFloat16,
        global_dims=(
            SELECTOR_HEAD_DIM,
            1,
            num_heads,
            seq_len,
            batch_size,
        ),
        global_strides=(
            Int64(16),
            Int64(16),
            Int64(num_heads) * Int64(16),
            Int64(seq_len) * Int64(num_heads) * Int64(16),
        ),
        box_dims=(64, 1, 1, SELECTOR_Q_BLOCK, 1),
        swizzle=TensorMapSwizzle.s128b,
        # NaN OOB fill contaminates MMA reductions for partial Q tails.
        oob_fill=TensorMapFloatOOBFill.none,
    )
    k_summary_tma = create_tensor_map_tiled(
        global_address=k_summary.iterator.toint(),
        dtype=BFloat16,
        global_dims=(
            SELECTOR_HEAD_DIM,
            num_kv_blocks,
            num_heads,
            batch_size,
        ),
        global_strides=(
            Int64(num_heads) * Int64(16),
            Int64(16),
            Int64(num_kv_blocks) * Int64(num_heads) * Int64(16),
        ),
        box_dims=(64, SELECTOR_N_TILE, 1, 1),
        swizzle=TensorMapSwizzle.s128b,
        # Zero fill keeps the final key-summary word's tail finite.
        oob_fill=TensorMapFloatOOBFill.none,
    )
    num_q_blocks = (seq_len + SELECTOR_Q_BLOCK - 1) // SELECTOR_Q_BLOCK
    _exact_selector_kernel(
        q_tma,
        k_summary_tma,
        k_mean,
        k_var_diag,
        exact_bits,
        tau,
        sm_scale,
        Int32(seq_len),
        Int32(num_kv_blocks),
    ).launch(
        grid=(num_q_blocks, num_heads, batch_size),
        block=(SELECTOR_NUM_THREADS, 1, 1),
        stream=stream,
    )


__all__ = ["launch_fused_selector"]
