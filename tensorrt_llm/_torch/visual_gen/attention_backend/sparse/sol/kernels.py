# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""CuTe DSL kernels and the thin torch operator for SOL prediction."""

from __future__ import annotations

import torch

from tensorrt_llm._torch.cute_dsl_utils import IS_CUTLASS_DSL_AVAILABLE

if IS_CUTLASS_DSL_AVAILABLE:
    import cutlass
    import cutlass.cute as cute

    from ._exact_selector import launch_fused_selector
    from ._kv_summary import launch_kv_summary

    try:
        from cuda.bindings import driver as cuda
    except ImportError:
        from cuda import cuda


_BLOCK_SIZE = 64
_HEAD_DIM = 128
_NUM_THREADS = 128


if IS_CUTLASS_DSL_AVAILABLE:

    class _SolPredictorKernel:
        """Three ordered launches for summaries, statistics, and selection."""

        def __init__(self, batch_size: int, seq_len: int, num_heads: int) -> None:
            self.batch_size = batch_size
            self.seq_len = seq_len
            self.num_heads = num_heads
            self.num_blocks = (seq_len + _BLOCK_SIZE - 1) // _BLOCK_SIZE

        @cute.jit
        def __call__(
            self,
            q: cute.Tensor,
            k: cute.Tensor,
            v: cute.Tensor,
            exact_block_bits: cute.Tensor,
            k_summary: cute.Tensor,
            v_summary: cute.Tensor,
            k_mean: cute.Tensor,
            k_var_diag: cute.Tensor,
            tau: cutlass.Float32,
            sm_scale: cutlass.Float32,
            stream: cuda.CUstream,
        ):
            launch_kv_summary(
                k,
                v,
                k_summary,
                v_summary,
                stream,
                self.batch_size,
                self.seq_len,
                self.num_heads,
                self.num_blocks,
            )
            self._statistics(k_summary, k_mean, k_var_diag).launch(
                grid=[self.batch_size, self.num_heads, 1],
                block=[_NUM_THREADS, 1, 1],
                stream=stream,
            )
            launch_fused_selector(
                q,
                k_summary,
                k_mean,
                k_var_diag,
                exact_block_bits,
                tau,
                sm_scale,
                stream,
                self.batch_size,
                self.seq_len,
                self.num_heads,
                self.num_blocks,
            )

        @cute.kernel
        def _statistics(
            self,
            k_summary: cute.Tensor,
            k_mean: cute.Tensor,
            k_var_diag: cute.Tensor,
        ):
            dim, _, _ = cute.arch.thread_idx()
            batch_idx, head_idx, _ = cute.arch.block_idx()
            total = cutlass.Float32(0.0)
            total_sq = cutlass.Float32(0.0)
            for block_idx in cutlass.range(self.num_blocks, unroll=8):
                value = cutlass.Float32(k_summary[batch_idx, block_idx, head_idx, dim])
                total += value
                total_sq += value * value
            mean = total / cutlass.Float32(self.num_blocks)
            variance = total_sq / cutlass.Float32(self.num_blocks) - mean * mean
            if variance < cutlass.Float32(0.0):
                variance = cutlass.Float32(0.0)
            k_mean[batch_idx, head_idx, dim] = mean
            k_var_diag[batch_idx, head_idx, dim] = variance

    class SolPredictorKernelRunner:
        """Compile cache used by the graph-visible custom operator."""

        _cache: dict[tuple[int, int, int, int], object] = {}

        @classmethod
        def _key(
            cls, device_index: int, batch_size: int, seq_len: int, num_heads: int
        ) -> tuple[int, int, int, int]:
            return (device_index, batch_size, seq_len, num_heads)

        @classmethod
        def is_compiled(
            cls, device_index: int, batch_size: int, seq_len: int, num_heads: int
        ) -> bool:
            return cls._key(device_index, batch_size, seq_len, num_heads) in cls._cache

        @classmethod
        def compile(cls, device_index: int, batch_size: int, seq_len: int, num_heads: int) -> None:
            key = cls._key(device_index, batch_size, seq_len, num_heads)
            if key in cls._cache:
                return
            blocks = (seq_len + _BLOCK_SIZE - 1) // _BLOCK_SIZE
            words = (blocks + 31) // 32
            tensor_shape = (batch_size, seq_len, num_heads, _HEAD_DIM)
            summary_shape = (batch_size, blocks, num_heads, _HEAD_DIM)
            stats_shape = (batch_size, num_heads, _HEAD_DIM)
            bits_shape = (batch_size, num_heads, blocks, words)

            def _compact(dtype: type, shape: tuple[int, ...]) -> cute.Tensor:
                return cute.runtime.make_fake_compact_tensor(
                    dtype,
                    shape,
                    stride_order=tuple(reversed(range(len(shape)))),
                    assumed_align=16,
                )

            with torch.cuda.device(device_index):
                fake_stream = cute.runtime.make_fake_stream(use_tvm_ffi_env_stream=True)
                kernel = _SolPredictorKernel(batch_size, seq_len, num_heads)
                cls._cache[key] = cute.compile(
                    kernel,
                    _compact(cutlass.BFloat16, tensor_shape),
                    _compact(cutlass.BFloat16, tensor_shape),
                    _compact(cutlass.BFloat16, tensor_shape),
                    _compact(cutlass.Uint32, bits_shape),
                    _compact(cutlass.BFloat16, summary_shape),
                    _compact(cutlass.BFloat16, summary_shape),
                    _compact(cutlass.Float32, stats_shape),
                    _compact(cutlass.Float32, stats_shape),
                    cutlass.Float32(0.0),
                    cutlass.Float32(1.0),
                    fake_stream,
                    options="--enable-tvm-ffi --opt-level 3",
                )

        @classmethod
        def run(
            cls,
            q: torch.Tensor,
            k: torch.Tensor,
            v: torch.Tensor,
            exact_block_bits: torch.Tensor,
            k_summary: torch.Tensor,
            v_summary: torch.Tensor,
            k_mean: torch.Tensor,
            k_var_diag: torch.Tensor,
            tau: float,
            sm_scale: float,
        ) -> None:
            device_index = q.device.index
            if device_index is None:
                device_index = torch.cuda.current_device()
            batch_size, seq_len, num_heads, _ = q.shape
            key = cls._key(device_index, batch_size, seq_len, num_heads)
            if key not in cls._cache:
                if torch.cuda.is_current_stream_capturing():
                    raise RuntimeError("SOL predictor must be prepared before CUDA graph capture")
                cls.compile(device_index, batch_size, seq_len, num_heads)
            cls._cache[key](
                q,
                k,
                v,
                exact_block_bits,
                k_summary,
                v_summary,
                k_mean,
                k_var_diag,
                tau,
                sm_scale,
            )

    @torch.library.custom_op(
        "trtllm::visual_gen_sol_predictor",
        mutates_args=(
            "exact_block_bits",
            "k_summary",
            "v_summary",
            "k_mean",
            "k_var_diag",
        ),
        device_types="cuda",
    )
    def visual_gen_sol_predictor(
        q: torch.Tensor,
        k: torch.Tensor,
        v: torch.Tensor,
        exact_block_bits: torch.Tensor,
        k_summary: torch.Tensor,
        v_summary: torch.Tensor,
        k_mean: torch.Tensor,
        k_var_diag: torch.Tensor,
        tau: float,
        sm_scale: float,
    ) -> None:
        """Update caller-owned SOL route and proxy tensors in place."""

        SolPredictorKernelRunner.run(
            q,
            k,
            v,
            exact_block_bits,
            k_summary,
            v_summary,
            k_mean,
            k_var_diag,
            tau,
            sm_scale,
        )

    @torch.library.register_fake("trtllm::visual_gen_sol_predictor")
    def _(
        q: torch.Tensor,
        k: torch.Tensor,
        v: torch.Tensor,
        exact_block_bits: torch.Tensor,
        k_summary: torch.Tensor,
        v_summary: torch.Tensor,
        k_mean: torch.Tensor,
        k_var_diag: torch.Tensor,
        tau: float,
        sm_scale: float,
    ) -> None:
        return None


def require_runner() -> type:
    """Return the CuTe runner or raise a precise optional-dependency error."""

    if not IS_CUTLASS_DSL_AVAILABLE:
        raise RuntimeError("SOL predictor requires the nvidia-cutlass-dsl package")
    return SolPredictorKernelRunner


__all__ = ["require_runner"]
