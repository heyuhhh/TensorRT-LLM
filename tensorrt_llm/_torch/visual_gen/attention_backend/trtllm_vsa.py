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

"""TRT-LLM VSA fine stage implemented by the FlashInfer PrimTS wrapper."""

from typing import Optional

import torch

from tensorrt_llm.logger import logger

from .block_sparse import (
    FlashInferBlockSparseAttention,
    get_flashinfer_block_sparse_import_error,
    is_flashinfer_block_sparse_available,
)
from .vsa import VSA_TILE_SIZE, VSAAttentionBase


class TrtllmVSAAttention(VSAAttentionBase):
    """VSA frontend with TRT-LLM task-scheduled block-sparse fine attention."""

    def __init__(
        self,
        layer_idx: int = 0,
        num_heads: int = 8,
        head_dim: int = 128,
        num_kv_heads: Optional[int] = None,
        dtype: Optional[torch.dtype] = None,
        sparse_attention_config=None,
        attention_metadata_state: Optional[dict] = None,
        **kwargs,
    ) -> None:
        super().__init__(
            layer_idx=layer_idx,
            num_heads=num_heads,
            head_dim=head_dim,
            num_kv_heads=num_kv_heads,
            dtype=dtype,
            sparse_attention_config=sparse_attention_config,
            **kwargs,
        )
        if not is_flashinfer_block_sparse_available():
            logger.warning_once(
                "TRTLLM VSA requires a FlashInfer build with the PrimTS block-sparse "
                "dynamic-metadata API; using the dense SDPA fine-stage fallback. "
                f"Import/capability error: {get_flashinfer_block_sparse_import_error()}",
                key="visual_gen_trtllm_vsa_primts_unavailable",
            )
        if attention_metadata_state is None:
            self._block_sparse_attention = FlashInferBlockSparseAttention()
        else:
            cache_key = "vsa_block_sparse_attention"
            block_sparse_attention = attention_metadata_state.get(cache_key)
            if block_sparse_attention is None:
                block_sparse_attention = FlashInferBlockSparseAttention()
                attention_metadata_state[cache_key] = block_sparse_attention
            self._block_sparse_attention = block_sparse_attention

    def _select_physical_kv_tile_size(
        self,
        q_tiled: torch.Tensor,
        *,
        q_block_size: int,
        kv_block_size: int,
        kv_token_mask: torch.Tensor | None,
    ) -> int | None:
        """Select the SM100 BF16 VSA-specialized physical KV256 execution tile."""

        if not getattr(self._block_sparse_attention, "supports_kv_tile_size", False):
            return None
        if q_tiled.dtype != torch.bfloat16 or int(q_tiled.shape[-1]) != 128:
            return None
        if q_block_size != 64 or kv_block_size != 64 or kv_token_mask is None:
            return None
        if int(q_tiled.shape[1]) % q_block_size != 0:
            return None
        if torch.cuda.get_device_capability(q_tiled.device) != (10, 0):
            return None
        # The specialized role-local core is selected only by FlashInfer's
        # persistent scheduler. Avoid forcing generic physical KV256 for small
        # grids, where the automatic masked path intentionally remains KV128.
        num_q_tiles = int(q_tiled.shape[1]) // q_block_size
        direct_ctas = int(q_tiled.shape[0]) * int(q_tiled.shape[2]) * num_q_tiles
        sm_count = int(torch.cuda.get_device_properties(q_tiled.device).multi_processor_count)
        if sm_count <= 0 or direct_ctas <= sm_count:
            return None
        return 256

    @torch.compiler.disable
    def _forward_sparse_fine(
        self,
        q_tiled: torch.Tensor,
        k_tiled: torch.Tensor,
        v_tiled: torch.Tensor,
        topk_indices: torch.Tensor,
        variable_block_sizes: torch.LongTensor,
        kv_token_mask: torch.BoolTensor,
        cur_topk: int,
        num_cubes: int,
    ) -> Optional[torch.Tensor]:
        del variable_block_sizes, cur_topk, num_cubes
        block_size = VSA_TILE_SIZE[0] * VSA_TILE_SIZE[1] * VSA_TILE_SIZE[2]
        if not self._block_sparse_attention.is_supported(
            q_tiled,
            k_tiled,
            v_tiled,
            q_block_size=block_size,
            kv_block_size=block_size,
        ):
            if is_flashinfer_block_sparse_available():
                logger.warning_once(
                    "TRTLLM VSA input is outside the PrimTS block-sparse kernel envelope "
                    "(compact CUDA BSHD FP16/BF16 MHA, D=128, SM100/SM103); using the "
                    "dense SDPA fine-stage fallback.",
                    key="visual_gen_trtllm_vsa_primts_unsupported_envelope",
                )
            return None
        kv_tile_size = self._select_physical_kv_tile_size(
            q_tiled,
            q_block_size=block_size,
            kv_block_size=block_size,
            kv_token_mask=kv_token_mask,
        )
        return self._block_sparse_attention.forward(
            q_tiled,
            k_tiled,
            v_tiled,
            block_indices=topk_indices,
            q_block_size=block_size,
            kv_block_size=block_size,
            kv_tile_size=kv_tile_size,
            kv_token_mask=kv_token_mask,
            sm_scale=q_tiled.shape[-1] ** -0.5,
        )


__all__ = ["TrtllmVSAAttention"]
