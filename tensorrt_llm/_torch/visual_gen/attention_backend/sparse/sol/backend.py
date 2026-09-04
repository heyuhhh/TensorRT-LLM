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

"""VisualGen SOL attention using the generic TRTLLM sparse lifecycle."""

from __future__ import annotations

from typing import Optional

import torch

from tensorrt_llm._torch.attention_backend.block_sparse import BlockSparseForwardInputs
from tensorrt_llm._torch.attention_backend.fmha.prims_ts_block_sparse import PrimsTSBlockSparseFmha
from tensorrt_llm._torch.attention_backend.fmha.utils import get_bmm1_scale
from tensorrt_llm._torch.attention_backend.interface import PredefinedAttentionMask
from tensorrt_llm._torch.attention_backend.sparse.params import SparseRuntimeParams

from ...trtllm import SparseForwardInputs, TrtllmAttention
from .params import SolParams
from .predictor import BLOCK_SIZE, SOLSparsePredictor


class SOLTrtllmAttention(TrtllmAttention):
    """Predict SOL routes, then execute them through generic block-sparse FMHA."""

    def __init__(self, *, sparse_params: SolParams | None = None, **kwargs) -> None:
        if not isinstance(sparse_params, SolParams):
            raise TypeError("SOLTrtllmAttention requires SolParams")
        self.sol_params = sparse_params
        self._prepared_graph_phase: int | None = None
        super().__init__(sparse_params=None, **kwargs)
        attention_metadata_state = kwargs["attention_metadata_state"]
        predictor_cache = attention_metadata_state.setdefault("sparse_predictors", {})
        predictor = predictor_cache.get("sol_attn")
        if predictor is None:
            predictor = SOLSparsePredictor()
            predictor_cache["sol_attn"] = predictor
        elif not isinstance(predictor, SOLSparsePredictor):
            raise TypeError("model-scoped SOL predictor cache contains an invalid value")
        self.predictor = predictor

    def block_sparse_attn_predict(
        self,
        q: torch.Tensor,
        k: Optional[torch.Tensor],
        v: Optional[torch.Tensor],
        *,
        batch_size: int,
        seq_len: int,
        seq_len_kv: int,
        attention_mask: PredefinedAttentionMask,
        forward_kwargs: dict[str, object],
    ) -> SparseForwardInputs | None:
        """Return SOL inputs for sparse calls and ``None`` for dense calls."""

        graph_phase = forward_kwargs.pop("sol_graph_phase", None)
        if graph_phase is None and self.sol_params.disabled_until_timestep is not None:
            is_capturing = torch.cuda.is_current_stream_capturing()
            if is_capturing:
                graph_phase = self._prepared_graph_phase
            else:
                graph_phase = self.sol_params.get_graph_phase_for_timestep(
                    forward_kwargs.get("timestep"),
                    disabled_until_timestep=self.sol_params.disabled_until_timestep,
                )
                self._prepared_graph_phase = graph_phase
            if is_capturing and graph_phase is None:
                raise RuntimeError("SOL graph phase must be prepared before CUDA Graph capture")
        if not self.sol_params.should_use_sparse(
            layer_idx=self.layer_idx,
            timestep=forward_kwargs.get("timestep"),
            graph_phase=graph_phase,
        ):
            return None

        if self.quant_attention_config is not None:
            raise ValueError("SOL sparse execution does not support quant_attention_config")
        if not any(
            isinstance(fmha, PrimsTSBlockSparseFmha) for fmha in self._fmha_manager.fmha_libs
        ):
            raise RuntimeError("SOL sparse execution requires PrimTS block-sparse FMHA")
        if attention_mask != PredefinedAttentionMask.FULL:
            raise ValueError("SOL sparse execution requires a full attention mask")
        if k is None or v is None:
            raise ValueError("SOL sparse execution requires separate q, k, and v tensors")
        if seq_len_kv != seq_len:
            raise ValueError("SOL sparse execution supports only self-attention")

        expected_shape = (batch_size, seq_len, self.num_heads, self.head_dim)
        if tuple(q.shape) != expected_shape:
            raise ValueError(
                f"SOL sparse execution requires q shape {expected_shape}; got {tuple(q.shape)}"
            )
        if tuple(k.shape) != expected_shape or tuple(v.shape) != expected_shape:
            raise ValueError("SOL predictor requires uniform self-attention q/k/v shapes")

        # Fused projections produce strided split views. Compact once and share
        # these effective tensors between prediction and generic sparse FMHA.
        q = q.contiguous()
        k = k.contiguous()
        v = v.contiguous()
        unsupported_reason = self.predictor.support_reason(q, k, v)
        if unsupported_reason is not None:
            raise ValueError(unsupported_reason)

        outputs = self.predictor.predict(
            q,
            k,
            v,
            tau=self.sol_params.tau,
            sm_scale=get_bmm1_scale(self),
        )
        block_sparse_inputs = BlockSparseForwardInputs(
            q_block_size=BLOCK_SIZE,
            kv_block_size=BLOCK_SIZE,
            exact_block_bits=outputs.exact_block_bits,
            k_summary=outputs.k_summary,
            v_summary=outputs.v_summary,
        )
        return SparseForwardInputs(
            q=q,
            k=k,
            v=v,
            batch_size=batch_size,
            seq_len=seq_len,
            seq_len_kv=seq_len_kv,
            attention_mask=attention_mask,
            sparse_runtime_params=SparseRuntimeParams(
                block_sparse_inputs=block_sparse_inputs,
            ),
            forward_kwargs=forward_kwargs,
        )

    @classmethod
    def support_fused_qkv(cls) -> bool:
        """SOL prediction requires separate Q, K, and V tensors."""

        return False


__all__ = ["SOLTrtllmAttention"]
