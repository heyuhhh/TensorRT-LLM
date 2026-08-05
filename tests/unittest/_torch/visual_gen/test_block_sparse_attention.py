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

"""Lightweight contract tests for VisualGen block-sparse attention metadata."""

from types import SimpleNamespace

import pytest
import torch

from tensorrt_llm._torch.visual_gen.attention_backend import block_sparse as block_sparse_module
from tensorrt_llm._torch.visual_gen.attention_backend.block_sparse import (
    FlashInferBlockSparseAttention,
    build_block_sparse_indptr,
    canonicalize_block_sparse_indices,
    pack_kv_token_mask,
)
from tensorrt_llm._torch.visual_gen.attention_backend.trtllm_vsa import TrtllmVSAAttention
from tensorrt_llm._torch.visual_gen.attention_backend.utils import create_attention
from tensorrt_llm._torch.visual_gen.attention_backend.vsa import VSAMetadataBuilder
from tensorrt_llm._torch.visual_gen.config import (
    DiffusionModelConfig,
    create_attention_metadata_state,
)
from tensorrt_llm._torch.visual_gen.modules.attention import Attention, QKVMode
from tensorrt_llm.visual_gen.args import AttentionConfig, VideoSparseAttentionConfig


def _trtllm_vsa_config() -> AttentionConfig:
    return AttentionConfig(
        backend="TRTLLM",
        sparse_attention_config=VideoSparseAttentionConfig(vsa_sparsity=0.9),
    )


def test_block_sparse_indptr_uses_absolute_fixed_row_offsets() -> None:
    indptr = build_block_sparse_indptr(
        batch_size=2,
        num_heads=2,
        num_q_blocks=3,
        row_nnz=4,
        device=torch.device("cpu"),
    )

    assert indptr.tolist() == [
        [[0, 4, 8, 12], [12, 16, 20, 24]],
        [[24, 28, 32, 36], [36, 40, 44, 48]],
    ]


def test_block_sparse_indptr_rejects_int32_offset_overflow() -> None:
    with pytest.raises(ValueError, match="exceed int32 capacity"):
        build_block_sparse_indptr(
            batch_size=1 << 20,
            num_heads=1 << 11,
            num_q_blocks=1,
            row_nnz=2,
            device=torch.device("cpu"),
        )


def test_block_sparse_routes_are_sorted_per_row_and_flattened() -> None:
    routes = torch.tensor(
        [[[[3, 1, 2], [2, 0, 1]], [[4, 2, 3], [1, 0, 2]]]],
        dtype=torch.int64,
    )

    canonical = canonicalize_block_sparse_indices(routes)

    assert canonical.dtype == torch.int32
    assert canonical.tolist() == [1, 2, 3, 0, 1, 2, 2, 3, 4, 0, 1, 2]


def test_kv_token_mask_uses_lsb_first_words_and_zero_padding() -> None:
    mask = torch.zeros((1, 35), dtype=torch.bool)
    mask[0, [0, 2, 31, 32, 34]] = True

    packed = pack_kv_token_mask(mask)

    assert packed.dtype == torch.uint32
    assert packed.shape == (1, 2)
    assert packed.tolist() == [[0x80000005, 0x00000005]]


def test_trtllm_vsa_factory_reuses_model_scoped_plan_cache() -> None:
    metadata_state = {}

    first = create_attention(
        backend="TRTLLM",
        layer_idx=0,
        num_heads=4,
        head_dim=128,
        attention_config=_trtllm_vsa_config(),
        attention_metadata_state=metadata_state,
    )
    second = create_attention(
        backend="TRTLLM",
        layer_idx=1,
        num_heads=4,
        head_dim=128,
        attention_config=_trtllm_vsa_config(),
        attention_metadata_state=metadata_state,
    )

    assert isinstance(first, TrtllmVSAAttention)
    assert isinstance(second, TrtllmVSAAttention)
    assert first._block_sparse_attention is second._block_sparse_attention


def test_trtllm_vsa_factory_requires_model_scoped_metadata() -> None:
    with pytest.raises(ValueError, match="requires `attention_metadata_state`"):
        create_attention(
            backend="TRTLLM",
            layer_idx=0,
            num_heads=4,
            head_dim=128,
            attention_config=_trtllm_vsa_config(),
        )


def test_trtllm_vsa_rejects_gqa_at_construction() -> None:
    with pytest.raises(ValueError, match="GQA/MQA is not supported"):
        TrtllmVSAAttention(num_heads=4, num_kv_heads=2, head_dim=128)


def test_explicit_self_attention_keeps_vsa_with_separate_qkv() -> None:
    config = DiffusionModelConfig(
        pretrained_config=SimpleNamespace(
            hidden_size=64,
            num_attention_heads=4,
            attention_head_dim=16,
            eps=1e-6,
        ),
        attention=_trtllm_vsa_config(),
        attention_metadata_state=create_attention_metadata_state(),
        skip_create_weights_in_init=False,
    )

    attention = Attention(
        hidden_size=64,
        num_attention_heads=4,
        head_dim=16,
        qkv_mode=QKVMode.SEPARATE_QKV,
        config=config,
        is_cross_attention=False,
    )

    assert attention.attn_backend == "TRTLLM"
    assert isinstance(attention.attn, TrtllmVSAAttention)


def test_block_sparse_capture_requires_preplanned_geometry(monkeypatch) -> None:
    class FakeBlockSparseWrapper:
        pass

    monkeypatch.setattr(block_sparse_module, "_BlockSparseTSWrapper", FakeBlockSparseWrapper)
    monkeypatch.setattr(torch.cuda, "is_current_stream_capturing", lambda: True)
    adapter = FlashInferBlockSparseAttention()
    monkeypatch.setattr(adapter, "is_supported", lambda *args, **kwargs: True)

    q = torch.zeros((1, 64, 1, 128), dtype=torch.bfloat16)
    k = torch.zeros_like(q)
    v = torch.zeros_like(q)
    routes = torch.zeros((1, 1, 1, 1), dtype=torch.int32)

    with pytest.raises(RuntimeError, match="plan cache miss during CUDA Graph capture"):
        adapter.forward(
            q,
            k,
            v,
            block_indices=routes,
            q_block_size=64,
            kv_block_size=64,
        )


def test_block_sparse_adapter_plans_once_and_refreshes_dynamic_metadata(monkeypatch) -> None:
    class FakeBlockSparseWrapper:
        def __init__(self) -> None:
            self.plan_calls = 0
            self.run_calls = 0
            self.route_snapshots = []
            self.mask_snapshots = []

        def plan(self, block_indptr, block_indices, *args, kv_valid_bits=None, **kwargs) -> None:
            self.plan_calls += 1
            self.block_indices = block_indices
            self.kv_valid_bits = kv_valid_bits

        def run(self, q, k, v, *, sm_scale=None, out=None):
            self.run_calls += 1
            self.route_snapshots.append(self.block_indices.clone())
            self.mask_snapshots.append(self.kv_valid_bits.clone())
            return q if out is None else out.copy_(q)

    monkeypatch.setattr(block_sparse_module, "_BlockSparseTSWrapper", FakeBlockSparseWrapper)
    adapter = FlashInferBlockSparseAttention()
    monkeypatch.setattr(adapter, "is_supported", lambda *args, **kwargs: True)

    q = torch.zeros((1, 128, 2, 128), dtype=torch.bfloat16)
    k = torch.zeros((1, 256, 2, 128), dtype=torch.bfloat16)
    v = torch.zeros_like(k)
    routes = torch.tensor(
        [[[[3, 1], [0, 2]], [[2, 0], [3, 1]]]],
        dtype=torch.int32,
    )
    token_mask = torch.ones(256, dtype=torch.bool)
    token_mask[5] = False

    adapter.forward(
        q,
        k,
        v,
        block_indices=routes,
        q_block_size=64,
        kv_block_size=64,
        kv_token_mask=token_mask,
    )
    routes.copy_(torch.tensor([[[[2, 0], [3, 1]], [[1, 0], [2, 3]]]], dtype=torch.int32))
    token_mask[5] = True
    token_mask[129] = False
    adapter.forward(
        q,
        k,
        v,
        block_indices=routes,
        q_block_size=64,
        kv_block_size=64,
        kv_token_mask=token_mask,
    )

    wrapper = adapter._wrapper
    assert wrapper.plan_calls == 1
    assert wrapper.run_calls == 2
    assert len(adapter._plans) == 1
    assert not torch.equal(wrapper.route_snapshots[0], wrapper.route_snapshots[1])
    assert not torch.equal(wrapper.mask_snapshots[0], wrapper.mask_snapshots[1])


def test_vsa_metadata_exposes_exact_ragged_token_mask() -> None:
    metadata = VSAMetadataBuilder().build(
        current_timestep=0,
        raw_latent_shape=(5, 6, 7),
        patch_size=(1, 1, 1),
        vsa_sparsity=0.9,
        device=torch.device("cpu"),
    )

    expected_mask = metadata.tile_partition_indices >= 0
    assert torch.equal(metadata.kv_token_mask, expected_mask)
    assert metadata.kv_token_mask.sum().item() == 5 * 6 * 7
    assert metadata.kv_token_mask.numel() == metadata.padded_seq_length
