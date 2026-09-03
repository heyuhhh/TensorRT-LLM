# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from dataclasses import FrozenInstanceError, dataclass, replace
from unittest.mock import Mock

import pytest
import torch

from tensorrt_llm._torch.attention_backend.block_sparse import BlockSparseForwardInputs
from tensorrt_llm._torch.attention_backend.interface import PredefinedAttentionMask
from tensorrt_llm._torch.attention_backend.sparse.params import SparseAttentionPrediction
from tensorrt_llm._torch.visual_gen.attention_backend import trtllm as visual_trtllm


class _FakeBaseTrtllmAttentionMetadata:
    def __init__(self, **kwargs):
        self.kwargs = kwargs
        self.prepare_calls = 0
        self.seq_lens = None
        self.num_contexts = None
        self.max_seq_len = None
        self.request_ids = None

    def prepare(self):
        self.prepare_calls += 1


class _RecordingSparseAttention(visual_trtllm.TrtllmAttention):
    def block_sparse_attn_predict(
        self,
        q,
        k,
        v,
        *,
        batch_size,
        seq_len,
        seq_len_kv,
        attention_mask,
        forward_kwargs,
        block_sparse_inputs=None,
    ):
        if block_sparse_inputs is not None:
            raise ValueError("block_sparse_inputs must be produced by block_sparse_attn_predict")
        self.lifecycle_events.append("predict")
        self.predict_inputs = {
            "q": q,
            "k": k,
            "v": v,
            "batch_size": batch_size,
            "seq_len": seq_len,
            "seq_len_kv": seq_len_kv,
            "attention_mask": attention_mask,
            "forward_kwargs": forward_kwargs,
        }
        self.prediction = visual_trtllm.SparseForwardInputs(
            q=q + 1,
            k=k,
            v=v,
            batch_size=batch_size,
            seq_len=seq_len,
            seq_len_kv=seq_len_kv,
            attention_mask=attention_mask,
            block_sparse_inputs=self.predicted_carrier,
            forward_kwargs=forward_kwargs,
        )
        return self.prediction

    def _forward_impl(
        self,
        q,
        k,
        v,
        batch_size,
        seq_len,
        attention_mask=PredefinedAttentionMask.FULL,
        seq_len_kv=None,
        block_sparse_inputs=None,
        sparse_attention_prediction=None,
        **kwargs,
    ):
        self.lifecycle_events.append("forward_impl")
        self.impl_inputs = {
            "q": q,
            "k": k,
            "v": v,
            "batch_size": batch_size,
            "seq_len": seq_len,
            "attention_mask": attention_mask,
            "seq_len_kv": seq_len_kv,
            "block_sparse_inputs": block_sparse_inputs,
            "sparse_attention_prediction": sparse_attention_prediction,
            "kwargs": kwargs,
        }
        return q.reshape(batch_size, seq_len, -1)

    def sparse_post_process(self, output, prediction):
        self.lifecycle_events.append("post_process")
        self.post_process_prediction = prediction
        return output + 2


class _NoPredictionSparseAttention(_RecordingSparseAttention):
    def block_sparse_attn_predict(self, *args, forward_kwargs, **kwargs):
        del args, kwargs
        self.lifecycle_events.append("predict")
        forward_kwargs.pop("private_sparse_phase", None)
        return None


@dataclass(frozen=True, slots=True, kw_only=True, eq=False)
class _ExtendedSparseForwardInputs(visual_trtllm.SparseForwardInputs):
    post_marker: str


def _make_block_sparse_inputs():
    return BlockSparseForwardInputs(
        q_block_size=64,
        kv_block_size=64,
        max_blocks_per_row=1,
        block_indptr=torch.tensor([[[0, 1]]], dtype=torch.int32),
        block_indices=torch.tensor([0], dtype=torch.int32),
    )


class _StopAtFmhaDispatch(Exception):
    pass


def _make_core_forward_metadata():
    metadata = object.__new__(visual_trtllm.BaseTrtllmAttentionMetadata)
    seq_lens = torch.tensor([4], dtype=torch.int32)
    metadata._seq_lens = seq_lens
    metadata._seq_lens_kv = seq_lens
    metadata._seq_lens_cuda = None
    metadata.kv_cache_manager = None
    metadata._max_seq_len_storage = 4
    metadata.use_paged_context_fmha = False
    metadata.cu_q_seqlens = None
    metadata.cu_kv_seqlens = None
    metadata.enable_flash_mla = False
    metadata.spec_bl_tree_first_sparse_mask_offset_kv = None
    metadata.spec_decoding_bl_tree_mask = None
    metadata.kv_lens_cuda_runtime = torch.tensor([4], dtype=torch.int32)
    metadata.kv_lens_runtime = torch.tensor([4], dtype=torch.int32)
    metadata.prompt_lens_cuda_runtime = torch.tensor([4], dtype=torch.int32)
    metadata.prompt_lens_cpu_runtime = torch.tensor([4], dtype=torch.int32)
    metadata.host_request_types_runtime = torch.tensor([0], dtype=torch.int32)
    metadata.max_context_q_len_override = None
    return metadata


def test_trtllm_attention_metadata_caches_distinct_seq_lens(monkeypatch):
    monkeypatch.setattr(
        visual_trtllm,
        "BaseTrtllmAttentionMetadata",
        _FakeBaseTrtllmAttentionMetadata,
    )
    attention_metadata_state = {}
    metadata = visual_trtllm.TrtllmAttentionMetadata(
        device=torch.device("cpu"),
        attention_metadata_state=attention_metadata_state,
    )

    first_seq_lens = torch.tensor([64], dtype=torch.int32)
    first_metadata = metadata.prepare(batch_size=1, seq_lens=first_seq_lens)
    first_seq_lens.fill_(999)

    second_metadata = metadata.prepare(batch_size=1, seq_lens=torch.tensor([96], dtype=torch.int32))
    first_metadata_again = metadata.prepare(
        batch_size=1,
        seq_lens=torch.tensor([64], dtype=torch.int32),
    )

    assert first_metadata is first_metadata_again
    assert first_metadata is not second_metadata
    assert first_metadata.prepare_calls == 1
    assert second_metadata.prepare_calls == 1

    metadata_cache = attention_metadata_state["metadata_cache"]
    assert set(metadata_cache) == {
        (1, (64,)),
        (1, (96,)),
    }
    assert metadata_cache[(1, (64,))]["metadata"] is first_metadata
    assert metadata_cache[(1, (96,))]["metadata"] is second_metadata

    first_cached_seq_lens = metadata_cache[(1, (64,))]["seq_lens"]
    second_cached_seq_lens = metadata_cache[(1, (96,))]["seq_lens"]
    assert torch.equal(first_cached_seq_lens, torch.tensor([64], dtype=torch.int32))
    assert torch.equal(second_cached_seq_lens, torch.tensor([96], dtype=torch.int32))
    assert first_cached_seq_lens is not second_cached_seq_lens
    assert first_cached_seq_lens.data_ptr() != second_cached_seq_lens.data_ptr()
    assert first_metadata.seq_lens is first_cached_seq_lens
    assert second_metadata.seq_lens is second_cached_seq_lens


def test_trtllm_attention_layers_share_block_sparse_plan_cache(monkeypatch):
    observed_cache_states = []

    def _base_init(self, **kwargs):
        del kwargs
        observed_cache_states.append(self._block_sparse_fmha_cache_state)

    monkeypatch.setattr(visual_trtllm.BaseTrtllmAttention, "__init__", _base_init)
    attention_metadata_state = {}

    first = visual_trtllm.TrtllmAttention(
        attention_metadata_state=attention_metadata_state,
    )
    second = visual_trtllm.TrtllmAttention(
        attention_metadata_state=attention_metadata_state,
    )

    assert first._block_sparse_fmha_cache_state is second._block_sparse_fmha_cache_state
    assert observed_cache_states == [
        attention_metadata_state["block_sparse_fmha_cache"],
        attention_metadata_state["block_sparse_fmha_cache"],
    ]
    assert attention_metadata_state["block_sparse_fmha_cache"] == {
        "contiguous_wrappers": {},
        "paged_wrappers": {},
    }


def test_sparse_forward_inputs_are_immutable_and_copy_forward_kwargs():
    q = torch.randn(1, 4, 2, 8)
    kwargs = {"timestep": torch.tensor([12]), "sol_graph_phase": "sparse"}
    prediction = visual_trtllm.SparseForwardInputs(
        q=q,
        k=None,
        v=None,
        batch_size=1,
        seq_len=4,
        seq_len_kv=4,
        attention_mask=PredefinedAttentionMask.FULL,
        forward_kwargs=kwargs,
    )
    kwargs["sol_graph_phase"] = "dense"

    assert prediction.q is q
    assert prediction.forward_kwargs["sol_graph_phase"] == "sparse"
    with pytest.raises(FrozenInstanceError):
        prediction.seq_len = 8
    with pytest.raises(TypeError):
        prediction.forward_kwargs["sol_graph_phase"] = "dense"


def test_sparse_forward_inputs_use_identity_equality_and_compact_repr():
    prediction = visual_trtllm.SparseForwardInputs(
        q=torch.randn(1, 4, 2, 8),
        k=None,
        v=None,
        batch_size=1,
        seq_len=4,
        seq_len_kv=4,
        attention_mask=PredefinedAttentionMask.FULL,
    )

    copied_prediction = replace(prediction)

    assert copied_prediction is not prediction
    assert copied_prediction != prediction
    assert "tensor(" not in repr(prediction)


def test_sparse_lifecycle_accepts_typed_prediction_context():
    class _TypedSparseAttention(visual_trtllm.TrtllmAttention):
        def block_sparse_attn_predict(
            self,
            q,
            k,
            v,
            *,
            batch_size,
            seq_len,
            seq_len_kv,
            attention_mask,
            forward_kwargs,
            block_sparse_inputs=None,
        ) -> _ExtendedSparseForwardInputs:
            assert block_sparse_inputs is None
            return _ExtendedSparseForwardInputs(
                q=q,
                k=k,
                v=v,
                post_marker="vsa",
                batch_size=batch_size,
                seq_len=seq_len,
                seq_len_kv=seq_len_kv,
                attention_mask=attention_mask,
                forward_kwargs=forward_kwargs,
            )

        def _forward_impl(self, q, *args, **kwargs):
            return q.reshape(1, 4, -1)

        def sparse_post_process(self, output, sparse_inputs):
            self.post_inputs = sparse_inputs
            return output

    attention = object.__new__(_TypedSparseAttention)
    q = torch.randn(1, 4, 2, 8)

    output = attention.forward(q, None, None, batch_size=1, seq_len=4)

    assert isinstance(attention.post_inputs, _ExtendedSparseForwardInputs)
    assert attention.post_inputs.post_marker == "vsa"
    assert output.shape == (1, 4, 16)


def test_sparse_lifecycle_does_not_require_generic_parameterization():
    assert getattr(visual_trtllm.TrtllmAttention, "__parameters__", ()) == ()


def test_sparse_lifecycle_default_predictor_returns_none():
    attention = object.__new__(visual_trtllm.TrtllmAttention)
    q = torch.randn(1, 4, 2, 8)

    assert "block_sparse_attn_predict" in visual_trtllm.TrtllmAttention.__dict__
    assert "sparse_attn_predict" not in visual_trtllm.TrtllmAttention.__dict__
    assert "sparse_predict" not in visual_trtllm.TrtllmAttention.__dict__
    assert (
        attention.block_sparse_attn_predict(
            q,
            None,
            None,
            batch_size=1,
            seq_len=4,
            seq_len_kv=4,
            attention_mask=PredefinedAttentionMask.FULL,
            forward_kwargs={},
        )
        is None
    )
    assert "_enable_sparse_workflow" not in visual_trtllm.TrtllmAttention.__dict__
    assert "_should_use_sparse_workflow" not in visual_trtllm.TrtllmAttention.__dict__


def test_sparse_lifecycle_predicts_block_inputs_then_runs_normal_forward():
    assert "sparse_preprocess" not in visual_trtllm.TrtllmAttention.__dict__
    attention = object.__new__(_RecordingSparseAttention)
    attention.lifecycle_events = []
    attention.predicted_carrier = _make_block_sparse_inputs()
    q = torch.randn(1, 4, 2, 8)
    k = torch.randn_like(q)
    v = torch.randn_like(q)
    timestep = torch.tensor([12])
    gate_compress = torch.randn(1, 4, 1)

    output = attention.forward(
        q,
        k,
        v,
        batch_size=1,
        seq_len=4,
        attention_mask=PredefinedAttentionMask.FULL,
        timestep=timestep,
        gate_compress=gate_compress,
    )

    assert attention.lifecycle_events == ["predict", "forward_impl", "post_process"]
    assert attention.predict_inputs["q"] is q
    assert attention.impl_inputs["q"] is attention.prediction.q
    assert attention.impl_inputs["k"] is k
    assert attention.impl_inputs["v"] is v
    assert attention.impl_inputs["block_sparse_inputs"] is None
    core_prediction = attention.impl_inputs["sparse_attention_prediction"]
    assert isinstance(core_prediction, SparseAttentionPrediction)
    assert core_prediction.block_sparse_inputs is attention.predicted_carrier
    assert attention.impl_inputs["kwargs"] == {
        "timestep": timestep,
        "gate_compress": gate_compress,
    }
    assert attention.post_process_prediction is attention.prediction
    torch.testing.assert_close(output, (q + 1).reshape(1, 4, -1) + 2)


def test_sparse_lifecycle_dense_fallback_still_hands_off_prediction_sentinel():
    attention = object.__new__(_RecordingSparseAttention)
    attention.lifecycle_events = []
    attention.predicted_carrier = None
    q = torch.randn(1, 4, 2, 8)
    k = torch.randn_like(q)
    v = torch.randn_like(q)

    output = attention.forward(q, k, v, batch_size=1, seq_len=4)

    assert attention.lifecycle_events == ["predict", "forward_impl", "post_process"]
    assert attention.impl_inputs["q"] is attention.prediction.q
    assert attention.impl_inputs["k"] is k
    assert attention.impl_inputs["v"] is v
    assert attention.impl_inputs["block_sparse_inputs"] is None
    core_prediction = attention.impl_inputs["sparse_attention_prediction"]
    assert isinstance(core_prediction, SparseAttentionPrediction)
    assert core_prediction.block_sparse_inputs is None
    assert attention.post_process_prediction is attention.prediction
    torch.testing.assert_close(output, (q + 1).reshape(1, 4, -1) + 2)


def test_sparse_lifecycle_none_prediction_uses_normal_forward_and_cleans_kwargs():
    attention = object.__new__(_NoPredictionSparseAttention)
    attention.lifecycle_events = []
    attention.predicted_carrier = _make_block_sparse_inputs()
    q = torch.randn(1, 4, 2, 8)
    timestep = torch.tensor([12])

    output = attention.forward(
        q,
        None,
        None,
        batch_size=1,
        seq_len=4,
        timestep=timestep,
        private_sparse_phase="dense",
    )

    assert attention.lifecycle_events == ["predict", "forward_impl"]
    assert "private_sparse_phase" not in attention.impl_inputs["kwargs"]
    torch.testing.assert_close(output, q.reshape(1, 4, -1))


def test_sparse_lifecycle_rejects_caller_block_sparse_inputs():
    attention = object.__new__(_RecordingSparseAttention)
    attention.lifecycle_events = []
    attention.predicted_carrier = _make_block_sparse_inputs()
    q = torch.randn(1, 4, 2, 8)

    with pytest.raises(ValueError, match="must be produced by block_sparse_attn_predict"):
        attention.forward(
            q,
            q,
            q,
            batch_size=1,
            seq_len=4,
            block_sparse_inputs=_make_block_sparse_inputs(),
        )

    assert attention.lifecycle_events == []


def test_dense_trtllm_attention_runs_noop_predictor_without_post_process(monkeypatch):
    captured = {}
    events = []
    q = torch.randn(1, 4, 2, 8)

    def _no_prediction(*args, **kwargs):
        del args, kwargs
        events.append("predict")
        return None

    def _fail_post_process(*args, **kwargs):
        raise AssertionError(f"no-op prediction invoked post-process: {args}, {kwargs}")

    def _capture_impl(self, *args, **kwargs):
        captured.update(args=args, kwargs=kwargs)
        return q.reshape(1, 4, -1)

    monkeypatch.setattr(
        visual_trtllm.TrtllmAttention,
        "block_sparse_attn_predict",
        _no_prediction,
        raising=False,
    )
    monkeypatch.setattr(
        visual_trtllm.TrtllmAttention,
        "sparse_post_process",
        _fail_post_process,
        raising=False,
    )
    monkeypatch.setattr(
        visual_trtllm.TrtllmAttention, "_forward_impl", _capture_impl, raising=False
    )

    attention = object.__new__(visual_trtllm.TrtllmAttention)
    output = attention.forward(q, None, None, batch_size=1, seq_len=4)

    assert events == ["predict"]
    assert captured["args"][:5] == (q, None, None, 1, 4)
    assert output.shape == (1, 4, 16)


def test_plain_trtllm_attention_forwards_caller_block_sparse_inputs(monkeypatch):
    captured = {}
    prepared_metadata = object()

    monkeypatch.setattr(
        visual_trtllm.TrtllmAttention,
        "_prepare_metadata",
        lambda self, batch_size, seq_len: prepared_metadata,
    )

    def _capture_base_forward(self, q, k, v, metadata, forward_args=None, **kwargs):
        captured.update(
            q=q,
            k=k,
            v=v,
            metadata=metadata,
            forward_args=forward_args,
            kwargs=kwargs,
        )
        return q

    monkeypatch.setattr(
        visual_trtllm.BaseTrtllmAttention,
        "forward",
        _capture_base_forward,
    )

    attention = object.__new__(visual_trtllm.TrtllmAttention)
    attention.quant_attention_config = None
    block_sparse_inputs = _make_block_sparse_inputs()
    q = torch.randn(1, 4, 2, 8)
    k = torch.randn_like(q)
    v = torch.randn_like(q)

    output = attention.forward(
        q,
        k,
        v,
        batch_size=1,
        seq_len=4,
        block_sparse_inputs=block_sparse_inputs,
    )

    assert output.shape == (1, 4, 16)
    assert captured["q"].shape == (4, 16)
    assert captured["k"].shape == (4, 16)
    assert captured["v"].shape == (4, 16)
    assert captured["metadata"] is prepared_metadata
    assert captured["forward_args"].block_sparse_inputs is block_sparse_inputs
    assert captured["forward_args"].sparse_attention_prediction is None
    assert captured["kwargs"] == {}


@pytest.mark.parametrize("has_block_sparse_inputs", [False, True])
def test_sparse_lifecycle_passes_only_typed_prediction_to_core(
    monkeypatch,
    has_block_sparse_inputs,
):
    captured = {}
    prepared_metadata = object()

    class _CoreHandoffSparseAttention(visual_trtllm.TrtllmAttention):
        def block_sparse_attn_predict(
            self,
            q,
            k,
            v,
            *,
            batch_size,
            seq_len,
            seq_len_kv,
            attention_mask,
            forward_kwargs,
            block_sparse_inputs=None,
        ):
            assert block_sparse_inputs is None
            self.prediction = visual_trtllm.SparseForwardInputs(
                q=q + 1,
                k=k,
                v=v,
                batch_size=batch_size,
                seq_len=seq_len,
                seq_len_kv=seq_len_kv,
                attention_mask=attention_mask,
                block_sparse_inputs=self.predicted_carrier,
                forward_kwargs=forward_kwargs,
            )
            return self.prediction

        def sparse_post_process(self, output, sparse_inputs):
            self.post_process_prediction = sparse_inputs
            return output + 2

    monkeypatch.setattr(
        _CoreHandoffSparseAttention,
        "_prepare_metadata",
        lambda self, batch_size, seq_len: prepared_metadata,
    )
    monkeypatch.setattr(
        _CoreHandoffSparseAttention,
        "_concat_qkv",
        lambda self, q, k, v, batch_size, seq_len, kv_seq_len: torch.cat(
            [
                q.reshape(batch_size * seq_len, -1),
                k.reshape(batch_size * kv_seq_len, -1),
                v.reshape(batch_size * kv_seq_len, -1),
            ],
            dim=-1,
        ),
    )

    def _capture_base_forward(self, q, k, v, metadata, forward_args=None, **kwargs):
        captured.update(
            q=q,
            k=k,
            v=v,
            metadata=metadata,
            forward_args=forward_args,
            kwargs=kwargs,
        )
        return q[:, :16]

    monkeypatch.setattr(
        visual_trtllm.BaseTrtllmAttention,
        "forward",
        _capture_base_forward,
    )

    attention = object.__new__(_CoreHandoffSparseAttention)
    attention.quant_attention_config = None
    attention.predicted_carrier = _make_block_sparse_inputs() if has_block_sparse_inputs else None
    q = torch.randn(1, 4, 2, 8)
    k = torch.randn_like(q)
    v = torch.randn_like(q)

    output = attention.forward(q, k, v, batch_size=1, seq_len=4)

    forward_args = captured["forward_args"]
    assert isinstance(forward_args.sparse_attention_prediction, SparseAttentionPrediction)
    assert (
        forward_args.sparse_attention_prediction.block_sparse_inputs is attention.predicted_carrier
    )
    assert forward_args.block_sparse_inputs is None
    assert captured["metadata"] is prepared_metadata
    if has_block_sparse_inputs:
        torch.testing.assert_close(captured["q"], (q + 1).reshape(4, 16))
        assert captured["k"].data_ptr() == k.data_ptr()
        assert captured["v"].data_ptr() == v.data_ptr()
    else:
        torch.testing.assert_close(captured["q"][:, :16], (q + 1).reshape(4, 16))
        assert captured["k"] is None
        assert captured["v"] is None
    assert captured["kwargs"] == {}
    assert attention.post_process_prediction is attention.prediction
    torch.testing.assert_close(output, (q + 1).reshape(1, 4, 16) + 2)


@pytest.mark.parametrize("has_block_sparse_inputs", [False, True])
def test_sparse_lifecycle_reaches_core_fmha_with_precomputed_prediction(
    monkeypatch,
    has_block_sparse_inputs,
):
    class _CoreHandoffSparseAttention(visual_trtllm.TrtllmAttention):
        def block_sparse_attn_predict(
            self,
            q,
            k,
            v,
            *,
            batch_size,
            seq_len,
            seq_len_kv,
            attention_mask,
            forward_kwargs,
            block_sparse_inputs=None,
        ):
            assert block_sparse_inputs is None
            return visual_trtllm.SparseForwardInputs(
                q=q,
                k=k,
                v=v,
                batch_size=batch_size,
                seq_len=seq_len,
                seq_len_kv=seq_len_kv,
                attention_mask=attention_mask,
                block_sparse_inputs=self.predicted_carrier,
                forward_kwargs=forward_kwargs,
            )

    metadata = _make_core_forward_metadata()
    monkeypatch.setattr(
        _CoreHandoffSparseAttention,
        "_prepare_metadata",
        lambda self, batch_size, seq_len: metadata,
    )

    attention = object.__new__(_CoreHandoffSparseAttention)
    attention.quant_attention_config = None
    attention.sparse_params = None
    attention.is_mla_enable = False
    attention.num_heads = 2
    attention.num_kv_heads = 2
    attention.head_dim = 8
    attention.get_local_layer_idx = Mock(return_value=0)
    attention._ensure_rope_table_size = Mock()
    attention.print_skip_softmax_stat = False
    attention.kv_scale_orig_quant = None
    attention.kv_scale_quant_orig = None
    attention.fmha_libs = [object()]
    attention.predict_sparse_attention = Mock()
    attention._select_fmha = Mock(side_effect=_StopAtFmhaDispatch)
    attention.predicted_carrier = _make_block_sparse_inputs() if has_block_sparse_inputs else None
    q = torch.randn(1, 4, 2, 8)
    k = torch.randn_like(q)
    v = torch.randn_like(q)

    with pytest.raises(_StopAtFmhaDispatch):
        attention.forward(q, k, v, batch_size=1, seq_len=4)

    attention.predict_sparse_attention.assert_not_called()
    attention._select_fmha.assert_called_once()
    core_forward_args = attention._select_fmha.call_args.args[4]
    prediction = core_forward_args.sparse_attention_prediction
    assert isinstance(prediction, SparseAttentionPrediction)
    assert prediction.block_sparse_inputs is attention.predicted_carrier
    assert core_forward_args.block_sparse_inputs is attention.predicted_carrier
