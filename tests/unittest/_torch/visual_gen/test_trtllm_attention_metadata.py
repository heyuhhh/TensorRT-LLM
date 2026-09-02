# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from dataclasses import FrozenInstanceError, dataclass, replace

import pytest
import torch

from tensorrt_llm._torch.attention_backend.block_sparse import BlockSparseForwardInputs
from tensorrt_llm._torch.attention_backend.interface import PredefinedAttentionMask
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
    _enable_sparse_workflow = True

    def sparse_predict(
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
    ):
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
            "kwargs": kwargs,
        }
        return q.reshape(batch_size, seq_len, -1)

    def sparse_post_process(self, output, prediction):
        self.lifecycle_events.append("post_process")
        self.post_process_prediction = prediction
        return output + 2


class _PredicateBypassSparseAttention(_RecordingSparseAttention):
    def _should_use_sparse_workflow(self, forward_kwargs):
        self.lifecycle_events.append("predicate")
        self.predicate_kwargs = forward_kwargs
        forward_kwargs.pop("private_sparse_phase", None)
        return False


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
    class _TypedSparseAttention(visual_trtllm.TrtllmAttention[_ExtendedSparseForwardInputs]):
        _enable_sparse_workflow = True

        def sparse_predict(
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
        ) -> _ExtendedSparseForwardInputs:
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


def test_sparse_lifecycle_requires_concrete_predictor():
    class _MissingPredictorAttention(visual_trtllm.TrtllmAttention):
        _enable_sparse_workflow = True

    attention = object.__new__(_MissingPredictorAttention)
    q = torch.randn(1, 4, 2, 8)

    with pytest.raises(NotImplementedError, match="must implement sparse_predict"):
        attention.forward(q, None, None, batch_size=1, seq_len=4)


def test_sparse_lifecycle_predicts_block_inputs_then_runs_normal_forward():
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
    assert attention.impl_inputs["block_sparse_inputs"] is attention.predicted_carrier
    assert attention.impl_inputs["kwargs"] == {
        "timestep": timestep,
        "gate_compress": gate_compress,
    }
    assert attention.post_process_prediction is attention.prediction
    torch.testing.assert_close(output, (q + 1).reshape(1, 4, -1) + 2)


def test_sparse_lifecycle_predicate_false_skips_prediction_and_cleans_kwargs():
    attention = object.__new__(_PredicateBypassSparseAttention)
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

    assert attention.lifecycle_events == ["predicate", "forward_impl"]
    assert attention.predicate_kwargs["timestep"] is timestep
    assert "private_sparse_phase" not in attention.impl_inputs["kwargs"]
    torch.testing.assert_close(output, q.reshape(1, 4, -1))


def test_sparse_lifecycle_rejects_caller_block_sparse_inputs():
    attention = object.__new__(_RecordingSparseAttention)
    attention.lifecycle_events = []
    attention.predicted_carrier = _make_block_sparse_inputs()
    q = torch.randn(1, 4, 2, 8)

    with pytest.raises(ValueError, match="must be produced by sparse_predict"):
        attention.forward(
            q,
            q,
            q,
            batch_size=1,
            seq_len=4,
            block_sparse_inputs=_make_block_sparse_inputs(),
        )

    assert attention.lifecycle_events == []


def test_dense_trtllm_attention_skips_sparse_hooks(monkeypatch):
    captured = {}
    q = torch.randn(1, 4, 2, 8)

    def _fail_hook(*args, **kwargs):
        raise AssertionError(f"dense fast path invoked a sparse hook: {args}, {kwargs}")

    def _capture_impl(self, *args, **kwargs):
        captured.update(args=args, kwargs=kwargs)
        return q.reshape(1, 4, -1)

    monkeypatch.setattr(visual_trtllm.TrtllmAttention, "sparse_predict", _fail_hook, raising=False)
    monkeypatch.setattr(
        visual_trtllm.TrtllmAttention, "sparse_post_process", _fail_hook, raising=False
    )
    monkeypatch.setattr(
        visual_trtllm.TrtllmAttention, "_forward_impl", _capture_impl, raising=False
    )

    attention = object.__new__(visual_trtllm.TrtllmAttention)
    output = attention.forward(q, None, None, batch_size=1, seq_len=4)

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
    assert captured["kwargs"] == {}
