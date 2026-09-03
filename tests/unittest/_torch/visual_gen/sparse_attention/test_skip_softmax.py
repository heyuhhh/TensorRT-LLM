# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Tests for VisualGen Skip Softmax Attention config behavior."""

import json
import math
from types import SimpleNamespace
from typing import Optional

import pytest
import torch
import yaml
from pydantic import ValidationError

from tensorrt_llm._torch.attention_backend.sparse.skip_softmax import (
    SkipSoftmaxParams,
    SkipSoftmaxScheduler,
)
from tensorrt_llm._torch.attention_backend.trtllm import TrtllmAttention as CoreTrtllmAttention
from tensorrt_llm._torch.visual_gen.attention_backend.cute_dsl import fmha as cute_dsl_fmha
from tensorrt_llm._torch.visual_gen.attention_backend.cute_dsl.fmha import (
    CuTeDSLAttention,
    _resolve_skip_softmax_threshold_scale_factor,
)
from tensorrt_llm._torch.visual_gen.attention_backend.trtllm import (
    TrtllmAttention as VisualGenTrtllmAttention,
)
from tensorrt_llm._torch.visual_gen.attention_backend.utils import create_attention
from tensorrt_llm._torch.visual_gen.config import DiffusionModelConfig
from tensorrt_llm._torch.visual_gen.models.modeling import BaseDiffusionModel
from tensorrt_llm._torch.visual_gen.modules.attention import Attention
from tensorrt_llm.visual_gen.args import AttentionConfig, QuantAttentionConfig, VisualGenArgs
from tensorrt_llm.visual_gen.sparse_attention import SkipSoftmaxAttentionConfig


def _ckpt_sparse_attention_config(
    *,
    log_a: float = -10.0,
    b: float = 2.0,
    target_sparsity: Optional[object] = 0.5,
    ignore: Optional[list[str]] = None,
    disabled_until_timestep: Optional[float] = None,
) -> dict:
    # ModelOpt stores sparse-attention metadata under config_groups. These
    # tests vary one skip-softmax group without requiring a real checkpoint.
    group = {
        "algorithm": "skip_softmax",
        "threshold_scale_factor": {
            "formula": "exp(log_a + b * target_sparsity)",
            "coefficients": {
                "log_a": log_a,
                "b": b,
            },
        },
    }
    if target_sparsity is not None:
        group["target_sparsity"] = target_sparsity
    if ignore is not None:
        group["ignore"] = ignore
    if disabled_until_timestep is not None:
        group["disabled_until_timestep"] = disabled_until_timestep
    return {
        "config_groups": {
            "group_0": group,
        },
    }


def _checkpoint_config(**kwargs) -> dict:
    return {"sparse_attention_config": _ckpt_sparse_attention_config(**kwargs)}


def _skip_softmax_group(checkpoint_config: dict) -> dict:
    return checkpoint_config["sparse_attention_config"]["config_groups"]["group_0"]


def _expected_threshold(log_a: float, b: float, target_sparsity: float) -> float:
    return math.exp(log_a + b * target_sparsity)


def _prefill_threshold(
    sparse_params: Optional[SkipSoftmaxParams],
    *,
    timestep: Optional[float] = None,
) -> float:
    assert isinstance(sparse_params, SkipSoftmaxParams)
    return sparse_params.scheduler.get_runtime_params(
        timestep=timestep
    ).threshold_scale_factor_prefill


@pytest.mark.skipif(not torch.cuda.is_available(), reason="requires CUDA")
def test_precomputed_skip_softmax_phase_is_cuda_graph_safe():
    scheduler = SkipSoftmaxScheduler(
        threshold_scale_factor_prefill=5000.0,
        disabled_until_timestep=0.6,
    )
    timestep = torch.tensor([0.0, 0.2], device="cuda")
    source = torch.ones(1, device="cuda")

    graph = torch.cuda.CUDAGraph()
    with scheduler.model_forward_phase_scope(
        timestep,
        disabled_until_timestep=0.6,
    ):
        with torch.cuda.graph(graph):
            runtime_params = scheduler.get_runtime_params(timestep=timestep)
            captured = source + runtime_params.threshold_scale_factor_prefill

    graph.replay()
    torch.cuda.synchronize()
    torch.testing.assert_close(captured, torch.full_like(captured, 5001.0))
    assert SkipSoftmaxScheduler.get_scoped_graph_phase() is None


@pytest.mark.cpu_only
class TestVisualGenSkipSoftmaxUserAPI:
    """User-facing config surface: VisualGen args only expose runtime knobs."""

    def test_python_api_parses_skip_softmax_config(self):
        # Python users configure Skip Softmax Attention through
        # AttentionConfig.sparse_attention_config.
        config = AttentionConfig(
            backend="TRTLLM",
            sparse_attention_config={
                "algorithm": "skip_softmax",
                "threshold_scale_factor": 5000.0,
                "target_sparsity": 0.5,
                "disabled_until_timestep": 0.6,
            },
        )

        sparse_config = config.sparse_attention_config

        assert isinstance(sparse_config, SkipSoftmaxAttentionConfig)
        assert sparse_config.threshold_scale_factor == 5000.0
        assert sparse_config.target_sparsity == 0.5
        assert sparse_config.disabled_until_timestep == 0.6
        assert AttentionConfig(**config.model_dump()).model_dump() == config.model_dump()

    def test_cutedsl_api_accepts_skip_softmax_with_quantized_attention(self):
        config = AttentionConfig(
            backend="CUTEDSL",
            quant_attention_config=QuantAttentionConfig(qk_dtype="bf16", v_dtype="fp8"),
            sparse_attention_config=SkipSoftmaxAttentionConfig(
                threshold_scale_factor=5000.0,
                disabled_until_timestep=0.6,
            ),
        )

        assert isinstance(config.sparse_attention_config, SkipSoftmaxAttentionConfig)
        assert config.quant_attention_config is not None

    def test_yaml_api_parses_skip_softmax_config(self):
        # YAML config should deserialize to the same public config object as
        # the Python API.
        config_dict = yaml.safe_load("""
attention_config:
  backend: TRTLLM
  sparse_attention_config:
    algorithm: skip_softmax
    target_sparsity: 0.5
    disabled_until_timestep: 0.6
""")

        args = VisualGenArgs(model="/tmp/dummy_model", **config_dict)

        sparse_config = args.attention_config.sparse_attention_config
        assert isinstance(sparse_config, SkipSoftmaxAttentionConfig)
        assert sparse_config.target_sparsity == 0.5
        assert sparse_config.disabled_until_timestep == 0.6

    @pytest.mark.parametrize(
        ("field", "value"),
        [
            ("formula", "a * target_sparsity"),
            ("coefficients", {"a": 1.0}),
            ("ignore", ["blocks.0.attn1"]),
            ("config_groups", {}),
        ],
    )
    def test_checkpoint_metadata_fields_are_not_user_api_fields(self, field, value):
        # Calibration formula, coefficients, group selection, and layer ignore
        # patterns come from checkpoint config.json, not the public API.
        with pytest.raises(ValidationError):
            AttentionConfig(
                backend="TRTLLM",
                sparse_attention_config={
                    "algorithm": "skip_softmax",
                    "threshold_scale_factor": 5000.0,
                    field: value,
                },
            )

    def test_visualgen_target_sparsity_user_value_is_scalar(self):
        # VisualGen has no prefill/decode split, so target_sparsity is scalar.
        with pytest.raises(ValidationError):
            SkipSoftmaxAttentionConfig(
                target_sparsity={
                    "prefill": 0.5,
                    "decode": 0.3,
                }
            )

    @pytest.mark.parametrize(
        "kwargs",
        [
            {"target_sparsity": -0.1},
            {"target_sparsity": 1.1},
            {"disabled_until_timestep": -0.1},
            {"disabled_until_timestep": 1.1},
        ],
    )
    def test_normalized_user_values_must_be_in_unit_interval(self, kwargs):
        # target_sparsity and disabled_until_timestep are normalized values.
        with pytest.raises(ValidationError):
            SkipSoftmaxAttentionConfig(**kwargs)

    def test_direct_threshold_lowers_without_checkpoint_metadata(self):
        # Direct threshold configuration bypasses ModelOpt calibration.
        config = SkipSoftmaxAttentionConfig(threshold_scale_factor=5000.0)

        sparse_params = config.to_sparse_params()

        assert _prefill_threshold(sparse_params) == pytest.approx(5000.0)

    def test_threshold_scale_factor_takes_precedence_without_checkpoint_formula(self):
        # Raw threshold wins over target_sparsity, so no formula is needed.
        config = SkipSoftmaxAttentionConfig(
            threshold_scale_factor=5000.0,
            target_sparsity=1.0,
        )

        sparse_params = config.to_sparse_params()

        assert _prefill_threshold(sparse_params) == pytest.approx(5000.0)


@pytest.mark.cpu_only
class TestVisualGenSkipSoftmaxCheckpointConfig:
    """Checkpoint metadata: ModelOpt calibration is consumed at lowering time."""

    def test_user_target_sparsity_lowers_through_checkpoint_formula(self):
        # User target_sparsity overrides the checkpoint default but still uses
        # the checkpoint's calibrated formula.
        config = SkipSoftmaxAttentionConfig(target_sparsity=0.4)

        sparse_params = config.to_sparse_params(
            checkpoint_config=_checkpoint_config(
                log_a=math.log(7e-5),
                b=7.929109,
                target_sparsity=0.9,
            )
        )

        assert _prefill_threshold(sparse_params) == pytest.approx(7e-5 * math.exp(7.929109 * 0.4))

    def test_checkpoint_target_sparsity_default_lowers_through_formula(self):
        # If the user omits target_sparsity, the checkpoint default can drive
        # threshold resolution through the same formula.
        config = SkipSoftmaxAttentionConfig()

        sparse_params = config.to_sparse_params(
            checkpoint_config=_checkpoint_config(
                log_a=math.log(7e-5),
                b=7.929109,
                target_sparsity=0.5,
            )
        )

        assert _prefill_threshold(sparse_params) == pytest.approx(7e-5 * math.exp(7.929109 * 0.5))

    def test_checkpoint_formula_accepts_arbitrary_numexpr_expression(self):
        # The formula is evaluated by numexpr, so it is not restricted to the
        # exp(log_a + b * target_sparsity) shape used by the helper.
        config = SkipSoftmaxAttentionConfig()
        checkpoint_config = _checkpoint_config(target_sparsity=0.25)
        _skip_softmax_group(checkpoint_config)["threshold_scale_factor"] = {
            "formula": "sqrt(a + target_sparsity)",
            "coefficients": {
                "a": 0.75,
            },
        }

        sparse_params = config.to_sparse_params(checkpoint_config=checkpoint_config)

        assert _prefill_threshold(sparse_params) == pytest.approx(1.0)

    def test_target_sparsity_requires_checkpoint_formula(self):
        # target_sparsity is semantic; without a formula it cannot be converted
        # into the kernel-facing threshold_scale_factor.
        config = SkipSoftmaxAttentionConfig(target_sparsity=0.5)
        checkpoint_config = _checkpoint_config()
        del _skip_softmax_group(checkpoint_config)["threshold_scale_factor"]

        with pytest.raises(ValueError, match="calibration formula"):
            config.to_sparse_params(checkpoint_config=checkpoint_config)

    def test_checkpoint_phase_target_sparsity_dict_is_rejected(self):
        # prefill/decode target_sparsity dictionaries are LLM-only.
        config = SkipSoftmaxAttentionConfig()
        checkpoint_config = _checkpoint_config(target_sparsity={"prefill": 0.5})

        with pytest.raises(ValueError, match="prefill/decode phase dictionaries"):
            config.to_sparse_params(checkpoint_config=checkpoint_config)

    def test_other_sparse_attention_groups_are_ignored(self):
        # Checkpoints may include groups for several sparse algorithms. Only
        # the skip-softmax group should affect this config.
        config = SkipSoftmaxAttentionConfig(target_sparsity=0.5)
        checkpoint_config = _checkpoint_config(log_a=-10.0, b=2.0)
        groups = checkpoint_config["sparse_attention_config"]["config_groups"]
        groups["group_1"] = {
            "algorithm": "vsa",
            "target_sparsity": 0.8,
        }

        sparse_params = config.to_sparse_params(checkpoint_config=checkpoint_config)

        assert _prefill_threshold(sparse_params) == pytest.approx(
            _expected_threshold(-10.0, 2.0, 0.5)
        )

    def test_multiple_skip_softmax_checkpoint_groups_are_invalid(self):
        # Multiple skip-softmax groups make calibration ambiguous.
        config = SkipSoftmaxAttentionConfig(target_sparsity=0.5)
        checkpoint_config = _checkpoint_config()
        groups = checkpoint_config["sparse_attention_config"]["config_groups"]
        groups["group_1"] = {
            "algorithm": "vsa",
            "target_sparsity": 0.8,
        }
        groups["group_2"] = dict(groups["group_0"])

        with pytest.raises(ValueError, match="multiple skip-softmax"):
            config.to_sparse_params(checkpoint_config=checkpoint_config)


@pytest.mark.cpu_only
class TestVisualGenSkipSoftmaxLayerFiltering:
    """Layer filtering: checkpoint ignore patterns disable selected modules."""

    def test_checkpoint_ignore_patterns_disable_matching_attention_layers(self):
        # Patterns are matched against full names, component-relative names,
        # and names with torch.compile's _orig_mod wrapper removed.
        config = SkipSoftmaxAttentionConfig(threshold_scale_factor=5000.0)
        checkpoint_config = _checkpoint_config(ignore=["blocks.0.attn1", "*.attn2"])

        assert (
            config.to_sparse_params(
                module_name="transformer.blocks.0.attn1",
                checkpoint_config=checkpoint_config,
            )
            is None
        )
        assert (
            config.to_sparse_params(
                module_name="transformer_2.blocks.0._orig_mod.attn1",
                checkpoint_config=checkpoint_config,
            )
            is None
        )
        assert (
            config.to_sparse_params(
                module_name="transformer.blocks.3.attn2",
                checkpoint_config=checkpoint_config,
            )
            is None
        )
        assert _prefill_threshold(
            config.to_sparse_params(
                module_name="transformer.blocks.1.attn1",
                checkpoint_config=checkpoint_config,
            )
        ) == pytest.approx(5000.0)


@pytest.mark.cpu_only
class TestVisualGenSkipSoftmaxTimestepCutoff:
    """Timestep cutoff: early denoising can run with skip-softmax disabled."""

    def test_user_disabled_until_timestep_uses_normalized_timestep(self):
        # Denoising timesteps move from high to low. The cutoff disables
        # skip-softmax while timestep >= disabled_until_timestep.
        config = SkipSoftmaxAttentionConfig(
            threshold_scale_factor=5000.0,
            disabled_until_timestep=0.6,
        )
        sparse_params = config.to_sparse_params()

        assert _prefill_threshold(sparse_params, timestep=1.0) == pytest.approx(0.0)
        assert _prefill_threshold(sparse_params, timestep=0.6) == pytest.approx(0.0)
        assert _prefill_threshold(sparse_params, timestep=0.59) == pytest.approx(5000.0)

    def test_checkpoint_disabled_until_timestep_applies_when_user_omits_it(self):
        # Checkpoint metadata can provide a default cutoff when the user omits
        # the field.
        config = SkipSoftmaxAttentionConfig(threshold_scale_factor=5000.0)
        sparse_params = config.to_sparse_params(
            checkpoint_config=_checkpoint_config(disabled_until_timestep=0.6)
        )

        assert _prefill_threshold(sparse_params, timestep=1.0) == pytest.approx(0.0)
        assert _prefill_threshold(sparse_params, timestep=0.6) == pytest.approx(0.0)
        assert _prefill_threshold(sparse_params, timestep=0.59) == pytest.approx(5000.0)

    def test_user_disabled_until_timestep_overrides_checkpoint_default(self):
        # User config has higher priority than checkpoint defaults.
        config = SkipSoftmaxAttentionConfig(
            threshold_scale_factor=5000.0,
            disabled_until_timestep=0.8,
        )
        sparse_params = config.to_sparse_params(
            checkpoint_config=_checkpoint_config(disabled_until_timestep=0.4)
        )

        assert _prefill_threshold(sparse_params, timestep=0.7) == pytest.approx(5000.0)

    @pytest.mark.parametrize(
        ("timestep", "expected"),
        [
            (1.0, 0),
            (0.6, 0),
            (0.59, 1),
            (None, None),
        ],
    )
    def test_graph_phase_tracks_disabled_until_timestep_boundary(
        self,
        timestep,
        expected,
    ):
        # CUDA graph keys need a stable sparse-attention phase so captured
        # graphs are not reused across disabled and enabled skip-softmax states.
        assert (
            SkipSoftmaxScheduler.get_graph_phase_for_timestep(
                timestep,
                disabled_until_timestep=0.6,
            )
            == expected
        )

    @pytest.mark.parametrize(
        ("timestep", "expected"),
        [
            (torch.tensor([0.0, 0.8]), 0),
            (torch.tensor([0.0, 0.2]), 1),
        ],
    )
    def test_graph_phase_waits_for_every_token_to_cross_cutoff(
        self,
        timestep,
        expected,
    ):
        assert (
            SkipSoftmaxScheduler.get_graph_phase_for_timestep(
                timestep,
                disabled_until_timestep=0.6,
            )
            == expected
        )

    @pytest.mark.parametrize("pass_timestep_by_keyword", [False, True])
    def test_model_forward_resolves_skip_softmax_phase_once(
        self,
        monkeypatch,
        pass_timestep_by_keyword,
    ):
        sparse_params = SkipSoftmaxAttentionConfig(
            threshold_scale_factor=5000.0,
            disabled_until_timestep=0.6,
        ).to_sparse_params()

        class _Model(BaseDiffusionModel):
            def forward(self, hidden_states, timestep=None):
                return (
                    hidden_states,
                    sparse_params.scheduler.get_runtime_params(timestep=timestep),
                    sparse_params.scheduler.get_runtime_params(timestep=timestep),
                    SkipSoftmaxScheduler.get_scoped_graph_phase(),
                )

        model = _Model(
            DiffusionModelConfig(
                pretrained_config=SimpleNamespace(),
                attention=AttentionConfig(
                    backend="TRTLLM",
                    sparse_attention_config=SkipSoftmaxAttentionConfig(
                        threshold_scale_factor=5000.0,
                        disabled_until_timestep=0.6,
                    ),
                ),
            )
        )
        original_item = torch.Tensor.item
        item_calls = 0

        def _count_item(tensor, *args, **kwargs):
            nonlocal item_calls
            item_calls += 1
            return original_item(tensor, *args, **kwargs)

        monkeypatch.setattr(torch.Tensor, "item", _count_item)
        timestep = torch.tensor([0.0, 0.2])
        if pass_timestep_by_keyword:
            _, first, second, scoped_phase = model(
                torch.ones(1),
                timestep=timestep,
            )
        else:
            _, first, second, scoped_phase = model(torch.ones(1), timestep)

        assert item_calls == 1
        assert first.threshold_scale_factor_prefill == pytest.approx(5000.0)
        assert second.threshold_scale_factor_prefill == pytest.approx(5000.0)
        assert scoped_phase == 1
        assert SkipSoftmaxScheduler.get_scoped_graph_phase() is None

    def test_scoped_visual_phase_does_not_gate_scheduler_without_cutoff(self):
        static_scheduler = SkipSoftmaxScheduler(threshold_scale_factor_prefill=123.0)

        with SkipSoftmaxScheduler.model_forward_phase_scope(
            0.8,
            disabled_until_timestep=0.6,
        ):
            runtime_params = static_scheduler.get_runtime_params()

        assert runtime_params.threshold_scale_factor_prefill == pytest.approx(123.0)

    def test_skip_softmax_requires_precomputed_phase_during_capture(self, monkeypatch):
        scheduler = SkipSoftmaxScheduler(
            threshold_scale_factor_prefill=5000.0,
            disabled_until_timestep=0.6,
        )
        monkeypatch.setattr(torch.cuda, "is_available", lambda: True)
        monkeypatch.setattr(torch.cuda, "is_current_stream_capturing", lambda: True)

        with pytest.raises(RuntimeError, match="precomputed"):
            scheduler.get_runtime_params(timestep=torch.tensor(0.2))


@pytest.mark.cpu_only
class TestVisualGenSkipSoftmaxTrtllm:
    """TRTLLM keeps SkipSoftmax scheduling in the shared core backend."""

    @staticmethod
    def _patch_visual_trtllm_init(monkeypatch):
        def _base_init(self, **kwargs):
            self.sparse_params = kwargs["sparse_params"]

        monkeypatch.setattr(VisualGenTrtllmAttention, "__init__", _base_init)

    def test_factory_builds_thin_backend_with_shared_predictor(self, monkeypatch):
        self._patch_visual_trtllm_init(monkeypatch)
        attention_config = AttentionConfig(
            backend="TRTLLM",
            sparse_attention_config=SkipSoftmaxAttentionConfig(
                threshold_scale_factor=5000.0,
            ),
        )
        sparse_params = attention_config.sparse_attention_config.to_sparse_params()

        backend = create_attention(
            backend="TRTLLM",
            layer_idx=0,
            num_heads=2,
            head_dim=8,
            attention_config=attention_config,
            attention_metadata_state={},
            sparse_params=sparse_params,
        )

        backend_cls = type(backend)
        assert backend_cls.__name__ == "SkipSoftmaxTrtllmAttention"
        assert backend_cls.__module__.endswith(".sparse.skip_softmax.backend")
        assert backend.sparse_params is sparse_params
        assert (
            backend.predict_sparse_attention.__func__
            is CoreTrtllmAttention.predict_sparse_attention
        )
        assert "predict_sparse_attention" not in backend_cls.__dict__
        assert "block_sparse_attn_predict" not in backend_cls.__dict__
        assert "forward" not in backend_cls.__dict__
        assert "scheduler" not in backend_cls.__dict__

    def test_factory_keeps_ignored_skip_softmax_layer_dense(self, monkeypatch):
        self._patch_visual_trtllm_init(monkeypatch)
        attention_config = AttentionConfig(
            backend="TRTLLM",
            sparse_attention_config=SkipSoftmaxAttentionConfig(
                threshold_scale_factor=5000.0,
            ),
        )

        backend = create_attention(
            backend="TRTLLM",
            layer_idx=0,
            num_heads=2,
            head_dim=8,
            attention_config=attention_config,
            attention_metadata_state={},
            sparse_params=None,
        )

        assert type(backend) is VisualGenTrtllmAttention
        assert backend.sparse_params is None

    def test_factory_rejects_invalid_skip_softmax_params(self, monkeypatch):
        self._patch_visual_trtllm_init(monkeypatch)
        attention_config = AttentionConfig(
            backend="TRTLLM",
            sparse_attention_config=SkipSoftmaxAttentionConfig(
                threshold_scale_factor=5000.0,
            ),
        )

        with pytest.raises(TypeError, match="requires SkipSoftmaxParams"):
            create_attention(
                backend="TRTLLM",
                layer_idx=0,
                num_heads=2,
                head_dim=8,
                attention_config=attention_config,
                attention_metadata_state={},
                sparse_params=object(),
            )


@pytest.mark.cpu_only
class TestVisualGenSkipSoftmaxCuTeDSL:
    """CuTeDSL lowering and runtime scheduling use the shared SkipSoftmax params."""

    def test_attention_lowers_skip_softmax_params_for_cutedsl(self):
        sparse_config = SkipSoftmaxAttentionConfig(
            threshold_scale_factor=5000.0,
            disabled_until_timestep=0.6,
        )
        quant_config = QuantAttentionConfig(qk_dtype="bf16", v_dtype="fp8")
        model_config = DiffusionModelConfig(
            component_name="transformer",
            pretrained_config=SimpleNamespace(),
            attention=AttentionConfig(
                backend="CUTEDSL",
                quant_attention_config=quant_config,
                sparse_attention_config=sparse_config,
            ),
            skip_create_weights_in_init=True,
        )

        attention = Attention(
            hidden_size=16,
            num_attention_heads=2,
            head_dim=8,
            qk_norm=False,
            config=model_config,
            module_name="blocks.0.attn1",
        )

        assert isinstance(attention.attn, CuTeDSLAttention)
        assert isinstance(attention.sparse_params, SkipSoftmaxParams)
        assert attention.attn.sparse_params is attention.sparse_params
        assert attention.attn.quant_attention_config is quant_config

    @pytest.mark.parametrize(
        ("timestep", "expected"),
        [
            (1.0, None),
            (0.6, None),
            (0.59, 5000.0),
            (None, 5000.0),
        ],
    )
    def test_runtime_threshold_tracks_timestep(self, timestep, expected):
        sparse_params = SkipSoftmaxAttentionConfig(
            threshold_scale_factor=5000.0,
            disabled_until_timestep=0.6,
        ).to_sparse_params()

        threshold = _resolve_skip_softmax_threshold_scale_factor(
            None,
            sparse_params,
            timestep,
        )

        assert threshold == expected

    def test_cuda_graph_phase_uses_checkpoint_timestep_cutoff(self):
        sparse_config = SkipSoftmaxAttentionConfig(threshold_scale_factor=5000.0)
        model_config = DiffusionModelConfig(
            pretrained_config=SimpleNamespace(
                sparse_attention_config=_ckpt_sparse_attention_config(disabled_until_timestep=0.6)
            ),
            attention=AttentionConfig(
                backend="CUTEDSL",
                sparse_attention_config=sparse_config,
            ),
        )

        class _Model(BaseDiffusionModel):
            phase_fn = None

            def forward(self, hidden_states, *, timestep=None):
                del hidden_states, timestep
                assert self.phase_fn is not None
                return self.phase_fn()

        model = _Model(model_config)

        class _Runner:
            def __init__(self):
                self.extra_key_fns = {}

            def register_extra_key_fn(self, name, fn):
                self.extra_key_fns[name] = fn

        runner = _Runner()
        model.register_cuda_graph_extra_key_fns(runner)

        phase_fn = runner.extra_key_fns["skip_softmax_phase"]
        model.phase_fn = phase_fn
        assert model(torch.ones(1), timestep=torch.tensor(0.6)) == 0
        assert model(torch.ones(1), timestep=torch.tensor(0.59)) == 1
        assert SkipSoftmaxScheduler.get_scoped_graph_phase() is None
        with pytest.raises(RuntimeError, match="inside a model forward"):
            phase_fn()

    @pytest.mark.skipif(
        cute_dsl_fmha._cute_dsl_import_error is not None,
        reason="CuTe DSL runtime unavailable",
    )
    def test_forward_threads_timestep_and_sparse_params_to_kernel_call(self, monkeypatch):
        """The timestep-gating feature hinges on `_fwd`'s `kwargs.get("timestep")`
        reaching `cute_dsl_fmha_fwd` unchanged; nothing else in this chain would
        raise if that silently dropped to `None` (the scheduler would just apply
        the full threshold during early, high-noise steps -- a quality
        regression, not an error). This exercises `CuTeDSLAttention.forward`
        (the boundary `_attn_impl`/`Attention.forward` call into, and the one
        `Attention._attn_impl` also threads `timestep` through unchanged to)
        directly, monkeypatching the actual kernel launcher so no CUDA/cutlass
        runtime is required.
        """
        sparse_params = SkipSoftmaxAttentionConfig(
            threshold_scale_factor=5000.0,
            disabled_until_timestep=0.6,
        ).to_sparse_params()

        captured_kwargs = {}

        def _fake_cute_dsl_fmha_fwd(q, k, v, o, **kwargs):
            captured_kwargs.update(kwargs)
            o.zero_()

        monkeypatch.setattr(
            "tensorrt_llm._torch.visual_gen.attention_backend.cute_dsl.fmha.cute_dsl_fmha_fwd",
            _fake_cute_dsl_fmha_fwd,
        )

        attn = CuTeDSLAttention(
            layer_idx=0,
            num_heads=2,
            head_dim=8,
            sparse_params=sparse_params,
        )

        batch, seq_len, num_heads, head_dim = 1, 4, 2, 8
        q = torch.zeros(batch, seq_len, num_heads, head_dim, dtype=torch.bfloat16)
        k = torch.zeros(batch, seq_len, num_heads, head_dim, dtype=torch.bfloat16)
        v = torch.zeros(batch, seq_len, num_heads, head_dim, dtype=torch.bfloat16)

        attn.forward(q, k, v, timestep=0.59)

        assert captured_kwargs.get("timestep") == 0.59
        assert captured_kwargs.get("sparse_params") is sparse_params


@pytest.mark.cpu_only
class TestVisualGenSkipSoftmaxPipelineConfig:
    """Pipeline config: multi-transformer checkpoints keep metadata separated."""

    def test_pipeline_config_keeps_checkpoint_metadata_per_model(self, tmp_path):
        from tensorrt_llm._torch.visual_gen.config import DiffusionPipelineConfig

        # Construct a minimal Diffusers-style layout with two transformer
        # components, each with its own config.json.
        (tmp_path / "transformer").mkdir()
        (tmp_path / "transformer_2").mkdir()
        (tmp_path / "model_index.json").write_text(
            json.dumps(
                {
                    "_class_name": "WanPipeline",
                    "transformer": ["diffusers", "WanTransformer3DModel"],
                    "transformer_2": ["diffusers", "WanTransformer3DModel"],
                }
            ),
            encoding="utf-8",
        )
        (tmp_path / "transformer" / "config.json").write_text(
            json.dumps(
                _checkpoint_config(
                    ignore=["blocks.0.attn1"],
                    log_a=-10.0,
                    b=2.0,
                )
            ),
            encoding="utf-8",
        )
        (tmp_path / "transformer_2" / "config.json").write_text(
            json.dumps(_checkpoint_config(log_a=-20.0, b=4.0)),
            encoding="utf-8",
        )

        pipeline_config = DiffusionPipelineConfig.from_pretrained(
            str(tmp_path),
            args=VisualGenArgs(
                model=str(tmp_path),
                attention_config=AttentionConfig(
                    backend="TRTLLM",
                    sparse_attention_config=SkipSoftmaxAttentionConfig(
                        target_sparsity=0.5,
                        disabled_until_timestep=0.6,
                    ),
                ),
            ),
        )

        sparse_config = pipeline_config.attention.sparse_attention_config
        assert isinstance(sparse_config, SkipSoftmaxAttentionConfig)
        assert sparse_config.model_dump() == {
            "algorithm": "skip_softmax",
            "threshold_scale_factor": None,
            "target_sparsity": 0.5,
            "disabled_until_timestep": 0.6,
        }

        # The public sparse config is shared at pipeline level, while
        # checkpoint calibration metadata stays attached to each component.
        transformer_config = pipeline_config.model_configs["transformer"]
        transformer_2_config = pipeline_config.model_configs["transformer_2"]
        assert transformer_config.pretrained_config is not transformer_2_config.pretrained_config

        transformer_params = sparse_config.to_sparse_params(
            module_name="transformer.blocks.1.attn1",
            pretrained_config=transformer_config.pretrained_config,
        )
        transformer_2_params = sparse_config.to_sparse_params(
            module_name="transformer_2.blocks.1.attn1",
            pretrained_config=transformer_2_config.pretrained_config,
        )
        transformer_disabled_params = sparse_config.to_sparse_params(
            module_name="transformer.blocks.0.attn1",
            pretrained_config=transformer_config.pretrained_config,
        )

        # The same public target_sparsity resolves through different per-model
        # formulas, and ignore patterns apply only to the owning component.
        assert _prefill_threshold(transformer_params, timestep=0.6) == pytest.approx(0.0)
        assert _prefill_threshold(transformer_params, timestep=0.59) == pytest.approx(
            _expected_threshold(-10.0, 2.0, 0.5)
        )
        assert _prefill_threshold(transformer_2_params, timestep=0.59) == pytest.approx(
            _expected_threshold(-20.0, 4.0, 0.5)
        )
        assert transformer_disabled_params is None
