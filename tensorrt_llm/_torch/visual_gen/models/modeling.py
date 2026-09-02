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
"""Base classes for VisualGen model components."""

import inspect
from typing import TYPE_CHECKING

import torch
import torch.nn as nn

from tensorrt_llm._torch.attention_backend.sparse.skip_softmax import SkipSoftmaxScheduler
from tensorrt_llm._torch.visual_gen.attention_backend.sparse.sol.params import SolParams
from tensorrt_llm._torch.visual_gen.config import DiffusionModelConfig
from tensorrt_llm.visual_gen.sparse_attention import SkipSoftmaxAttentionConfig, SolAttentionConfig

if TYPE_CHECKING:
    from tensorrt_llm._torch.visual_gen.cuda_graph_runner import CUDAGraphRunner


class BaseDiffusionModel(nn.Module):
    """Base class for TRT-LLM VisualGen model components."""

    def __init__(self, model_config: DiffusionModelConfig):
        super().__init__()
        self.model_config = model_config
        self.component_name = model_config.component_name
        self.pretrained_config = model_config.pretrained_config
        self._sol_timestep_arg_index = None
        sparse_config = model_config.attention.sparse_attention_config
        if isinstance(sparse_config, SolAttentionConfig) and (
            sparse_config.disabled_until_timestep is not None
        ):
            self._sol_timestep_arg_index = next(
                (
                    index
                    for index, parameter in enumerate(
                        tuple(inspect.signature(type(self).forward).parameters.values())[1:]
                    )
                    if parameter.name == "timestep"
                    and parameter.kind
                    in (
                        inspect.Parameter.POSITIONAL_ONLY,
                        inspect.Parameter.POSITIONAL_OR_KEYWORD,
                    )
                ),
                None,
            )

    def __call__(self, *args, **kwargs):
        """Scope a timestep-derived SOL phase around one model forward."""

        sparse_config = self.model_config.attention.sparse_attention_config
        if not isinstance(sparse_config, SolAttentionConfig):
            return super().__call__(*args, **kwargs)
        cutoff = sparse_config.disabled_until_timestep
        if cutoff is None:
            return super().__call__(*args, **kwargs)
        timestep = kwargs.get("timestep")
        if (
            "timestep" not in kwargs
            and self._sol_timestep_arg_index is not None
            and len(args) > self._sol_timestep_arg_index
        ):
            timestep = args[self._sol_timestep_arg_index]
        with SolParams.model_forward_phase_scope(
            timestep,
            disabled_until_timestep=cutoff,
        ):
            return super().__call__(*args, **kwargs)

    def forward(self, *args, timestep: torch.Tensor | None = None, **kwargs):
        """Run the diffusion transformer.

        Concrete VisualGen models own their full forward signatures. This base
        method defines the common arguments that every forward should accept.

        Args:
            timestep: Normalized denoising-time coordinate in ``[0, 1]``.
                Larger values correspond to earlier, noisier denoising steps.
                It may be ``None`` only for model paths that do not need a
                timestep-dependent model-forward decision.
                Model definers must pass the normalized value required by this
                contract and perform any conversion needed inside modules that
                reference this value. This TRT-LLM VisualGen contract
                intentionally differs from Diffusers' ``ModelMixin`` subclasses,
                where transformer ``timestep`` is model-specific. For example,
                WAN forwards raw integer scheduler timesteps in ``[0, 999]``,
                while FLUX forwards ``t / 1000``.
        """
        raise NotImplementedError("Diffusion model subclasses must implement forward().")

    def register_cuda_graph_extra_key_fns(self, runner: "CUDAGraphRunner") -> None:
        """Register CUDA graph key contributors that are not tensor shapes.

        Override this hook when a model.forward input changes captured
        execution without changing tensor shapes. Implementations should call
        ``runner.register_extra_key_fn(name, fn)``, where ``fn`` is the
        callback.

        The callback receives the wrapped model.forward ``*args`` and
        ``**kwargs`` and returns either a hashable key value or ``None``.
        If the callback returns ``None``, the runner omits that key part for
        the current call.

        Subclasses should call ``super()`` unless they intentionally replace
        the shared registrations.
        """
        sparse_config = self.model_config.attention.sparse_attention_config
        if isinstance(sparse_config, SkipSoftmaxAttentionConfig):
            disabled_until_timestep = sparse_config.resolve_disabled_until_timestep(
                pretrained_config=self.model_config.pretrained_config,
            )
            if disabled_until_timestep is None:
                return

            # Skip Softmax switches graph-visible attention behavior at the
            # timestep boundary while tensor shapes stay unchanged. Key the dense
            # and sparse phases separately; if timestep is absent or None, the
            # scheduler returns None and the runner omits this key part.
            runner.register_extra_key_fn(
                "skip_softmax_phase",
                lambda *args, **kwargs: SkipSoftmaxScheduler.get_graph_phase_for_timestep(
                    kwargs.get("timestep"),
                    disabled_until_timestep=disabled_until_timestep,
                ),
            )
            return

        if not isinstance(sparse_config, SolAttentionConfig):
            return
        disabled_until_timestep = sparse_config.disabled_until_timestep
        if disabled_until_timestep is None:
            return

        def _sol_graph_phase(*args, **kwargs) -> int | None:
            del args, kwargs
            phase = SolParams.get_scoped_graph_phase()
            if phase is None:
                raise RuntimeError("SOL graph key must be evaluated inside a model forward")
            return phase

        runner.register_extra_key_fn("sol_attn_phase", _sol_graph_phase)
