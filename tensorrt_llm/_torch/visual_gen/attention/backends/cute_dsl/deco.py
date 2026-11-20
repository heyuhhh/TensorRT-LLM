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
"""
DeCo-Attention CuTe DSL Backend for Visual Generation Models

Runs self-attention on the DeCo-Attention kernels (``cute_dsl_kernels.blackwell.deco_attention``):
each call runs the DeCo preparation and then the attention kernel of the recipe, two stateless
functions that work in a workspace buffer this backend hands them.

* ``qk_dtype="nvfp4", v_dtype="fp8"``: NVFP4 Q/K with compensation channels and FP8 PV, on SM100,
  SM103, SM107 (tcgen05 MMA) and SM120 (warp-level MMA).
* ``qk_dtype="nvfp4", v_dtype="nvfp4"``: the same Q/K with NVFP4 PV, on SM120.

The preparation rotates Q and K, quantizes them to NVFP4 and appends compensation channels for
the first-order quantization error of the scores; it smooths V over blocks of keys before
quantizing it. It estimates the compensation channels from a sample of the tokens: a model
whose sequence packs text or audio tokens with the video tokens passes ``token_tags``, and the
sample then takes half of its rows from the text and audio tokens instead of a draw over the
whole sequence, which leaves the few text and audio tokens almost unsampled.

The kernels take NHD layout ([B, S, H, D]) with D = 128, as many K/V heads as Q heads, queries
and keys of one sequence, bfloat16 and no mask.

Padded sequences run on their valid prefix and return zeros for the padded rows, as the TRTLLM
backend does: the prefix length comes from the caller's host ``valid_seq_len``, or from a
``key_padding_mask`` whose valid keys lead every batch row (HunyuanVideo 1.5's text padding),
which the backend reads back to the host once per mask tensor and runs row by row when the rows
differ. A mask that is not a prefix, and any mask under CUDA graph capture (its lengths cannot be
read), runs on the VANILLA backend.

Other calls run unquantized: a causal mask on the dense CuTe DSL FMHA (on its GPUs; a padded
call keeps its valid prefix); cross-attention and other dtypes on VANILLA.
"""

from typing import Any, Callable

import torch

from tensorrt_llm.logger import logger
from tensorrt_llm.visual_gen.args import QuantAttentionConfig

from .....attention.backends.interface import PredefinedAttentionMask
from ..interface import AttentionBackend, AttentionTensorLayout
from ..vanilla import VanillaAttention
from .fmha import SUPPORTED_GPU_ARCHS, CuTeDSLAttention, _cute_dsl_import_error

_deco_import_error: BaseException | None = None
try:
    from tensorrt_llm._torch.visual_gen.cute_dsl_kernels.blackwell import deco_attention
except (ImportError, OSError) as e:
    deco_attention = None
    _deco_import_error = e

# The token tags and key padding masks of this many recent requests are kept with their basis rows
# and prefix lengths.
_REQUESTS_KEPT = 4


class DeCoAttention(AttentionBackend):
    """DeCo-Attention backend for diffusion self-attention (see the module docstring).

    The DeCo layers of one model component share one workspace through
    ``attention_metadata_state``: the layers run one after another on one stream, the kernels keep
    nothing in the workspace between calls, and each call writes its output into its own tensor.
    The state also keeps what the layers of a request share: the basis rows of its token tags
    and the prefix lengths of its key padding mask.

    The kernels compile, and the inputs of the preparation that depend only on the sequence length
    or the device are built, on first use. A CUDA graph capture therefore follows an eager forward
    of the same shape, which the CUDA graph runner's warm-up provides; a first use under capture
    raises a ``RuntimeError``.
    """

    def __init__(
        self,
        layer_idx: int = 0,
        num_heads: int = 8,
        head_dim: int = 128,
        num_kv_heads: int | None = None,
        dtype: torch.dtype | None = None,
        quant_attention_config: QuantAttentionConfig | None = None,
        attention_metadata_state: dict[str, Any] | None = None,
        **kwargs,
    ):
        if _deco_import_error is not None:
            raise ImportError(
                f"DeCo-Attention kernels are not available. Import error: {_deco_import_error}"
            ) from _deco_import_error
        if quant_attention_config is None or not quant_attention_config.deco_attention:
            raise ValueError(
                "DeCoAttention needs a quant_attention_config with deco_attention=True."
            )
        num_kv_heads = num_kv_heads or num_heads
        kernel_head_dim = deco_attention.constants.HEAD_DIM
        if head_dim != kernel_head_dim or num_kv_heads != num_heads:
            raise ValueError(
                f"DeCo-Attention needs head_dim {kernel_head_dim} and as many K/V heads as Q "
                f"heads; got head_dim {head_dim}, {num_heads} Q heads and {num_kv_heads} K/V heads."
            )
        # The V format of the recipe names the PV format of the kernel; Q/K are always NVFP4.
        try:
            pv_formats = deco_attention.pv_formats()
        except NotImplementedError as e:
            raise ValueError(
                "DeCo-Attention has kernels for SM100, SM103, SM107 and SM120 GPUs only."
            ) from e
        qk_dtype, v_dtype = quant_attention_config.qk_dtype, quant_attention_config.v_dtype
        if qk_dtype != "nvfp4" or v_dtype not in pv_formats:
            raise ValueError(
                f"DeCo-Attention has no kernel for qk_dtype={qk_dtype!r}, v_dtype={v_dtype!r} on "
                f"this GPU, whose kernels take qk_dtype='nvfp4' with v_dtype in {pv_formats}."
            )

        if attention_metadata_state is None:
            raise ValueError(
                "DeCoAttention requires `attention_metadata_state` from DiffusionModelConfig: "
                "the layers of a model component share one workspace buffer."
            )

        self.layer_idx = layer_idx
        self.num_heads = num_heads
        self.head_dim = head_dim
        self.num_kv_heads = num_kv_heads
        self.dtype = dtype
        self.quant_attention_config = quant_attention_config
        self._state = attention_metadata_state
        self._noted_unquantized = False
        self._preferred_layout = AttentionTensorLayout.NHD
        backend_args = dict(
            layer_idx=layer_idx,
            num_heads=num_heads,
            head_dim=head_dim,
            num_kv_heads=num_kv_heads,
            dtype=dtype,
        )
        self._vanilla = VanillaAttention(**backend_args)
        # The dense CuTe DSL FMHA takes the self-attention calls the DeCo kernels do not (a causal
        # mask) with this backend's own kernels; VANILLA where it is unavailable
        # or the GPU is not one of its architectures.
        self._dense = self._vanilla
        if _cute_dsl_import_error is None and torch.cuda.is_available():
            capability = torch.cuda.get_device_capability()
            if f"sm_{capability[0]}{capability[1]}a" in SUPPORTED_GPU_ARCHS:
                self._dense = CuTeDSLAttention(**backend_args)

    def _unserved(
        self,
        q: torch.Tensor,
        k: torch.Tensor,
        v: torch.Tensor,
        lengths: list[int] | None,
        **kwargs: Any,
    ) -> str | None:
        """Why the DeCo kernels do not take this call, or None when they do; ``lengths`` is the
        valid prefix of every batch row when the call is padded."""
        if not q.is_cuda:
            return "inputs that are not on a CUDA device"
        if q.numel() == 0:
            return "empty inputs"
        if not q.shape == k.shape == v.shape:
            return "cross-attention"
        if not q.dtype == k.dtype == v.dtype == torch.bfloat16:
            return f"{q.dtype} inputs"
        if kwargs.get("attention_mask", PredefinedAttentionMask.FULL) != (
            PredefinedAttentionMask.FULL
        ):
            return "an attention mask"
        if kwargs.get("key_padding_mask") is not None and lengths is None:
            if torch.cuda.is_current_stream_capturing():
                return "a key padding mask under CUDA graph capture"
            return "a key padding mask whose valid keys are not a prefix"
        return None

    def _delegate(
        self,
        q: torch.Tensor,
        k: torch.Tensor,
        v: torch.Tensor,
        dense: bool,
        lengths: list[int] | None = None,
        **kwargs: Any,
    ) -> torch.Tensor:
        """Run a call the kernels do not take: on the dense CuTe DSL FMHA (``dense``: bfloat16
        self-attention with a full or causal mask, over the valid prefixes ``lengths`` of a padded
        call), else on VANILLA, which works in HND ([B, H, S, D]) and applies a key padding mask
        itself.

        A model's cross-attention layers take this path on every call, inside the compiled
        model, so it logs nothing; ``forward`` warns once when a self-attention call takes it.
        """
        if dense and self._dense is not self._vanilla:
            if lengths is None:
                return self._dense.forward(q, k, v, **kwargs)

            def attend(q: torch.Tensor, k: torch.Tensor, v: torch.Tensor, out: torch.Tensor):
                out.copy_(self._dense.forward(q, k, v, **kwargs))

            return self._launch(q, k, v, attend, lengths)
        out = self._vanilla.forward(
            q.transpose(1, 2), k.transpose(1, 2), v.transpose(1, 2), **kwargs
        )
        return out.transpose(1, 2)

    @torch.compile
    def _compact_qkv(
        self, q: torch.Tensor, k: torch.Tensor, v: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        # Slices of a fused QKV projection are strided and the kernels read gap-free
        # [B, S, H, 128] tensors; the compiled copy keeps them on a vectorized kernel, while
        # already contiguous inputs pass through without a copy.
        return q.contiguous(), k.contiguous(), v.contiguous()

    def _workspace(self, num_bytes: int, device: torch.device) -> torch.Tensor:
        """The workspace of a call: the component's buffer, grown to its largest call.

        A CUDA graph keeps the addresses it was captured with, so a call under capture takes
        its workspace from the graph's memory pool instead, like its other temporaries.
        """
        if torch.cuda.is_current_stream_capturing():
            return torch.empty(num_bytes, dtype=torch.uint8, device=device)
        workspace = self._state.get("deco_workspace")
        if workspace is None or workspace.numel() < num_bytes or workspace.device != device:
            # Release the smaller buffer before allocating the larger one.
            workspace = self._state["deco_workspace"] = None
            workspace = torch.empty(num_bytes, dtype=torch.uint8, device=device)
            self._state["deco_workspace"] = workspace
        return workspace

    @torch.compiler.disable
    def _prefix_lengths(self, mask: torch.Tensor) -> list[int] | None:
        """The valid prefix length of every batch row of a [B, S] key padding mask (True =
        valid), or None when a row's valid keys are not a prefix. Read back to the host once per
        mask tensor, which the layers and steps of a request share, and kept with it in the
        component's state; not possible under CUDA graph capture, where it returns None."""
        if torch.cuda.is_current_stream_capturing():
            return None
        prefixes = self._state.setdefault("deco_prefix_lengths", [])
        for kept, lengths in prefixes:
            if kept is mask:
                return lengths
        valid = mask.bool()
        counts = valid.sum(1)
        positions = torch.arange(valid.shape[1], device=valid.device)
        is_prefix = bool((valid == (positions[None, :] < counts[:, None])).all())
        lengths = counts.tolist() if is_prefix else None
        prefixes.insert(0, (mask, lengths))
        del prefixes[_REQUESTS_KEPT:]
        return lengths

    def _basis_rows(
        self, token_tags: torch.Tensor | None, seq_len: int, device: torch.device
    ) -> torch.Tensor | None:
        """The tokens the preparation estimates its basis from: None, its own draw over the
        whole sequence, without token tags; with them, the draw by modality of the kernels'
        ``sample_rows`` over the first ``seq_len`` tags (a valid prefix), half of it from the
        text and audio tokens.

        The draw by modality reads the token counts back to the host, so it runs once per tags
        tensor and length and outside CUDA graph capture: every layer and step of a request
        passes the same tensor, and the component state keeps it with its rows, so that no
        other tensor takes its identity meanwhile.
        """
        if token_tags is None:
            return None
        if token_tags.numel() < seq_len:
            raise ValueError(
                f"DeCo-Attention got {token_tags.numel()} token tags for a sequence of "
                f"{seq_len} tokens."
            )
        tagged_rows = self._state.setdefault("deco_tagged_rows", [])
        for tags, length, rows in tagged_rows:
            if tags is token_tags and length == seq_len:
                return rows
        rows = deco_attention.sample_rows(seq_len, device, token_tags[:seq_len])
        tagged_rows.insert(0, (token_tags, seq_len, rows))
        del tagged_rows[_REQUESTS_KEPT:]
        return rows

    def _attend(
        self,
        q: torch.Tensor,
        k: torch.Tensor,
        v: torch.Tensor,
        out: torch.Tensor,
        token_tags: torch.Tensor | None,
    ) -> None:
        """The preparation and the attention of contiguous q, k, v [B, S, H, 128] into out."""
        batch_size, seq_len, num_heads, _ = q.shape
        pv = self.quant_attention_config.v_dtype
        num_bytes = deco_attention.workspace_size(batch_size, seq_len, num_heads, q.device, pv)
        rows = self._basis_rows(token_tags, seq_len, q.device)
        operands = deco_attention.prepare(
            q, k, v, self._workspace(num_bytes, q.device), pv, rows=rows
        )
        deco_attention.attention(operands, out)

    @torch.compiler.disable
    def _launch(
        self,
        q: torch.Tensor,
        k: torch.Tensor,
        v: torch.Tensor,
        attend: Callable[[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor], None],
        lengths: list[int] | None,
    ) -> torch.Tensor:
        """``attend`` (q, k, v into out) over the valid prefix of every batch row (``lengths``,
        one per row; None: the whole sequence), zeros on the padded rows. A prefix of one batch
        row is a contiguous view, so the rows of a padded call run one launch each."""
        seq_len = q.shape[1]
        if lengths is None or all(n == seq_len for n in lengths):
            out = torch.empty_like(q)
            attend(q, k, v, out)
            return out
        out = torch.zeros_like(q)
        for b, n in enumerate(lengths):
            if n > 0:
                rows = slice(b, b + 1)
                attend(q[rows, :n], k[rows, :n], v[rows, :n], out[rows, :n])
        return out

    @torch.compiler.disable
    def _note_unquantized(self, reason: str) -> None:
        """Warn, once per layer, that one of its self-attention calls is delegated."""
        self._noted_unquantized = True
        logger.warning_once(
            f"DeCo-Attention runs self-attention unquantized because of {reason} (first seen "
            f"in layer {self.layer_idx}); its kernels take bfloat16 self-attention inputs without a "
            f"mask, or with a valid-prefix key padding mask.",
            key=f"deco_attention_unquantized_{reason}",
        )

    def forward(
        self,
        q: torch.Tensor,
        k: torch.Tensor,
        v: torch.Tensor,
        **kwargs,
    ) -> torch.Tensor:
        """Attention over q, k, v ``[B, S, H, D]``; returns ``[B, S, H, D]`` in q's dtype.

        Keyword arguments the backend reads: ``attention_mask`` (FULL runs on the kernels, CAUSAL
        on the dense FMHA); ``valid_seq_len`` and ``key_padding_mask``, the valid prefix of a
        padded sequence (module docstring); ``token_tags`` (``[S]``, ``AttentionBackend.forward``),
        the modality of every token of a packed multimodal sequence, by which the preparation
        draws its basis rows. The rest is ignored.
        """
        self_attention = q.shape == k.shape
        lengths = None
        valid_seq_len = kwargs.get("valid_seq_len")
        key_padding_mask = kwargs.get("key_padding_mask")
        if valid_seq_len is not None:
            lengths = [int(valid_seq_len)] * q.shape[0]
        elif key_padding_mask is not None and self_attention:
            lengths = self._prefix_lengths(key_padding_mask)
        reason = self._unserved(q, k, v, lengths, **kwargs)
        if reason is None:
            token_tags = kwargs.get("token_tags")

            def attend(q: torch.Tensor, k: torch.Tensor, v: torch.Tensor, out: torch.Tensor):
                self._attend(q, k, v, out, token_tags)

            return self._launch(*self._compact_qkv(q, k, v), attend, lengths)
        if self_attention and not self._noted_unquantized:
            self._note_unquantized(reason)
        dense = self_attention and q.is_cuda and q.numel() > 0 and q.dtype == torch.bfloat16
        dense = dense and (key_padding_mask is None or lengths is not None)
        return self._delegate(q, k, v, dense, lengths if dense else None, **kwargs)

    @property
    def preferred_layout(self) -> AttentionTensorLayout:
        return self._preferred_layout

    @classmethod
    def support_fused_qkv(cls) -> bool:
        return False
