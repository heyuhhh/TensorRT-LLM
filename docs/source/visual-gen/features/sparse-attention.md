# VisualGen Sparse Attention

```{note}
This page is an unindexed draft until the VisualGen documentation hub is introduced.
```

- [Overview](#overview)
  - [Algorithms](#algorithms)
- [Skip Softmax Attention](#skip-softmax-attention)
- [Video Sparse Attention (VSA)](#video-sparse-attention-vsa)

## Overview

Visual generation models naturally operate on long image or video token sequences. Each denoising step is closer to a full-context prefill pass than to autoregressive decoding, and attention can dominate runtime for high-resolution image generation or long video generation.

Sparse attention in VisualGen is configured through `VisualGenArgs.attention_config.sparse_attention_config`. The user-facing config stays in VisualGen args or model config. Checkpoint calibration metadata remains internal and is lowered into per-attention-backend `SparseParams` when each attention module is constructed.

### Algorithms

| `algorithm` | Config class | Status |
|---|---|---|
| `skip_softmax` | `SkipSoftmaxAttentionConfig` | Supported |
| `vsa` | `VideoSparseAttentionConfig` | Prototype |

## Skip Softmax Attention

Skip Softmax Attention is a kernel-level method, also known as BLASST, that dynamically skips computation in a FlashAttention-style kernel. It can accelerate existing full-attention VisualGen models in a plug-and-play manner.

The value actually consumed by the kernel is **`threshold_scale_factor`**. The kernel combines it with the **sequence length** to compute the **threshold** at runtime. Other configuration paths resolve to that scalar before the attention backend is constructed.

### Checkpoint Config

[NVIDIA Model Optimizer](https://github.com/NVIDIA/Model-Optimizer) (ModelOpt) can perform calibration and store metadata for Skip Softmax Attention in the model checkpoint's `config.json`. The checkpoint config provides the formula that maps `target_sparsity` to `threshold_scale_factor`.

This checkpoint config is **optional**. It is only required when using `target_sparsity`, which is a [0, 1] scalar that is more intuitive than directly choosing the kernel-facing `threshold_scale_factor`. `target_sparsity` only serves as guidance; the actual **achieved** sparsity in the kernel can vary.

Example checkpoint config:

```json
{
  "sparse_attention_config": {
    "config_groups": {
      "group_0": {
        "algorithm": "skip_softmax",
        "threshold_scale_factor": {
          "formula": "a * exp(b * target_sparsity)",
          "coefficients": {
            "a": 1000.0,
            "b": 5.0
          }
        },
        "target_sparsity": 0.5,
        "disabled_until_timestep": 0.8,
        "ignore": [
          "blocks.0.attn1",
          "blocks.0.attn2"
        ]
      }
    }
  }
}
```

The checkpoint config may contain multiple `config_groups` for different sparse attention algorithms. At most one group may configure Skip Softmax Attention. Multiple groups whose `algorithm` is `skip_softmax` are invalid.

- `formula` — an **arbitrary** [numexpr](https://numexpr.readthedocs.io/) expression of `threshold_scale_factor` using `target_sparsity` and one or more named coefficients. Standard math functions such as `exp`, `log`, `sqrt`, `pow`, and `**` are available. The runtime parses and evaluates it directly, so calibration is not locked to a fixed functional form.
- `coefficients` — scalar coefficient values referenced by `formula`.
- `target_sparsity` — optional checkpoint-provided target value. User-provided `target_sparsity` overrides this checkpoint default.
- `disabled_until_timestep` — optional normalized `[0, 1]` transformer-forward timestep cutoff. Denoising starts near 1 and moves toward 0, so Skip Softmax Attention is disabled while `timestep >= disabled_until_timestep` and enabled after the timestep drops below the cutoff.
- `ignore` — optional fnmatch layer patterns where the calibrated Skip Softmax Attention config should not apply. Patterns match both full module names and component-relative names, so `blocks.0.attn1` matches `transformer.blocks.0.attn1` and `transformer_2.blocks.0.attn1`.

Diffusers checkpoints with multiple transformer components keep calibration per component:

```text
checkpoint/
  model_index.json
  transformer/config.json
  transformer_2/config.json
```

Each component reads its own `config.json`, so formulas and `ignore` patterns can differ between `transformer` and `transformer_2`.

### User Configuration

User configuration is supplied through Python or YAML and controls how the checkpoint metadata is consumed:

- Set `threshold_scale_factor` directly to pass a concrete threshold to the kernel. This does not require checkpoint calibration metadata.
- Set `target_sparsity` to request a sparsity target. The runtime resolves it to `threshold_scale_factor` using the checkpoint calibration formula. If the checkpoint does not provide the required Skip Softmax Attention metadata, the runtime raises an error.
- Set `disabled_until_timestep` to disable Skip Softmax Attention at the beginning of denoising. This cutoff is normalized and therefore independent of the user-selected number of denoising steps.

`threshold_scale_factor` and `target_sparsity` are alternatives: if both are present, `threshold_scale_factor` takes precedence and the calibration formula is not used. User-provided `target_sparsity` and `disabled_until_timestep` override checkpoint defaults. Checkpoint `ignore` patterns always disable Skip Softmax Attention for matching layers.

Skip Softmax Attention only works with the **TRTLLM** attention backend in VisualGen. Set `attention_config.backend` to `TRTLLM` when enabling it.

#### Python API

```python
from tensorrt_llm.visual_gen import (
    AttentionConfig,
    SkipSoftmaxAttentionConfig,
    VisualGen,
    VisualGenArgs,
)

# Direct threshold:
args = VisualGenArgs(
    model="<path_or_hf_id>",
    attention_config=AttentionConfig(
        backend="TRTLLM",
        sparse_attention_config=SkipSoftmaxAttentionConfig(
            threshold_scale_factor=5000.0,
        ),
    ),
)

pipe = VisualGen(args)
```

```python
# Target sparsity (requires a calibrated checkpoint):
args = VisualGenArgs(
    model="<path_or_hf_id>",
    attention_config=AttentionConfig(
        backend="TRTLLM",
        sparse_attention_config=SkipSoftmaxAttentionConfig(
            target_sparsity=0.5,
            disabled_until_timestep=0.6,
        ),
    ),
)
```

#### YAML

```yaml
# Direct threshold:
attention_config:
  backend: TRTLLM
  sparse_attention_config:
    algorithm: skip_softmax
    threshold_scale_factor: 5000.0
```

```yaml
# Target sparsity (requires a calibrated checkpoint):
attention_config:
  backend: TRTLLM
  sparse_attention_config:
    algorithm: skip_softmax
    target_sparsity: 0.5
    disabled_until_timestep: 0.6
```

### CUDA Graphs

`disabled_until_timestep` creates two sparse-attention phases when it is set: the high-timestep disabled phase and the enabled phase after the cutoff. VisualGen includes that phase in CUDA graph keys so graph capture does not reuse a graph across different Skip Softmax Attention settings. See [VisualGen CUDA Graphs](cuda-graph.md) for the general capture and replay design.

Graphs are captured lazily. The first denoising step seen for a given tensor shape and sparse-attention phase captures a graph; later steps with the same shape and phase replay that graph. When denoising crosses the cutoff, the phase key changes, so VisualGen captures a second graph for the enabled phase instead of replaying the graph from the disabled phase.

## Video Sparse Attention (VSA)

VSA uses dense coarse attention over mean-pooled 4×4×4 video-token cubes to
select the top-K K/V cubes for each query cube. A backend-specific block-sparse
fine stage then attends to those selected cubes, and learned checkpoint gates
combine the coarse and fine outputs.

Set `attention_config.backend` to choose the implementation:

| Backend | Fine stage | Notes |
|---|---|---|
| `TRTLLM` | TRT-LLM PrimTS through FlashInfer's block-sparse wrapper | Recommended; uses exact K/V token validity and supports plan-once CUDA graph replay. |
| `CUTEDSL` | Original CuTe DSL VSA kernel | Compatibility/comparison path with legacy per-block padding-count semantics; supports CUDA graph capture after JIT warmup. |

The TRT-LLM path requires a FlashInfer build exporting
`flashinfer.attention.prims_ts.block_sparse.BlockSparseTSWrapper`, including
the `plan(..., dynamic_metadata=True)` capability. A missing capability emits
a warning and falls back to dense SDPA. Its
current kernel envelope is inference-only compact BSHD MHA with FP16/BF16
tensors, head dimension 128, block sizes divisible by 64, and SM100 or SM103
GPUs. Other inputs also fall back to dense SDPA.

The wrapper accepts a boolean `kv_token_mask` at the VisualGen adapter boundary
and packs it into FlashInfer's LSB-first `kv_valid_bits`. VSA derives this mask
from the exact cube-major padding map, including non-contiguous valid tokens in
boundary cubes. The `CUTEDSL` compatibility path receives only each block's
valid-token count, so use the `TRTLLM` path when exact boundary-token masking is
required.

```python
from tensorrt_llm.visual_gen import (
    AttentionConfig,
    VideoSparseAttentionConfig,
    VisualGenArgs,
)

args = VisualGenArgs(
    model="FastVideo/Wan2.1-VSA-T2V-14B-720P-Diffusers",
    attention_config=AttentionConfig(
        backend="TRTLLM",
        sparse_attention_config=VideoSparseAttentionConfig(vsa_sparsity=0.9),
    ),
)
```

`vsa_sparsity` is the fraction of K/V cubes skipped by the fine branch. VSA
requires its fine-tuned checkpoint gates and does not support Ring Attention or
Attention2D because it does not return per-split LSE. Ulysses is supported.

### CUDA Graphs

TRT-LLM VSA supports CUDA graphs with a plan-once/run-many lifecycle. The two
eager runner warmups create one FlashInfer plan per static tensor geometry and
allocate fixed-address route and token-mask workspaces. Capture records route
canonicalization, token-mask packing, in-place workspace refreshes, and
`run()`, so replay observes per-forward top-K routes and fine-grained padding
masks. A plan-cache miss during capture is rejected; run an eager warmup for
each new geometry first.

The model-scoped workspace is shared across serialized transformer layers to
plan once and avoid one large route allocation per layer. Consequently, calls
on the same model instance must remain ordered on one CUDA stream; concurrent
replays need separate model/adapter instances. Pipeline cleanup resets captured
graphs before releasing all cached VSA plans.

FlashInfer requires strictly increasing block IDs in each BSR row. The adapter
therefore canonicalizes VSA's top-K routes on every eager call; CUDA graph
capture records the same sort so replay continues to consume updated routes.
