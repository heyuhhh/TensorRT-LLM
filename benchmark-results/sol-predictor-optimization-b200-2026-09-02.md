<!-- SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved. -->
<!-- SPDX-License-Identifier: Apache-2.0 -->

# B200 SOL predictor optimization

Date: 2026-09-02 UTC

## Summary

This change optimizes only the first stage of the two-stage SOL path:

```text
Q/K/V -> SOL predictor -> exact-block bitmap + K/V summaries
      -> generic PrimTS block-sparse attention
```

The PrimTS attention kernel is unchanged. The predictor still owns no attention
wrapper and continues to publish graph-stable live tensors to the generic FMHA.

On the Wan 14B predictor shape (`B=1, S=10800, H=40, D=128`), exact-current
paired CUDA-Graph timing improves predictor latency by **36.37%**, equivalent to
**1.57x throughput**. Across the four measured shapes, latency is reduced by
22.26%-45.96%. The optimized predictor is also 28.74%-42.14% faster than the
original FlashInfer experimental workflow used as the implementation reference.

## Implementation

The original TensorRT-LLM predictor already had the correct three-stage
lifecycle, but its summary and selector kernels used scalar global loads and
CUDA-core dot products:

1. scalar K/V block summary;
2. diagonal K-centroid statistics;
3. scalar exact-block selection.

The optimized implementation keeps the existing host lifecycle, one composite
TVM-FFI custom op, plan cache, stable output buffers, and statistics kernel, but
changes the two expensive stages:

- K/V summary uses TMA to stage each K64/V64 tile in shared memory;
- exact selection uses TMA plus Blackwell `tcgen05` matrix multiply for each
  Q64-by-KC32 score tile;
- the existing statistics kernel is retained because it is materially faster
  than the experimental implementation, especially at 512 K blocks.

The TMA summary and selector schedules are adapted from the FlashInfer
`experimental/sol_attention` implementation; the selector follows its revision
7 schedule. The port intentionally reuses the vendored PrimTS tensor-map helper
and private MLA descriptor/TMEM helpers rather than duplicating those low-level
Blackwell primitives.

Relevant paths:

- `tensorrt_llm/_torch/visual_gen/attention_backend/sparse/sol/_kv_summary.py`
- `tensorrt_llm/_torch/visual_gen/attention_backend/sparse/sol/_exact_selector.py`
- `tensorrt_llm/_torch/visual_gen/attention_backend/sparse/sol/kernels.py`
- `tensorrt_llm/_torch/visual_gen/attention_backend/sparse/sol/predictor.py`

The predictor contract remains BF16 compact BSHD self-MHA with D128 and
Q64/KV64 on SM100/SM103. TMA additionally requires 16-byte-aligned Q/K/V base
addresses; the public support check now fails early with a precise reason for a
misaligned compact view. All performance and GPU correctness measurements in
this report are on SM100; SM103 remains supported by the existing contract but
is not measured here.

## Exact source under test

Pre-optimization SOL commit:

```text
71b899899b7baa9bcfd13d35f768cb812149e9fe
```

The performance rerun was made after the final cleanup, using these exact
working-tree SHA256 values:

| File | SHA256 |
|---|---|
| `kernels.py` | `f19459880d5469836b6cbe6398170d82cf8745d2a4c80994317222c6ffe38c32` |
| `_kv_summary.py` | `53437ce8c40d26a1248a89d778dbd9cac3b0af6cfdff1ddf7d2a6a69a5e9d7b9` |
| `_exact_selector.py` | `fb17f1bf4e7c7ad24f80841ad7a1765f6c4877fa9f209058b772cce6302ad227` |
| `predictor.py` | `7319c5f794ea66e34a13916ae904f25b782208c88a13a8231bf9282ca88ca075` |

The `predictor.py` hash above identifies the source used for this report's
actual measurement closure. In the current backend-lifecycle refactor working
tree, `predictor.py` has SHA256
`1b258f1bf516de14af7c55961a4b5707029ec11ee19ade4618d0da88cbf6e976`
after class/export naming integration. Performance was not rerun after that
source and host-lifecycle refactor, and this report makes no performance-
equivalence claim for the refactored working tree.

The experimental comparator lives in an untracked source directory, so its Git
HEAD does not identify the code under test. Its exact file hashes are:

| Experimental file | SHA256 |
|---|---|
| `kernels/exact_selector.py` | `5d4b3954b0d7bff589a1917d2810889837ed1727419bfab2d1af78aab6edbde1` |
| `kernels/kc_stats.py` | `957656abe9400618ab7e258536c462810ee443de19966632b655546d383d905f` |
| `kernels/kv_summary.py` | `88f259e47fbed0bf36a5f31dd8e409273d0a21c410346d103a8141708f264770` |
| `kernels/selector_config.py` | `b7de2277118f7a0ec67a0f53f5c58d9f6ce71bc2617460ee4d23cef610a11ae9` |
| `workflow.py` | `8b2378adf1687dfa3fa1c4e1eceb84f1758d8b53de5660450926b4e94ad1ffce` |

## Benchmark environment and protocol

- GPU: NVIDIA B200 SM100, physical GPU 6,
  `GPU-fc7cf4d2-52e3-2cd0-c385-787ae4d46c93`
- PyTorch: `2.12.0a0+5aff3928d8.nv26.05`
- CUDA reported by PyTorch: 13.2
- Inputs: deterministic BF16 BSHD Gaussian tensors, B1/D128
- Runtime scalars: `tau=1.0`, `sm_scale=1/sqrt(128)`
- Both implementations were captured in one process over identical Q/K/V
- 100 alternating warmups, 32 timing rounds, 64 graph replays per round
- AB/BA order alternated each round
- Primary estimator: geometric mean of paired latency ratios with a
  20,000-resample bootstrap 95% confidence interval

The paired ratio is the primary result because this shared B200 sometimes
changes performance mode during a run. Both sides of a pair move together, so
an isolated median can be misleading. The reported bootstrap intervals are
descriptive: timing samples have mode persistence and are not strictly IID.
AB-versus-BA ratios differ by less than one percentage point, and every raw
exact-current paired ratio remains below 0.87, so the direction is robust.

## Predictor latency

### Versus the original scalar TensorRT-LLM predictor

| Shape | Optimized median | Scalar median | Paired optimized/scalar (95% CI) | Latency reduction | Throughput speedup |
|---|---:|---:|---:|---:|---:|
| S4096/H8 | 20.372 us | 32.812 us | 0.62051 (0.61974-0.62125) | 37.95% | 1.61x |
| S16384/H10 | 85.927 us | 161.416 us | 0.54037 (0.53430-0.54807) | 45.96% | 1.85x |
| S32768/H10 | 239.423 us | 315.976 us | 0.77738 (0.76577-0.78996) | 22.26% | 1.29x |
| S10800/H40 | 156.958 us | 247.944 us | 0.63628 (0.62725-0.64711) | **36.37%** | **1.57x** |

### Versus the original FlashInfer experimental predictor

| Shape | Optimized median | Experimental median | Paired optimized/experimental (95% CI) | Latency reduction | Throughput speedup |
|---|---:|---:|---:|---:|---:|
| S4096/H8 | 20.427 us | 32.764 us | 0.62316 (0.62256-0.62380) | 37.68% | 1.60x |
| S16384/H10 | 85.864 us | 131.192 us | 0.65564 (0.65443-0.65785) | 34.44% | 1.53x |
| S32768/H10 | 239.407 us | 426.584 us | 0.57861 (0.56596-0.59180) | 42.14% | 1.73x |
| S10800/H40 | 156.847 us | 219.432 us | 0.71260 (0.70322-0.72055) | **28.74%** | **1.40x** |

The experimental predictor has the same optimized summary/selector topology,
but its statistics implementation is much slower. The final hybrid therefore
outperforms both comparators at every tested shape.

## Kernel-stage breakdown

NSYS reports 200 uncaptured invocations after 50 warmups. Each total below is
the sum of independently reported kernel medians.

| Shape | Implementation | Summary | Statistics | Selector | Stage sum |
|---|---|---:|---:|---:|---:|
| S16384/H10 | optimized | 17.888 us | 13.184 us | 53.409 us | 84.481 us |
| S16384/H10 | scalar | 64.416 us | 12.384 us | 82.784 us | 159.584 us |
| S32768/H10 | optimized | 30.272 us | 39.648 us | 167.649 us | 237.569 us |
| S32768/H10 | scalar | 95.008 us | 37.920 us | 177.825 us | 310.753 us |
| S10800/H40 | optimized | 39.296 us | 15.104 us | 100.512 us | **154.912 us** |
| S10800/H40 | scalar | 103.057 us | 14.976 us | 127.617 us | 245.649 us |
| S10800/H40 | experimental | 39.328 us | 75.552 us | 101.713 us | 216.593 us |

At the target shape, TMA reduces summary time by 61.9%, while the tensor-core
selector reduces selector time by 21.2%. Keeping the existing statistics
kernel avoids the experimental path's 5x statistics overhead.

## Correctness and lifecycle validation

For all four benchmark shapes, optimized versus both comparators produced:

- zero differing valid route bits;
- bitwise-equal K summaries and V summaries;
- bitwise-equal K means and diagonal variances.

The focused B200 regression suite also covers:

- S257 tail handling with B/H greater than one;
- the 258-block/9-word proxy-group boundary and invalid tail-bit clearing;
- dynamic `tau` and `sm_scale`;
- plan ownership and stable output/scratch addresses;
- independent compiled predictor instances;
- CUDA-Graph capture and replay with mutated live Q/K/V;
- custom-op mutation schema and Meta registration;
- integration through `SOLTrtllmAttention` and the generic proxy FMHA;
- contiguous BSR, proxy-bitmask, and paged generic block-sparse paths.

Fresh command:

```bash
CUDA_VISIBLE_DEVICES=6 PYTHONDONTWRITEBYTECODE=1 PYTHONPATH="$PWD" \
  python3 -m pytest -p no:cacheprovider -q \
  tests/unittest/_torch/visual_gen/sparse_attention/test_sol_predictor.py \
  tests/unittest/_torch/visual_gen/sparse_attention/test_sol_attention.py \
  tests/unittest/_torch/attention/test_prims_ts_block_sparse.py
```

Result: `44 passed`.

## Real Wan 14B validation

The optimized predictor also completed the checkpoint-loaded 40-layer Wan 14B
transformer replay and the prompt-derived 720p/9-frame/4-step pipeline with
CUDA Graph enabled. Both runs were finite and deterministic across replay.

The target-shape predictor saves about 91 us per layer, or roughly 3.6 ms over
40 self-attention layers. Across four denoising steps that projects to about
14.5 ms. This is less than 1% of the full transformer/pipeline latency and is
smaller than the 30-60 ms performance-mode variation observed on the shared
GPU. Consequently, the current full-model medians are validation evidence, not
a statistically defensible end-to-end speedup claim.

Compared with the preceding scalar-predictor transformer run:

- output cosine: 0.9998872;
- output RMSE: 0.004735 BF16 units;
- mean exact-route density: 18.336978% optimized versus 18.336851% scalar.

The different BF16/tensor-core accumulation order can move
threshold-adjacent routes. Over four denoising steps those small changes can
amplify in the decoded video; the observed old-versus-new LPIPS Alex mean was
0.0955. This comparison is not a dense-quality score. Production quality still
requires the separate SOL `tau`/dense-prefix calibration described in the
broader SOL/VSA report.

## Scope boundary

This optimization does not modify the generic PrimTS block-sparse attention
kernel. At S10800/H40, the predictor improvement alone would reduce a prior
approximately 1.016 ms `predict + attention` invocation by roughly 9%, assuming
the attention core remains at its measured approximately 0.79 ms. Attention
kernel optimization is an independent workstream and is intentionally excluded
from the predictor speedup claim.

## Reproduction

Run from the target TensorRT-LLM worktree:

```bash
export CUDA_VISIBLE_DEVICES=6
export TRTLLM_REPO="$PWD"
export SOL_EXPERIMENT_REPO=/home/scratch.yuhangh_gpu_1/workspace/AIGV/flashinfer-primts-proxy-live-rebase
export SOL_HISTORICAL_REPO=/home/scratch.yuhangh_gpu_1/workspace/AIGV/flashinfer-primts-sol-attn
export SOL_PROFILE_ARTIFACT_ROOT=/tmp/sol_predictor_profile_20260902/final_source_rerun
export SOL_SCALAR_BASELINE_KERNELS=/tmp/sol_predictor_profile_20260902/scalar_baseline_tree/tensorrt_llm/_torch/visual_gen/attention_backend/sparse/sol/kernels.py
export PYTHONPATH="$PWD:$SOL_EXPERIMENT_REPO"

python3 /tmp/sol_predictor_profile_20260902/profile_predictors.py \
  --mode paired --candidate scalar_baseline \
  --shape 4096x8 --shape 16384x10 --shape 32768x10 --shape 10800x40 \
  --warmup 100 --rounds 32 --iters 64
```

Replace `scalar_baseline` with `experimental` for the second comparison.

## Artifacts

- Exact-current paired JSON:
  `/tmp/sol_predictor_profile_20260902/final_source_rerun/`
- Exact-current S4096 paired JSON:
  `/tmp/sol_predictor_profile_20260902/final_source_s4096/`
- NSYS/NCU traces and raw paired samples:
  `/tmp/sol_predictor_profile_20260902/`
- Optimized Wan transformer:
  `/tmp/sol_predictor_optimized_wan14b/`
- Optimized prompt-derived pipeline:
  `/tmp/sol_predictor_optimized_wan14b_pipeline/`
- Broader SOL/VSA/PR #18329 comparison:
  `benchmark-results/sol-vsa-pr18329-b200-2026-09-02.md`
