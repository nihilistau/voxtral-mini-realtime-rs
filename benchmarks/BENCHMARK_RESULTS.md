# RTF Benchmark Results — Voxtral Mini Q4 GGUF

**Hardware:** Intel NUC 11 Extreme (Beast Canyon)
- **Discrete GPU:** NVIDIA GeForce RTX 2060 (6 GB VRAM)
- **Integrated GPU:** Intel UHD Graphics (shared system memory)
- **CPU:** Intel Core i9-11900KB
- **OS:** Windows, Vulkan backend

**Model:** Voxtral Mini 4B Q4_0 GGUF (~2.5 GB)

**Date:** 2026-05-06

---

## Summary

| Mode | 3.4s Audio | 34s Audio | 120s Audio | Notes |
|------|-----------|-----------|------------|-------|
| **Discrete (RTX)** | 1.91 RTF | 0.63 RTF | **0.55 RTF** | Fastest. Real-time at ≥30s audio |
| Discrete + SP | 1.42 RTF | 0.97 RTF | 1.39 RTF | SP overhead hurts when VRAM is available |
| Integrated + SP | 14.80 RTF | — | — | Too slow for practical use |
| Hybrid (RTX↔iGPU) | 7.35 RTF | 3.27 RTF | 3.67 RTF | iGPU decode is 7-9x bottleneck |
| Hybrid + Pipeline | 7.33 RTF | 3.27 RTF | 3.68 RTF | Pipeline overlap negligible vs decode cost |

**Key finding:** RTX discrete-only is the clear winner (0.55 RTF = 1.8x real-time at 2 min audio). Shannon-Prime adds overhead when VRAM isn't constrained. Hybrid mode's value is freeing RTX VRAM for other workloads, not throughput.

---

## Detailed Results

### Short Audio (3.4s — "Mary had a little lamb")

| Mode | Pre (ms) | Enc (ms) | Xfer (ms) | Dec (ms) | Total (ms) | RTF | Tok/s |
|------|---------|---------|----------|---------|-----------|-----|-------|
| discrete | 372 | 2,099 | 0 | 4,111 | 6,582 | 1.91 | 7.3 |
| discrete+SP | 24 | 1,435 | 0 | 3,423 | 4,882 | 1.42 | 8.8 |
| integrated+SP | 71 | 26,756 | 0 | 24,101 | 50,928 | 14.80 | 1.2 |
| hybrid | 23 | 1,523 | 339 | 23,390 | 25,275 | 7.35 | 1.3 |
| hybrid+pipe | 23 | 1,181 | 0 | 23,638 | 25,200 | 7.33 | 1.3 |

### Medium Audio (34.4s — 10x concatenation)

| Mode | Pre (ms) | Enc (ms) | Xfer (ms) | Dec (ms) | Total (ms) | RTF | Tok/s |
|------|---------|---------|----------|---------|-----------|-----|-------|
| discrete | 420 | 6,121 | 0 | 15,053 | 21,593 | 0.63 | 14.8 |
| discrete+SP | 403 | 5,954 | 0 | 27,014 | 33,371 | 0.97 | 8.3 |
| hybrid | 423 | 5,280 | 357 | 106,444 | 112,503 | 3.27 | 2.1 |
| hybrid+pipe | 391 | 5,356 | 319 | 106,377 | 112,443 | 3.27 | 2.1 |

### Long Audio (120.4s — 35x concatenation)

| Mode | Pre (ms) | Enc (ms) | Xfer (ms) | Dec (ms) | Total (ms) | RTF | Tok/s |
|------|---------|---------|----------|---------|-----------|-----|-------|
| discrete | 326 | 19,137 | 0 | 47,079 | 66,542 | 0.55 | 16.2 |
| discrete+SP | 344 | 18,728 | 0 | 147,753 | 166,825 | 1.39 | 5.2 |
| hybrid | 326 | 17,778 | 297 | 423,914 | 442,315 | 3.67 | 1.8 |
| hybrid+pipe | 325 | 17,770 | 324 | 424,309 | 442,727 | 3.68 | 1.8 |

---

## Analysis

### Scaling with Audio Length

Discrete mode improves dramatically with longer audio — from 1.91 RTF (3.4s) to 0.55 RTF (120s). This is because the fixed model-load and warmup costs amortize over more audio. At 120s, the RTX 2060 transcribes at 1.8x real-time speed.

### Shannon-Prime Overhead

On discrete GPU, Shannon-Prime VHT2 compression adds significant decode overhead (3.1x slower at 120s). The VHT2 compress/decompress cycles on every KV cache access dominate when VRAM isn't constrained. SP's value is enabling inference on memory-constrained devices (iGPU), not throughput optimization.

### Hybrid Decode Bottleneck

The iGPU decode is 7-9x slower than RTX decode. This completely dominates the total time, making the encode phase (which runs at RTX speed) irrelevant to the overall RTF.

### Pipeline Overlap

Pipelined hybrid shows virtually no improvement over non-pipelined hybrid. The reason: encode time (~18s for 120s audio) is dwarfed by decode time (~424s). Even if you perfectly overlap all encode work with decode work, you save at most 18s out of 442s total — a 4% improvement, within measurement noise.

### When to Use Each Mode

- **Discrete:** Best throughput. Use when RTX has available VRAM (~2.5 GB)
- **Hybrid:** When RTX VRAM is needed for other workloads (rendering, other models). Frees 2.5 GB RTX VRAM at cost of 6.6x slower inference
- **Integrated-only:** Only when no discrete GPU is available. Too slow for real-time use
- **Shannon-Prime:** Only beneficial on memory-constrained devices. Do not enable on discrete GPU
