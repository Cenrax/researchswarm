# Overview: Research Papers for "Build the JEPA Model from Scratch in Python"

**Objective**: Build the JEPA (Joint Embedding Predictive Architecture) model from scratch in Python using PyTorch.
**Date**: 2026-03-26

---

## Papers Processed

| Rank | Paper | File | Relevance | Core Contribution |
|------|-------|------|-----------|-------------------|
| 1 | LeWorldModel: Stable End-to-End JEPA from Pixels | `jepa.pdf` | HIGH | First end-to-end JEPA with provable anti-collapse (SIGReg) — complete architecture + training recipe |

---

## Paper 1 — LeWorldModel (jepa.pdf)

**Relevance: HIGH**

This is the primary and only paper processed. It is the direct implementation reference for building a JEPA from scratch in Python. The paper provides:

- Complete architecture specification (encoder: ViT-Tiny 12L/3H/d=192; predictor: ViT-S 6L/16H/d=192 with AdaLN action conditioning)
- Full pseudocode for the training loop (Algorithm 1)
- Exact loss function with two terms: MSE prediction + SIGReg regularizer
- Mathematical derivation of the anti-collapse mechanism (Epps-Pulley test, Cramér-Wold theorem)
- All critical hyperparameters: batch_size=128, sub-trajectory T=4, frame_skip=5, image 224x224, lambda=0.1, M=1024 projections, 10 epochs
- CEM planning algorithm (Algorithm 2) for inference
- Ablation studies covering dropout (0.1 optimal), embedding dim (192 minimum), predictor size (ViT-S optimal), lambda range ([0.01, 0.2])

**Key design choices** essential for a from-scratch implementation:
1. BatchNorm (not LayerNorm) in the projection head — enables SIGReg to work
2. AdaLN initialized to zero — stabilizes early training
3. Teacher-forcing (not autoregressive) during training
4. SIGReg applied step-wise per timestep, then averaged
5. No EMA, no stop-gradient, no pretrained weights required

**Distinction from I-JEPA** (the original image JEPA by Assran et al., 2023): The paper being analyzed is LeWM, a world-model variant of JEPA for sequential decision-making with actions, not the masked image patch prediction I-JEPA. LeWM replaces I-JEPA's EMA/stop-gradient with SIGReg, making it end-to-end trainable.

---

## Implementation Roadmap (Priority Order)

Based on the single paper analyzed, the recommended build order for a from-scratch Python/PyTorch implementation is:

### Phase 1: Core Components
1. **SIGReg regularizer** — implement Epps-Pulley statistic with trapezoid quadrature; vectorize over M projections
2. **ViT-Tiny Encoder** — use `timm` (`vit_tiny_patch14_224`) or HuggingFace; add `Linear + BatchNorm1d` projection head on `[CLS]` token
3. **Causal Transformer Predictor** — 6-layer ViT-S with causal attention mask; AdaLN (zero-init) for action injection; same projection head as encoder

### Phase 2: Training Infrastructure
4. **Dataset + DataLoader** — offline trajectories of (obs, action) pairs; sub-trajectories of length 4; frame skip 5; 224x224 images
5. **Training loop** — teacher-forcing MSE + step-wise SIGReg; joint backprop; single optimizer (AdamW recommended)

### Phase 3: Planning (Inference)
6. **CEM solver** — 300 samples, 30 iterations, top-30 elites; initialize mu=0, sigma=1
7. **Autoregressive rollout** — encode initial obs; predict forward H=5 steps; compute L2 cost to goal embedding

### Phase 4: Optional Diagnostics
8. **Decoder** (visualization only) — cross-attention transformer decoder from 192-dim CLS token to 224x224 image
9. **Temporal straightness metric** — cosine similarity between consecutive latent velocities

---

## Key Numbers Reference Card

| Quantity | Value |
|----------|-------|
| Latent dimension ($d$) | 192 |
| Encoder: ViT-Tiny layers | 12 |
| Encoder: attention heads | 3 |
| Encoder: patch size | 14 |
| Encoder: ~params | 5M |
| Predictor: layers | 6 |
| Predictor: attention heads | 16 |
| Predictor: dropout | 0.1 |
| Predictor: ~params | 10M |
| Total model params | ~15M |
| SIGReg projections M | 1024 |
| SIGReg lambda (loss weight) | 0.1 |
| SIGReg t-range | [0.2, 4] |
| SIGReg quadrature knots | 17 (insensitive) |
| Batch size | 128 |
| Sub-trajectory length | 4 frames |
| Frame skip | 5 |
| Image resolution | 224 x 224 |
| History length N (predictor) | 3 (PushT, OGBench); 1 (TwoRoom) |
| Training epochs | 10 |
| CEM samples per iter | 300 |
| CEM iterations | 30 (PushT); 10 (others) |
| CEM elite count | 30 |
| Planning horizon H | 5 steps (= 25 env steps) |

---

## Files Produced

| File | Description |
|------|-------------|
| `jepa_summary.md` | Full structured implementation-focused summary with architecture details, equations, pseudocode, and ablation data |
| `jepa_equations.md` | All equations extracted and catalogued by category with implementation notes |
| `_overview.md` | This file — paper ranking and implementation roadmap |
