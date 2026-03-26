# Paper Alignment Check: LeWorldModel (LeWM) JEPA

**Paper**: LeWorldModel: Stable End-to-End Joint-Embedding Predictive Architecture from Pixels (arXiv 2603.19312v1)
**Code directory**: `output/.../code/`
**Reviewed**: 2026-03-26

---

## 1. Algorithm Fidelity

### Training Loop (Algorithm 1)
- [x] Teacher-forcing: encoder embeddings used as predictor inputs (not autoregressive predictions during training) - confirmed in `lewm/models/lewm.py:forward` and `lewm/training/trainer.py:train_epoch`
- [x] Prediction loss: `F.mse_loss(z_hat[:, :-1], z[:, 1:])` correctly aligns predictor output at t with encoder target at t+1 - confirmed in `lewm/losses/prediction_loss.py`
- [x] SIGReg applied step-wise per timestep, then averaged - confirmed in `lewm/losses/sigreg.py:compute_sigreg_stepwise`
- [x] Total loss = MSE + 0.1 * SIGReg - confirmed in `lewm/training/trainer.py:68-72`
- [x] Joint optimization of encoder + predictor (no EMA, no stop-gradient) - confirmed by absence of `.detach()` calls on `z` before passing to predictor
- [x] AdamW optimizer - confirmed in `lewm/training/trainer.py:38-42`

### Planning (CEM, Section 3.4)
- [x] 300 samples, 30 iterations, 30 elites - confirmed in `lewm/config.py:CEMConfig` defaults
- [x] Horizon H=5 - confirmed in `lewm/config.py:CEMConfig.horizon`
- [x] Cost = L2 distance to goal embedding - confirmed in `lewm/planning/cem.py:93`
- [x] Autoregressive rollout in latent space - confirmed in `lewm/planning/rollout.py`

---

## 2. Equation Implementation

### Eq. 1: Prediction Loss
| Aspect | Paper | Code | Match? |
|--------|-------|------|--------|
| Formula | `||z_hat_{t+1} - z_{t+1}||^2` | `F.mse_loss(z_hat[:, :-1], z[:, 1:])` | YES |
| Teacher-forcing | ground-truth z as input | yes, `z` from encoder passed to predictor | YES |

### Eq. 2: SIGReg
| Aspect | Paper | Code | Match? |
|--------|-------|------|--------|
| M projections | 1024 | `num_projections=1024` in `SIGRegConfig` | YES |
| Unit-norm directions | `u ~ S^{D-1}` | `U / U.norm(dim=0)` in `sigreg.py:52` | YES |
| Projection | `h = Z @ u` | `H = Z @ U` in `sigreg.py:72` | YES |
| Average over M | `(1/M) sum T^(m)` | `result.mean()` in `sigreg.py:105` | YES |

### EP: Epps-Pulley Test Statistic
| Aspect | Paper | Code | Match? |
|--------|-------|------|--------|
| ECF formula | `|phi_N - phi_0|^2` expanded real form | vectorized real expansion via cos/sin | YES |
| Term 1 | `(1/N^2) sum cos(t(h_j-h_k))` | `mean_cos^2 + mean_sin^2` (equivalent via `|ECF|^2 = Re^2 + Im^2`) | YES |
| Term 2 | `(2/N) sum cos(t*h_n) * exp(-t^2/2)` | `2 * mean_cos * exp(-t^2/2)` | YES |
| Term 3 | `exp(-t^2)` | `torch.exp(-t**2)` | YES |
| Weight `w(t)` | `exp(-t^2/2)` | `exp(-t^2/2)` in `sigreg.py:40` | YES |
| Integration range | `[0.2, 4]` via trapezoid | `t_min=0.2, t_max=4.0` with `torch.trapezoid` | YES |
| Knots | ~17 (paper ablates 4-32) | `num_knots=17` default | YES |

**Note**: The vectorized implementation `|phi_N(t)|^2 = mean_cos^2 + mean_sin^2` is mathematically equivalent to `(1/N^2) sum_{j,k} cos(t(h_j - h_k))` and avoids the O(N^2) pairwise computation. This is a correct and more efficient implementation.

### Eq. 3: Total Loss
| Aspect | Paper | Code | Match? |
|--------|-------|------|--------|
| Formula | `L_pred + 0.1 * SIGReg` | `l_pred + self.lambda_sigreg * l_sigreg` | YES |
| Default lambda | 0.1 | `loss_weight=0.1` in `SIGRegConfig` | YES |

---

## 3. Hyperparameter Defaults

| Parameter | Paper Value | Code Default | Match? |
|-----------|-------------|--------------|--------|
| Encoder: patch size | **14** | **16** (`vit_tiny_patch16_224`) | **NO** |
| Encoder: layers | 12 | 12 (`num_layers=12`) | YES |
| Encoder: heads | 3 | 3 (`num_heads=3`) | YES |
| Encoder: hidden dim | 192 | 192 (`embed_dim=192`) | YES |
| Predictor: layers | 6 | 6 (`num_layers=6`) | YES |
| Predictor: heads | 16 | 16 (`num_heads=16`) | YES |
| Predictor: dropout | 0.1 | 0.1 (`dropout=0.1`) | YES |
| Batch size | 128 | 128 | YES |
| Sub-traj length | 4 | 4 | YES |
| Frame skip | 5 | 5 | YES |
| SIGReg M | 1024 | 1024 | YES |
| SIGReg lambda | 0.1 | 0.1 | YES |
| CEM samples | 300 | 300 | YES |
| CEM elites | 30 | 30 | YES |
| CEM iterations | 30 | 30 | YES |
| CEM horizon | 5 | 5 | YES |

---

## 4. Architecture Match

### Encoder
- [x] ViT-Tiny backbone loaded via `timm.create_model` with `num_classes=0` (returns CLS token)
- [x] `[CLS]` token output projected via projection head
- [x] Projection head uses `Linear + BatchNorm1d` (NOT LayerNorm) - confirmed in `projection_head.py`
- [ ] **DEVIATION**: Model name is `vit_tiny_patch16_224` (patch size 16), but the paper specifies **patch size 14**. The correct timm model for patch=14 would be `vit_tiny_patch14_224` or `vit_tiny_patch14_reg4_dinov2`. The docstring on `encoder.py:12` even incorrectly says "patch=16" despite the paper requiring 14. This affects the number of tokens processed, the sequence length, and parameter count.

### Predictor
- [x] 6 transformer layers
- [x] 16 attention heads
- [x] 0.1 dropout
- [x] Causal attention mask (upper triangular filled with `-inf`) - confirmed in `predictor.py:121-125`
- [x] Learned positional embeddings - confirmed in `predictor.py:93-95`
- [x] Projection head (Linear + BatchNorm1d) on output
- [x] AdaLN at each layer for action conditioning
- [ ] **DEVIATION**: Action conditioning passes a single mean-pooled action embedding `action_cond = action_emb.mean(dim=1)` to all layers. The paper specifies that the action `a_t` at each timestep conditions the corresponding layer. Using the temporal mean obscures which action corresponds to which prediction step. Per-step action conditioning would inject `action_emb[:, t, :]` at each position `t` rather than the mean across all timesteps.

### AdaLN
- [x] Zero initialization of `cond_proj` weight and bias - confirmed in `adaln.py:30-31`
- [x] `scale * LayerNorm(x) + shift` formula - confirmed in `adaln.py:48`
- [x] `elementwise_affine=False` on the inner LayerNorm - confirmed in `adaln.py:26`

---

## 5. Training Pipeline

- [x] Optimizer: AdamW (paper does not specify; AdamW is a well-justified default)
- [x] Learning rate: 3e-4 (paper does not specify; reasonable default)
- [x] Loss: MSE + 0.1 * SIGReg
- [x] No EMA
- [x] No stop-gradient on encoder embeddings before SIGReg or before predictor
- [ ] **DEVIATION**: The SIGReg weighting function uses `w(t) = exp(-t^2/2)` with an implicit internal lambda=1. The paper equations distinguish the internal SIGReg lambda (weighting function parameter) from the loss weight lambda=0.1. The code uses the correct weighting function formula but does not expose the internal lambda as a configurable parameter. This is a minor structural issue, not a correctness issue for the default case.

---

## 6. Planning Pipeline

- [x] CEM with correct hyperparameters
- [x] Goal embedding from encoder
- [x] Autoregressive rollout in latent space
- [x] Cost = squared L2 distance to goal embedding at horizon H
- [x] Elite selection and distribution update

---

## 7. Completeness

- [x] Encoder
- [x] Predictor
- [x] SIGReg
- [x] Training loop
- [x] CEM planner
- [x] Autoregressive rollout
- [x] Temporal straightness diagnostic (Eq. 9)
- [x] Optional decoder (visualization only, not in training)
- [x] `main.py` entry point
- [x] `requirements.txt`
- [x] Unit tests for all major components

---

## Alignment Score: 7 / 10

**Rationale**:

The implementation is structurally complete and mathematically correct for all core equations. All SIGReg math (Epps-Pulley, trapezoid integration), loss terms, and training loop semantics are faithfully reproduced. The critical AdaLN zero-init, BatchNorm1d in projection heads, teacher-forcing, and no-EMA/no-stop-grad requirements are all correctly implemented.

Two deviations lower the score from a 9:

1. **Patch size 14 vs 16 (major)**: The paper explicitly specifies patch size 14 (`vit_tiny_patch14`). The code uses `vit_tiny_patch16_224`. This changes the spatial resolution of the ViT backbone and is inconsistent with the paper's architecture.

2. **Action conditioning mean-pooling (minor)**: Using `action_emb.mean(dim=1)` as a single conditioning signal for all sequence positions is a simplification that blurs per-timestep action information. The correct approach injects the action for step `t` at position `t`.
