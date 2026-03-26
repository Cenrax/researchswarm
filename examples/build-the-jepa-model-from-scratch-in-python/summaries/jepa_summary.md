# LeWorldModel: Stable End-to-End Joint-Embedding Predictive Architecture from Pixels

**Source**: `/Users/subham/Desktop/codes/agentsclaude/input/week_15/jepa.pdf`
**Authors**: Lucas Maes, Quentin Le Lidec, Damien Scieur, Yann LeCun, Randall Balestriero
**Affiliations**: Mila & Université de Montréal, New York University, Samsung SAIL, Brown University
**arXiv**: 2603.19312v1 (13 Mar 2026)
**Date Read**: 2026-03-26
**Relevance**: HIGH — this paper is the primary reference for building a JEPA (Joint Embedding Predictive Architecture) world model from scratch in Python/PyTorch.

---

## 1. Core Idea

LeWorldModel (LeWM) is the first JEPA that trains **stably end-to-end from raw pixels** using only **two loss terms**: a next-embedding prediction loss (MSE) and a SIGReg regularizer that enforces an isotropic Gaussian distribution over latent embeddings. This eliminates the need for exponential moving averages (EMA), stop-gradient, pretrained encoders, or auxiliary supervision to avoid representation collapse.

The model trains an encoder (ViT-Tiny, ~5M params) and an action-conditioned predictor (ViT-S transformer, ~10M params) jointly. Total model size: ~15M parameters, trainable on a single GPU.

---

## 2. Relevance to Objective

This paper directly defines the JEPA variant you need to implement. Every component is described with sufficient architectural and hyperparameter specificity to reproduce from scratch:

- The encoder architecture (ViT-Tiny with CLS-token projection)
- The predictor architecture (causal transformer with AdaLN action conditioning)
- The exact loss function (MSE + SIGReg)
- The SIGReg anti-collapse regularizer (Epps-Pulley test on random projections)
- The training loop (teacher-forcing, joint optimization, no EMA/stop-grad)
- The planning procedure (CEM in latent space)

---

## 3. Key Techniques

- **Joint Embedding Predictive Architecture (JEPA)**: Encodes observations into a compact latent space and predicts future latent states conditioned on actions. No pixel reconstruction.
- **ViT-Tiny Encoder**: Vision Transformer (patch size 14, 12 layers, 3 heads, hidden dim 192). The `[CLS]` token output is projected via a 1-layer MLP with BatchNorm to form the latent embedding.
- **Causal Transformer Predictor (ViT-S)**: 6-layer transformer with 16 attention heads, 10% dropout, causal masking over a history of N frames. Action conditioning via Adaptive Layer Normalization (AdaLN) initialized to zero.
- **SIGReg (Sketched-Isotropic-Gaussian Regularizer)**: Projects embeddings onto M=1024 random unit-norm directions, applies the Epps-Pulley univariate normality test to each projection, and averages the statistics. By the Cramér-Wold theorem, matching all 1D marginals implies the joint distribution converges to N(0, I).
- **Teacher-Forcing Training**: The encoder encodes all frames; the predictor takes ground-truth latent embeddings as context (not its own predictions) during training.
- **Model Predictive Control (CEM) for Planning**: At inference, latent rollouts are generated autoregressively; the Cross-Entropy Method optimizes the action sequence to minimize distance to a goal embedding.
- **No EMA, no stop-gradient**: All parameters updated jointly through backpropagation.

---

## 4. Implementation-Critical Details

### Encoder
| Parameter | Value |
|-----------|-------|
| Architecture | ViT-Tiny (HuggingFace library) |
| Patch size | 14 |
| Number of layers | 12 |
| Attention heads | 3 |
| Hidden dimension | 192 |
| Parameters | ~5M |
| Output token | `[CLS]` token from last layer |
| Projection head | 1-layer MLP with BatchNorm (NOT LayerNorm) |

**Critical note**: The final ViT layer uses LayerNorm. The projection head must use BatchNorm, not LayerNorm, because LayerNorm prevents SIGReg from working properly. The `[CLS]` embedding is projected through `Linear -> BatchNorm -> (no activation needed)` to produce the latent embedding `z` of dimension 192.

### Predictor
| Parameter | Value |
|-----------|-------|
| Architecture | ViT-S backbone |
| Layers | 6 |
| Attention heads | 16 |
| Dropout | 10% (p=0.1) |
| Parameters | ~10M |
| Action conditioning | Adaptive Layer Normalization (AdaLN) at each layer |
| AdaLN init | Zero initialization (for training stability) |
| Context/history length | N=3 for PushT and OGBench-Cube; N=1 for TwoRoom |
| Masking | Temporal causal mask (cannot attend to future embeddings) |
| Positional embeddings | Learned |
| Projection head | Same as encoder: 1-layer MLP with BatchNorm |

**AdaLN zero init is critical**: initializing AdaLN parameters to zero ensures action conditioning begins with zero influence and ramps up gradually during training.

### Dropout ablation (Push-T):
| p | Success Rate |
|---|-------------|
| 0.0 | 78% |
| **0.1** | **96%** |
| 0.2 | 85% |
| 0.5 | 67% |

### Predictor size ablation (Push-T):
| Size | Success Rate |
|------|-------------|
| ViT-Tiny | 81% |
| **ViT-Small** | **96%** |
| ViT-Base | 87% |

### Loss Function
```
L_LeWM = L_pred + lambda * SIGReg(Z)
```
- `L_pred = ||z_hat_{t+1} - z_{t+1}||^2_2` (MSE, teacher-forcing)
- `SIGReg(Z) = (1/M) * sum_{m=1}^{M} T(h^(m))` where `h^(m) = Z @ u^(m)`
- `lambda = 0.1` (default; robust in range [0.01, 0.2])
- `M = 1024` random projections

### SIGReg Implementation
1. Sample M=1024 random unit-norm directions `u^(m)` in R^d (uniformly on the hypersphere).
2. Project embeddings: `h^(m) = Z @ u^(m)`, giving a 1D vector per direction.
3. Compute the Epps-Pulley test statistic on each `h^(m)`:
   - Compute the empirical characteristic function (ECF): `phi_N(t; h) = (1/N) * sum_n exp(i * t * h_n)`
   - Compare against standard Gaussian CF: `phi_0(t) = exp(-t^2/2)`
   - Integrate: `T^(m) = integral w(t) * |phi_N(t; h^(m)) - phi_0(t)|^2 dt`
   - Weighting: `w(t) = exp(-t^2 / (2*lambda^2))` (separate lambda from loss weight; use trapezoid quadrature over [0.2, 4] with T knots)
4. Average over M projections.
5. Apply SIGReg step-wise (per timestep), not across the temporal dimension.

**Implementation note**: SIGReg is applied to `emb.transpose(0, 1)` — i.e., it processes each timestep's batch of embeddings independently, then averages over timesteps.

### Training Hyperparameters
| Parameter | Value |
|-----------|-------|
| Batch size | 128 |
| Sub-trajectory length | 4 frames |
| Frame skip | 5 (4 blocks of 5 actions) |
| Image resolution | 224 x 224 pixels |
| Epochs | 10 |
| SIGReg weight (lambda) | 0.1 |
| SIGReg projections (M) | 1024 |
| SIGReg knots (T) | ~8-17 (insensitive; paper ablates 4 to 32) |
| Optimizer | Not specified — standard AdamW assumed |
| No EMA, no stop-gradient | Confirmed |

### Embedding Dimension Ablation
Performance saturates at dim ~192 and above; drops sharply below ~184.

---

## 5. Equations

### Forward Pass (LeWM)
```
z_t = enc_theta(o_t)           # Encoder: observation -> latent
z_hat_{t+1} = pred_phi(z_t, a_t)  # Predictor: latent + action -> next latent
```

### Prediction Loss (Eq. 1)
$$
\mathcal{L}_{\text{pred}} \triangleq \|\hat{z}_{t+1} - z_{t+1}\|_2^2, \quad \hat{z}_{t+1} = \text{pred}_\phi(z_t, a_t)
$$
Plain English: MSE between predicted next embedding and ground-truth next embedding (teacher-forcing).

### SIGReg Projection (Eq. 6)
$$
h^{(m)} \triangleq Z u^{(m)}, \quad u^{(m)} \in \mathbb{S}^{D-1}
$$
Plain English: Project the batch of embeddings Z onto a random unit-norm direction to get a 1D sample.

### Epps-Pulley Test Statistic (EP)
$$
T^{(m)} = \int_{-\infty}^{\infty} w(t) \left| \phi_N(t; h^{(m)}) - \phi_0(t) \right|^2 dt
$$
where:
- $\phi_N(t; h) = \frac{1}{N} \sum_{n=1}^{N} e^{i t h_n}$ — empirical characteristic function
- $\phi_0(t) = e^{-t^2/2}$ — standard Gaussian characteristic function
- $w(t) = e^{-t^2/(2\lambda^2)}$ — Gaussian weighting function
- Integral computed via trapezoid rule over $t \in [0.2, 4]$

### SIGReg Aggregation (Eq. 2 / SIGReg)
$$
\text{SIGReg}(Z) \triangleq \frac{1}{M} \sum_{m=1}^{M} T^{(m)}
$$
Plain English: Average Epps-Pulley statistics over M random projections to encourage the full embedding distribution to match N(0, I).

### Total Loss (Eq. 3)
$$
\mathcal{L}_{\text{LeWM}} \triangleq \mathcal{L}_{\text{pred}} + \lambda \cdot \text{SIGReg}(Z)
$$
where $\lambda = 0.1$ by default.

### Convergence Guarantee (Cramer-Wold)
$$
\text{SIGReg}(Z) \to 0 \iff \mathbb{P}_Z \to \mathcal{N}(0, I)
$$

### Planning Objective (Eq. 4, 5)
$$
\mathcal{C}(\hat{z}_H) = \|\hat{z}_H - z_g\|_2^2, \quad z_g = \text{enc}_\theta(o_g)
$$
$$
a^*_{1:H} = \arg\min_{a_{1:H}} \mathcal{C}(\hat{z}_H)
$$
solved via CEM (Cross-Entropy Method).

### Temporal Straightness Metric (Eq. 9, diagnostic only)
$$
\mathcal{S}_{\text{straight}} = \frac{1}{B(T-2)} \sum_{i=1}^{B} \sum_{t=1}^{T-2} \frac{\langle v_t^{(i)}, v_{t+1}^{(i)} \rangle}{\|v_t^{(i)}\| \|v_{t+1}^{(i)}\|}
$$
where $v_t = z_{t+1} - z_t$.

---

## 6. Architecture Diagram (ASCII)

```
Training:
                                              SIGReg
                                                |
 o_t ---[Encoder (ViT-Tiny)]---> z_t --------> +
 o_{t+1} -[Encoder (ViT-Tiny)]-> z_{t+1} ----> +
                                  |
                                MSE <----- z_hat_{t+1}
                                              ^
                              [Predictor (ViT-S, 6L, 16H)]
                              /          \
                           z_t          a_t (via AdaLN)

 Encoder details:
   ViT-Tiny (patch=14, L=12, H=3, d=192)
     -> [CLS] token
     -> MLP(Linear + BatchNorm)
     -> z_t in R^192

 Predictor details:
   History: [z_{t-N+1}, ..., z_t]  (N=3 frames)
   Action:  a_t injected via AdaLN at each of 6 layers
   Causal mask: prevents future token attention
   -> z_hat_{t+1} in R^192 (after projection head)

Planning (inference):
  o_1 --[Encoder]--> z_1
  o_g --[Encoder]--> z_g
  CEM optimizes a_{1:H} to minimize ||z_hat_H - z_g||^2
    rollout: z_hat_{t+1} = pred(z_hat_t, a_t)  [autoregressive]
    CEM: 300 samples, 30 iterations (PushT) / 10 iterations (others)
    elites: top 30
    horizon H=5 steps (= 25 env steps with frame skip 5)
```

---

## 7. Training Loop Pseudocode

```python
# Based directly on Algorithm 1 in the paper

def train_lewm(dataloader, encoder, predictor, optimizer, lambd=0.1, M=1024):
    for batch in dataloader:
        obs    = batch['obs']      # (B, T, C, H, W)  e.g. T=4, H=W=224
        actions = batch['actions'] # (B, T, A)

        # Forward pass
        emb = encoder(obs)           # (B, T, D)  D=192
        next_emb = predictor(emb, actions)  # (B, T, D)

        # 1) Prediction loss (teacher-forcing MSE)
        pred_loss = F.mse_loss(emb[:, 1:], next_emb[:, :-1])

        # 2) SIGReg anti-collapse (step-wise, per timestep)
        # emb.transpose(0,1) -> (T, B, D)
        sigreg_loss = mean_over_timesteps(SIGReg(emb.transpose(0, 1), M=M))

        loss = pred_loss + lambd * sigreg_loss

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()


def SIGReg(Z, M=1024):
    """
    Z: (N, B, D) - N=history, B=batch, D=embedding dim
    Returns scalar SIGReg loss.
    """
    # Flatten to (N*B, D)
    Z_flat = Z.reshape(-1, Z.shape[-1])
    D = Z_flat.shape[-1]

    # Sample M random unit-norm directions
    U = torch.randn(D, M)
    U = U / U.norm(dim=0, keepdim=True)  # (D, M)

    # Project embeddings: (N*B, M)
    H = Z_flat @ U

    # Compute Epps-Pulley statistic for each projection
    total = 0.0
    for m in range(M):
        h = H[:, m]  # (N*B,)
        total += epps_pulley_statistic(h)

    return total / M


def epps_pulley_statistic(h, n_knots=17, t_min=0.2, t_max=4.0):
    """
    Epps-Pulley test statistic for normality.
    h: 1D tensor of projected embeddings
    Returns scalar test statistic T.
    """
    N = h.shape[0]
    t_vals = torch.linspace(t_min, t_max, n_knots)  # quadrature points
    w = torch.exp(-t_vals**2 / 2)  # Gaussian weight w(t) = exp(-t^2/2)

    # Empirical CF: phi_N(t) = (1/N) * sum_n exp(i*t*h_n)
    # |phi_N(t) - phi_0(t)|^2 = real part expansion:
    # = (1/N^2)*sum_{n,m} cos(t*(h_n-h_m)) - (2/N)*sum_n cos(t*h_n)*exp(-t^2/2) + exp(-t^2)

    # Term 1: E[cos(t*(h_n - h_m))]
    h_diff = h.unsqueeze(0) - h.unsqueeze(1)  # (N, N)
    term1 = torch.cos(t_vals.unsqueeze(-1).unsqueeze(-1) * h_diff).mean(dim=(-1,-2))

    # Term 2: E[cos(t*h_n)] * exp(-t^2/2)
    term2 = torch.cos(t_vals.unsqueeze(-1) * h.unsqueeze(0)).mean(dim=-1) * torch.exp(-t_vals**2 / 2)

    # Term 3: exp(-t^2) [from phi_0(t)^2]
    term3 = torch.exp(-t_vals**2)

    integrand = w * (term1 - 2 * term2 + term3)

    # Trapezoid integration
    dt = (t_max - t_min) / (n_knots - 1)
    T = torch.trapz(integrand, dx=dt)
    return T
```

**Note on SIGReg application**: Per the paper, SIGReg is applied **step-wise** — call SIGReg once per timestep and average, not across the full (B, T, D) tensor at once.

---

## 8. Implementation Priority

### Build First (Core, required for training)
1. **Encoder**: ViT-Tiny (use `timm` or HuggingFace `transformers` library). Extract `[CLS]` token. Add `nn.Linear + nn.BatchNorm1d` projection head.
2. **SIGReg**: The Epps-Pulley statistic is the mathematical heart of the paper. Implement carefully; use trapezoid quadrature. Use vectorized complex exponential operations for efficiency.
3. **Training loop**: Teacher-forcing MSE + SIGReg. No EMA, no stop-gradient.
4. **Predictor**: Causal transformer. AdaLN action conditioning (init to zero). Causal attention mask. Projection head same as encoder.

### Build Second (Required for inference/planning)
5. **CEM planner**: Sample 300 action sequences, evaluate latent rollout cost, keep top 30 elites, update Gaussian distribution, repeat 30 iterations.
6. **Autoregressive rollout**: `z_hat_{t+1} = predictor(z_hat_t, a_t)` starting from encoded initial observation.

### Can Be Deferred (Diagnostic only)
7. **Decoder** (visualization only): Transformer decoder on 192-dim CLS token -> 224x224 image. Not used in training or planning. Used only to visualize what the latent space captures.
8. **Temporal straightness metric** (diagnostic): Cosine similarity between consecutive latent velocity vectors.

### Requires Clarification
- The exact optimizer (AdamW assumed, learning rate not specified in the paper — likely 1e-4 or 3e-4).
- The exact integration of `emb.transpose(0,1)` in `SIGReg` calls — the paper's pseudocode shows `mean(SIGReg(emb.transpose(0, 1)))` which implies SIGReg processes all (T*B) samples at once, but the text says "step-wise". Clarification: apply SIGReg to all B embeddings at each of the T timesteps separately and average over T.
- Whether the predictor projects actions through AdaLN before or after the attention sublayer — standard AdaLN (from DiT paper) applies it before attention+MLP via `scale` and `shift` parameters predicted from action embedding.

---

## 9. Limitations and Caveats

- **Short-horizon planning**: Autoregressive rollout accumulates errors. Planning horizon H=5 steps (25 env steps). Hierarchical world models needed for longer horizons.
- **Offline data dependency**: Requires sufficiently diverse offline datasets. SIGReg struggles in very low-complexity environments (e.g., TwoRoom) where the intrinsic dimensionality is much lower than the 192-dim latent space — the Gaussian prior in high-dim space becomes hard to satisfy.
- **Action label requirement**: Requires explicit action annotations. Inverse dynamics modeling is suggested as future work to relax this.
- **3D visual complexity**: Performance slightly worse than foundation-model-based methods (DINO-WM) on complex 3D environments (OGBench-Cube), especially for rotational quantities.
- **Rotational encoding limitation**: All three methods (LeWM, PLDM, DINO-WM) struggle to recover fine-grained rotational information (block quaternion, block yaw) in compact latent spaces.
- **Single GPU, 15M params**: Efficient, but SIGReg with M=1024 projections has computational cost. The paper reports negligible impact when reducing M, so it can be tuned down.
- **Lambda sensitivity**: Performance degrades sharply at lambda=0.5 (regularizer dominates). Stay in [0.01, 0.2].
