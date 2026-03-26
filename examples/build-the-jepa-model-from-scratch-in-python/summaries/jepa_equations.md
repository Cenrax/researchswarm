# Equations from LeWorldModel (LeWM)

**Paper**: LeWorldModel: Stable End-to-End Joint-Embedding Predictive Architecture from Pixels
**Source**: `/Users/subham/Desktop/codes/agentsclaude/input/week_15/jepa.pdf`
**Date**: 2026-03-26

---

## 1. Loss Functions

### Eq. 1: Prediction Loss (MSE, Teacher-Forcing)

$$
\mathcal{L}_{\text{pred}} \triangleq \|\hat{z}_{t+1} - z_{t+1}\|_2^2, \quad \hat{z}_{t+1} = \text{pred}_\phi(z_t, a_t)
$$

**Variables**:
- $\hat{z}_{t+1} \in \mathbb{R}^d$ — predicted latent embedding at time $t+1$, output of predictor
- $z_{t+1} \in \mathbb{R}^d$ — ground-truth latent embedding at time $t+1$, output of encoder on $o_{t+1}$
- $\text{pred}_\phi$ — predictor network parameterized by $\phi$
- $z_t$ — encoder latent at time $t$
- $a_t$ — action at time $t$

**Used in**: Main training objective; the primary learning signal that drives dynamic modeling.

**Implementation notes**:
- This is plain `F.mse_loss(pred_emb, target_emb)` in PyTorch.
- Teacher-forcing: use ground-truth encoder embeddings as input to predictor at each step, NOT autoregressive predictions.
- Applied over all T-1 consecutive pairs in a sub-trajectory of length T.

---

### Eq. 2 / SIGReg: SIGReg Anti-Collapse Regularizer

$$
\text{SIGReg}(Z) \triangleq \frac{1}{M} \sum_{m=1}^{M} T^{(m)}
$$

**Variables**:
- $Z \in \mathbb{R}^{N \times B \times d}$ — tensor of latent embeddings (history length $N$, batch size $B$, embedding dim $d$)
- $M = 1024$ — number of random projection directions
- $T^{(m)}$ — Epps-Pulley univariate normality test statistic for the $m$-th projection (see EP equation below)

**Used in**: Anti-collapse regularization; second term of the total loss.

**Implementation notes**:
- Applied **step-wise**: call SIGReg independently for each timestep's $(B, d)$ slice and average over timesteps.
- In code: `mean(SIGReg(emb.transpose(0, 1)))` where `emb` is `(B, T, D)`.
- Insensitive to $M$ (ablated over 64 to 1024, flat performance). Use $M=1024$ for correctness, reduce for speed if needed.

---

### Eq. 3: Total LeWM Training Objective

$$
\mathcal{L}_{\text{LeWM}} \triangleq \mathcal{L}_{\text{pred}} + \lambda \cdot \text{SIGReg}(Z)
$$

**Variables**:
- $\lambda = 0.1$ — regularization weight (only effective hyperparameter)
- $\mathcal{L}_{\text{pred}}$ — MSE prediction loss (Eq. 1)
- $\text{SIGReg}(Z)$ — Gaussian regularizer (Eq. 2)

**Used in**: Full training loop; single scalar optimized end-to-end.

**Implementation notes**:
- $\lambda \in [0.01, 0.2]$ gives success rate above 80% on Push-T. Peak near $\lambda = 0.09$.
- Performance collapses at $\lambda = 0.5$ (regularizer dominates).
- Can tune $\lambda$ via bisection search in O(log n).

---

## 2. Forward Pass

### LeWM Encoder and Predictor Equations

$$
z_t = \text{enc}_\theta(o_t)
$$
$$
\hat{z}_{t+1} = \text{pred}_\phi(z_t, a_t)
$$

**Variables**:
- $o_t \in \mathbb{R}^{C \times H \times W}$ — raw pixel observation at time $t$ (C=3, H=W=224)
- $z_t \in \mathbb{R}^{192}$ — latent embedding of observation $o_t$
- $\hat{z}_{t+1} \in \mathbb{R}^{192}$ — predicted latent for next timestep
- $\text{enc}_\theta$ — ViT-Tiny encoder with BatchNorm projection head, parameters $\theta$
- $\text{pred}_\phi$ — causal transformer predictor with AdaLN action conditioning, parameters $\phi$

**Used in**: Every forward pass during training and planning.

**Implementation notes**:
- Encoder: ViT-Tiny (patch=14, layers=12, heads=3, dim=192) + Linear + BatchNorm1d projection.
- Final ViT layer applies LayerNorm; the projection head must use BatchNorm (not LayerNorm) to make SIGReg effective.
- Predictor input: sequence of N=3 latent embeddings with causal masking; action injected at each layer via AdaLN.

---

## 3. Core Mechanism: SIGReg

### Eq. 6: Random Projection

$$
h^{(m)} \triangleq Z u^{(m)}, \quad u^{(m)} \in \mathbb{S}^{D-1}
$$

**Variables**:
- $h^{(m)} \in \mathbb{R}^{N \cdot B}$ — 1D projection of all embeddings onto direction $u^{(m)}$
- $Z \in \mathbb{R}^{(N \cdot B) \times D}$ — embedding matrix (flattened over history and batch)
- $u^{(m)} \in \mathbb{R}^D$ — unit-norm direction sampled uniformly on the $(D-1)$-sphere

**Used in**: First step of SIGReg computation.

**Implementation notes**:
```python
U = torch.randn(D, M)
U = U / U.norm(dim=0, keepdim=True)  # normalize columns to unit norm
H = Z @ U  # (N*B, M)
```

---

### EP: Epps-Pulley Test Statistic

$$
T^{(m)} = \int_{-\infty}^{\infty} w(t) \left| \phi_N(t; h^{(m)}) - \phi_0(t) \right|^2 dt
$$

**Variables**:
- $\phi_N(t; h) = \frac{1}{N} \sum_{n=1}^{N} e^{i t h_n}$ — empirical characteristic function (ECF)
- $\phi_0(t) = e^{-t^2/2}$ — standard Gaussian characteristic function
- $w(t) = e^{-t^2/(2\lambda^2)}$ — Gaussian weighting function (note: this $\lambda$ is an internal SIGReg parameter, distinct from the loss weight $\lambda$)
- $t$ — integration variable

**Used in**: SIGReg; measures how far each 1D projection deviates from standard Gaussian.

**Implementation notes**:
- Integral computed via **trapezoid rule** over $t \in [0.2, 4]$ with $T$ uniformly spaced nodes (paper ablates 4 to 32; insensitive, use ~17).
- The integrand can be expanded in real form to avoid complex arithmetic:
  $$
  |\phi_N(t;h) - \phi_0(t)|^2 = \underbrace{\frac{1}{N^2} \sum_{j,k} \cos(t(h_j - h_k))}_{\text{term 1}} - \underbrace{\frac{2}{N}\sum_n \cos(th_n) e^{-t^2/2}}_{\text{term 2}} + \underbrace{e^{-t^2}}_{\text{term 3}}
  $$
- Numerical stability: the weighting $w(t)$ decays, so large $t$ values contribute little; integration range [0.2, 4] is sufficient.

---

### Cramer-Wold Convergence Guarantee

$$
\text{SIGReg}(Z) \to 0 \iff \mathbb{P}_Z \to \mathcal{N}(0, I)
$$

**Used in**: Theoretical justification; not directly implemented.

**Implementation notes**: This guarantees that minimizing SIGReg over many random directions M drives the full joint distribution toward N(0, I). In practice M=1024 is sufficient; performance is insensitive to M.

---

## 4. Planning Equations

### Eq. 4: Latent Goal-Matching Cost

$$
\mathcal{C}(\hat{z}_H) = \|\hat{z}_H - z_g\|_2^2, \quad z_g = \text{enc}_\theta(o_g)
$$

**Variables**:
- $\hat{z}_H \in \mathbb{R}^{192}$ — predicted latent state at end of planning horizon $H$
- $z_g \in \mathbb{R}^{192}$ — goal latent embedding, obtained by encoding goal image $o_g$
- $H = 5$ — planning horizon (5 steps = 25 env steps with frame-skip 5)

**Used in**: CEM planning objective; evaluates candidate action sequences.

---

### Eq. 5: Optimal Control Problem

$$
a^*_{1:H} = \arg\min_{a_{1:H}} \mathcal{C}(\hat{z}_H)
$$

**Variables**:
- $a^*_{1:H}$ — optimal action sequence of length $H$
- $\hat{z}_{t+1} = \text{pred}_\phi(\hat{z}_t, a_t)$ — autoregressive latent rollout

**Used in**: Planning; solved via CEM.

**Implementation notes (CEM)**:
```python
# CEM hyperparameters
N_samples = 300    # candidate action sequences per iteration
N_elites  = 30     # top candidates kept as "elites"
N_iter    = 30     # optimization iterations (PushT); 10 for others
H         = 5      # planning horizon
sigma_init = 1.0   # initial sampling variance

mu = zeros(H, action_dim)
sigma = ones(H, action_dim) * sigma_init

for _ in range(N_iter):
    actions = sample_gaussian(mu, sigma, N_samples)  # (N_samples, H, A)
    costs = evaluate_in_world_model(z1, actions)      # rollout, compute cost
    elite_idx = argsort(costs)[:N_elites]
    mu = mean(actions[elite_idx], dim=0)
    sigma = std(actions[elite_idx], dim=0)

return mu  # best action sequence
```

---

## 5. Normalization / Regularization

### BatchNorm in Projection Head

The encoder's `[CLS]` token passes through:
```
Linear(192, 192) -> BatchNorm1d(192)
```
The predictor output passes through the same structure. **BatchNorm is critical** — the final ViT layer uses LayerNorm which prevents SIGReg from working. BatchNorm enables the Gaussian distribution matching.

### AdaLN (Adaptive Layer Normalization) in Predictor

At each of the 6 predictor transformer layers, action embedding modulates the layer norm:
$$
\text{AdaLN}(x, a) = \gamma(a) \cdot \text{LayerNorm}(x) + \beta(a)
$$
where $\gamma(a), \beta(a)$ are linear projections of the action embedding $a$, initialized to zero.

**Implementation notes**:
- Zero initialization ensures `gamma=0, beta=0` at the start, meaning the predictor begins training without any action influence and gradually incorporates it.
- Standard reference: DiT paper (Peebles & Xie, 2023).

---

## 6. Diagnostic Metrics

### Eq. 9: Temporal Straightness (diagnostic only, not a training objective)

$$
\mathcal{S}_{\text{straight}} = \frac{1}{B(T-2)} \sum_{i=1}^{B} \sum_{t=1}^{T-2} \frac{\langle v_t^{(i)}, v_{t+1}^{(i)} \rangle}{\|v_t^{(i)}\| \|v_{t+1}^{(i)}\|}
$$

where $v_t^{(i)} = z_{t+1}^{(i)} - z_t^{(i)}$ is the latent velocity vector.

**Variables**:
- $v_t^{(i)} \in \mathbb{R}^d$ — latent velocity at step $t$ for trajectory $i$
- $B$ — batch size
- $T$ — trajectory length

**Used in**: Analysis only; measures how straight latent trajectories are. Value near 1 = straight line. Emerges naturally in LeWM without explicit regularization.

---

## 7. Baseline Reference Equations (for comparison context)

### DINO-WM Loss (Eq. 7, baseline)

$$
\mathcal{L}_{\text{DINO-WM}} = \frac{1}{BT} \sum_{i}^{B} \sum_{t}^{T} \|\hat{z}_{t+1}^{(i)} - z_{t+1}^{(i)}\|_2^2
$$

Uses frozen DINOv2 encoder; predictor-only training.

### PLDM Loss (Eq. 8, baseline — 7-term objective LeWM replaces)

$$
\mathcal{L}_{\text{PLDM}} = \mathcal{L}_{\text{pred}} + \alpha \mathcal{L}_{\text{var}} + \beta \mathcal{L}_{\text{cov}} + \gamma \mathcal{L}_{\text{time-sim}} + \zeta \mathcal{L}_{\text{time-var}} + \nu \mathcal{L}_{\text{time-cov}} + \mu \mathcal{L}_{\text{IDM}}
$$

Has 6 tunable loss weights (O(n^6) grid search). LeWM reduces this to 1 weight ($\lambda$).

---

## Summary Table: Equations by Role

| Equation | Role | Implement? |
|----------|------|-----------|
| `z_t = enc(o_t)` | Forward: encode observation | Yes, core |
| `z_hat = pred(z_t, a_t)` | Forward: predict next latent | Yes, core |
| `L_pred = ||z_hat - z||^2` | Loss: MSE prediction | Yes, core |
| `h^(m) = Z @ u^(m)` | SIGReg: random projection | Yes, core |
| `T^(m) = integral w(t)|phi_N - phi_0|^2 dt` | SIGReg: Epps-Pulley stat | Yes, core |
| `SIGReg = (1/M) sum T^(m)` | SIGReg: aggregate | Yes, core |
| `L_total = L_pred + lambda * SIGReg` | Full objective | Yes, core |
| `C = ||z_hat_H - z_g||^2` | Planning: cost function | Yes, for inference |
| `a* = argmin C` via CEM | Planning: optimization | Yes, for inference |
| `SIGReg -> 0 iff P_Z -> N(0,I)` | Convergence guarantee | No, theoretical |
| `S_straight = cosine sim of velocities` | Diagnostic metric | Optional |
