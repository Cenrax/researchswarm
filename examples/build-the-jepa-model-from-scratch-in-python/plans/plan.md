# Implementation Plan: LeWorldModel (LeWM) JEPA

**Date**: 2026-03-26
**Paper**: LeWorldModel: Stable End-to-End Joint-Embedding Predictive Architecture from Pixels (arXiv 2603.19312v1)
**Dependencies**: See `dependencies.md` in this directory.

---

## 1. Objective Restatement

Build a complete, from-scratch PyTorch implementation of the LeWorldModel (LeWM) JEPA -- a world model that learns latent representations of visual observations and predicts future latent states conditioned on actions, trained end-to-end without EMA or stop-gradient. The system comprises three core neural network modules (ViT-Tiny encoder, causal transformer predictor, SIGReg regularizer), a training pipeline using teacher-forcing with MSE + SIGReg loss on offline trajectory data, and a CEM-based planner for goal-conditioned inference. The implementation must be modular, independently testable, and runnable on a single GPU with approximately 15M total parameters.

---

## 2. Architecture Overview

### 2.1 High-Level Component Diagram

```
+===========================================================================+
|                         LeWM JEPA System                                  |
+===========================================================================+
|                                                                           |
|  +-----------------+    +------------------+    +---------------------+   |
|  |   Data Module   |    |   Model Module   |    |  Planning Module    |   |
|  |                 |    |                  |    |                     |   |
|  | TrajectoryData  |--->| Encoder(ViT-T)   |--->| CEMPlanner          |   |
|  | SubTrajectory   |    | Predictor(ViT-S) |    |  - sample actions   |   |
|  | FrameSkip       |    | ProjectionHead   |    |  - rollout in latent|   |
|  | Augmentations   |    | SIGReg           |    |  - elite selection  |   |
|  +-----------------+    +------------------+    +---------------------+   |
|                                |                         |                |
|                                v                         v                |
|                         +-------------+          +--------------+         |
|                         | Training    |          | Inference    |         |
|                         | Loop        |          | Pipeline     |         |
|                         |             |          |              |         |
|                         | MSE + SIGReg|          | Encode obs   |         |
|                         | Joint optim |          | AR rollout   |         |
|                         | No EMA      |          | CEM optimize |         |
|                         +-------------+          +--------------+         |
|                                                                           |
|  +--------------------+    +--------------------+                         |
|  | Config Module      |    | Diagnostics        |                         |
|  |                    |    |                    |                         |
|  | Hyperparameters    |    | Decoder (optional) |                         |
|  | Model configs      |    | Straightness metric|                         |
|  | Training configs   |    | Embedding viz      |                         |
|  +--------------------+    +--------------------+                         |
+===========================================================================+
```

### 2.2 Data Flow

```
TRAINING:
  Raw images (B, T, 3, 224, 224) + Actions (B, T-1, A)
       |
       v
  [Encoder] -- ViT-Tiny + ProjectionHead(Linear + BatchNorm1d)
       |
       v
  Embeddings z (B, T, 192)
       |
       +---> [SIGReg] per-timestep: z[:, t, :] for t=0..T-1 --> scalar loss
       |
       +---> [Predictor] teacher-forcing: z[:, 0:T-1, :] + actions --> z_hat (B, T-1, 192)
                  |
                  v
              MSE(z_hat, z[:, 1:T, :]) --> scalar loss
       |
       v
  Total Loss = MSE + 0.1 * SIGReg --> backprop through encoder + predictor jointly

INFERENCE (Planning):
  Current obs o_1 --[Encoder]--> z_1
  Goal obs o_g ----[Encoder]--> z_g
       |
       v
  [CEM Planner]
    for iter in 30:
      Sample 300 action sequences from N(mu, sigma)
      for each sequence:
        [Predictor] autoregressive rollout: z_1 -> z_2 -> ... -> z_H
      Cost = ||z_H - z_g||^2
      Keep top 30 elites, update mu/sigma
       |
       v
  Return mu (best action sequence)
```

### 2.3 File Structure

```
lewm/
|-- __init__.py
|-- config.py                  # All hyperparameters and model configs
|-- models/
|   |-- __init__.py
|   |-- encoder.py             # ViT-Tiny encoder + projection head
|   |-- predictor.py           # Causal transformer predictor with AdaLN
|   |-- projection_head.py     # Shared Linear + BatchNorm1d module
|   |-- adaln.py               # Adaptive LayerNorm with zero-init
|   |-- lewm.py                # Combined LeWM model (encoder + predictor)
|-- losses/
|   |-- __init__.py
|   |-- sigreg.py              # SIGReg regularizer (Epps-Pulley + random projections)
|   |-- prediction_loss.py     # MSE prediction loss
|-- data/
|   |-- __init__.py
|   |-- dataset.py             # Trajectory dataset with sub-trajectory sampling
|   |-- transforms.py          # Image preprocessing (resize, normalize)
|-- planning/
|   |-- __init__.py
|   |-- cem.py                 # Cross-Entropy Method planner
|   |-- rollout.py             # Autoregressive latent rollout
|-- training/
|   |-- __init__.py
|   |-- trainer.py             # Training loop
|   |-- logger.py              # Logging utilities
|-- diagnostics/
|   |-- __init__.py
|   |-- decoder.py             # Optional decoder for visualization
|   |-- metrics.py             # Temporal straightness, etc.
|-- tests/
|   |-- test_sigreg.py
|   |-- test_encoder.py
|   |-- test_predictor.py
|   |-- test_adaln.py
|   |-- test_dataset.py
|   |-- test_cem.py
|   |-- test_training.py
|   |-- test_integration.py
```

---

## 3. Dependencies

See the full dependency analysis at:
`/Users/subham/Desktop/codes/agentsclaude/output/20260326_114622_build-the-jepa-model-from-scratch-in-python/plans/dependencies.md`

**Summary**: Core requirements are `torch>=2.1`, `torchvision>=0.16`, `timm>=0.9.12`, and `numpy>=1.24`. Optional packages include `tensorboard`, `matplotlib`, `tqdm`, `gymnasium`, and `pytest`.

---

## 4. Step-by-Step Implementation Plan

---

### Step 1: Configuration Module

**What**: Create a centralized configuration dataclass holding all hyperparameters with paper defaults.

**Why**: Every subsequent module references specific hyperparameters from the LeWM paper. Centralizing them prevents magic numbers and makes ablation studies trivial.

**Files**: `lewm/config.py`

**Acceptance Criteria**:
- All values from the paper's Table 1 and ablation sections are present.
- Config can be instantiated with defaults and overridden per-field.
- Printing the config shows all values.

**Code Skeleton**:
```python
from dataclasses import dataclass, field

@dataclass
class EncoderConfig:
    model_name: str = "vit_tiny_patch16_224"  # timm model name
    patch_size: int = 16         # Note: timm vit_tiny uses 16; paper says 14
    num_layers: int = 12
    num_heads: int = 3
    embed_dim: int = 192
    img_size: int = 224

@dataclass
class PredictorConfig:
    num_layers: int = 6
    num_heads: int = 16
    embed_dim: int = 192
    dropout: float = 0.1
    history_length: int = 3      # N=3 for PushT/OGBench; N=1 for TwoRoom
    action_dim: int = 2          # environment-specific

@dataclass
class SIGRegConfig:
    num_projections: int = 1024  # M
    num_knots: int = 17          # quadrature points
    t_min: float = 0.2
    t_max: float = 4.0
    loss_weight: float = 0.1     # lambda

@dataclass
class TrainingConfig:
    batch_size: int = 128
    sub_trajectory_length: int = 4  # T=4 frames
    frame_skip: int = 5
    num_epochs: int = 10
    learning_rate: float = 3e-4     # assumed, not specified in paper
    weight_decay: float = 0.01      # AdamW default
    optimizer: str = "adamw"

@dataclass
class CEMConfig:
    num_samples: int = 300
    num_elites: int = 30
    num_iterations: int = 30     # 30 for PushT; 10 for others
    horizon: int = 5             # H=5 latent steps = 25 env steps
    sigma_init: float = 1.0

@dataclass
class LeWMConfig:
    encoder: EncoderConfig = field(default_factory=EncoderConfig)
    predictor: PredictorConfig = field(default_factory=PredictorConfig)
    sigreg: SIGRegConfig = field(default_factory=SIGRegConfig)
    training: TrainingConfig = field(default_factory=TrainingConfig)
    cem: CEMConfig = field(default_factory=CEMConfig)
```

---

### Step 2: Projection Head

**What**: Implement the shared projection head used by both the encoder and predictor: `Linear(embed_dim, embed_dim) -> BatchNorm1d(embed_dim)`.

**Why**: The paper (Section 4, "Critical note") explicitly states BatchNorm is required instead of LayerNorm for SIGReg to function. Both encoder and predictor use identical projection heads.

**Files**: `lewm/models/projection_head.py`

**Acceptance Criteria**:
- Input shape `(B, D)` produces output shape `(B, D)`.
- Uses `nn.BatchNorm1d`, not `nn.LayerNorm`.
- No activation function after BatchNorm (paper does not mention one).
- Parameters are trainable.

**Code Skeleton**:
```python
import torch.nn as nn

class ProjectionHead(nn.Module):
    def __init__(self, embed_dim: int = 192):
        super().__init__()
        self.linear = nn.Linear(embed_dim, embed_dim)
        self.bn = nn.BatchNorm1d(embed_dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: (B, D) or (B*T, D) CLS token embeddings
        Returns:
            (B, D) or (B*T, D) projected embeddings
        """
        return self.bn(self.linear(x))
```

---

### Step 3: ViT-Tiny Encoder

**What**: Build the encoder module wrapping a `timm` ViT-Tiny backbone. Extract the `[CLS]` token from the final layer output and pass it through the ProjectionHead.

**Why**: The encoder is the representation learner described in LeWM Section 4. ViT-Tiny with patch=14, 12 layers, 3 heads, dim=192 produces approximately 5M parameters.

**Files**: `lewm/models/encoder.py`

**Acceptance Criteria**:
- Accepts input of shape `(B, 3, 224, 224)` and outputs `(B, 192)`.
- CLS token is correctly extracted (index 0 of sequence dimension).
- ProjectionHead (Linear + BatchNorm1d) is applied after CLS extraction.
- Total parameter count is approximately 5M.
- Works with batched trajectory input `(B*T, 3, 224, 224)` after reshaping.

**Code Skeleton**:
```python
import timm
import torch
import torch.nn as nn
from .projection_head import ProjectionHead

class ViTEncoder(nn.Module):
    def __init__(self, config: EncoderConfig):
        super().__init__()
        # Load ViT-Tiny from timm without pretrained weights
        self.vit = timm.create_model(
            config.model_name,
            pretrained=False,
            num_classes=0,        # remove classification head
            embed_dim=config.embed_dim,
        )
        self.projection = ProjectionHead(config.embed_dim)

    def forward(self, obs: torch.Tensor) -> torch.Tensor:
        """
        Args:
            obs: (B, 3, 224, 224) images
        Returns:
            z: (B, 192) latent embeddings
        """
        # timm ViT with num_classes=0 returns CLS token after final LayerNorm
        cls_token = self.vit(obs)           # (B, 192)
        z = self.projection(cls_token)       # (B, 192)
        return z

    def encode_trajectory(self, obs: torch.Tensor) -> torch.Tensor:
        """
        Args:
            obs: (B, T, 3, 224, 224)
        Returns:
            z: (B, T, 192)
        """
        B, T, C, H, W = obs.shape
        z = self.forward(obs.reshape(B * T, C, H, W))  # (B*T, 192)
        return z.reshape(B, T, -1)                       # (B, T, 192)
```

**Pitfall**: `timm` ViT-Tiny defaults to patch_size=16. The paper says patch_size=14. Either (a) use `vit_tiny_patch14_224` if available in timm, or (b) manually set `patch_size=14` in the timm model creation. The exact patch size affects the number of patch tokens (16x16=256 patches for patch=14 on 224x224 images). Verify the patch count matches expectations.

---

### Step 4: Adaptive Layer Normalization (AdaLN)

**What**: Implement AdaLN with zero-initialization for action conditioning in the predictor.

**Why**: LeWM predictor uses AdaLN at each transformer layer to inject action information (Section 4, Predictor). Zero-init is critical for training stability -- the predictor starts without action influence and gradually learns to use it. This follows the DiT paper convention.

**Files**: `lewm/models/adaln.py`

**Acceptance Criteria**:
- Takes input features `x: (B, S, D)` and action embedding `a: (B, D_a)` or `(B, D)`.
- Outputs modulated features of same shape as `x`.
- At initialization, `gamma=0` and `beta=0`, so output equals `LayerNorm(x)`.
- Gradients flow through both `x` and `a`.

**Code Skeleton**:
```python
import torch
import torch.nn as nn

class AdaLN(nn.Module):
    """Adaptive Layer Normalization with zero initialization."""

    def __init__(self, embed_dim: int, cond_dim: int):
        """
        Args:
            embed_dim: dimension of input features (192)
            cond_dim: dimension of conditioning signal (action embedding)
        """
        super().__init__()
        self.norm = nn.LayerNorm(embed_dim, elementwise_affine=False)
        # Project conditioning to scale and shift
        self.cond_proj = nn.Linear(cond_dim, 2 * embed_dim)
        # Zero initialization: at init, gamma=0, beta=0
        nn.init.zeros_(self.cond_proj.weight)
        nn.init.zeros_(self.cond_proj.bias)

    def forward(self, x: torch.Tensor, cond: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: (B, S, D) input features
            cond: (B, D_cond) conditioning signal (action embedding)
        Returns:
            (B, S, D) modulated features
        """
        gamma, beta = self.cond_proj(cond).chunk(2, dim=-1)  # each (B, D)
        gamma = gamma.unsqueeze(1)  # (B, 1, D)
        beta = beta.unsqueeze(1)    # (B, 1, D)
        return gamma * self.norm(x) + beta
```

**Pitfall**: The zero-init means `gamma * LayerNorm(x) + beta = 0 * LN(x) + 0 = 0` at initialization. This may be intentional (residual connection adds the original input back), or the paper may use `(1 + gamma) * LN(x) + beta` convention. Inspect DiT paper convention: DiT uses `gamma * LN(x) + beta` where residual connections handle the identity path. Since the transformer block has residual connections (`x + attention(adaln(x))`), the zero output from AdaLN means the block initially acts as identity. This is the correct interpretation.

---

### Step 5: Causal Transformer Predictor

**What**: Build the 6-layer causal transformer predictor with 16 attention heads, AdaLN action conditioning, causal masking, learned positional embeddings, and a projection head.

**Why**: This is the dynamics model described in LeWM Section 4. It takes a history of N latent embeddings and predicts the next embedding, conditioned on actions via AdaLN. Causal masking ensures each position can only attend to current and past positions.

**Files**: `lewm/models/predictor.py`

**Acceptance Criteria**:
- Accepts latent history `(B, N, 192)` and actions `(B, N, A)` where N is history length.
- Outputs predicted next embeddings `(B, N, 192)` (one prediction per history position).
- Causal mask prevents attending to future positions.
- AdaLN applied at each of the 6 layers.
- Dropout of 0.1 applied.
- During training (teacher-forcing): input is ground-truth embeddings.
- During inference: input is autoregressively generated embeddings.
- Total parameter count is approximately 10M.

**Code Skeleton**:
```python
import torch
import torch.nn as nn
import torch.nn.functional as F
from .adaln import AdaLN
from .projection_head import ProjectionHead

class PredictorBlock(nn.Module):
    """Single transformer block with AdaLN action conditioning."""

    def __init__(self, embed_dim: int, num_heads: int, dropout: float, action_embed_dim: int):
        super().__init__()
        self.adaln_attn = AdaLN(embed_dim, action_embed_dim)
        self.attn = nn.MultiheadAttention(
            embed_dim, num_heads, dropout=dropout, batch_first=True
        )
        self.adaln_mlp = AdaLN(embed_dim, action_embed_dim)
        self.mlp = nn.Sequential(
            nn.Linear(embed_dim, 4 * embed_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(4 * embed_dim, embed_dim),
            nn.Dropout(dropout),
        )

    def forward(self, x: torch.Tensor, action_emb: torch.Tensor,
                causal_mask: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: (B, S, D)
            action_emb: (B, D_a) action embedding for AdaLN
            causal_mask: (S, S) causal attention mask
        Returns:
            (B, S, D)
        """
        # Self-attention with AdaLN
        h = self.adaln_attn(x, action_emb)
        h, _ = self.attn(h, h, h, attn_mask=causal_mask)
        x = x + h

        # MLP with AdaLN
        h = self.adaln_mlp(x, action_emb)
        h = self.mlp(h)
        x = x + h

        return x


class CausalPredictor(nn.Module):
    """Causal transformer predictor for LeWM."""

    def __init__(self, config: PredictorConfig):
        super().__init__()
        self.embed_dim = config.embed_dim
        self.num_layers = config.num_layers

        # Action embedding: project raw action to embed_dim
        self.action_embed = nn.Sequential(
            nn.Linear(config.action_dim, config.embed_dim),
            nn.GELU(),
            nn.Linear(config.embed_dim, config.embed_dim),
        )

        # Learned positional embeddings
        self.pos_embed = nn.Parameter(
            torch.randn(1, config.history_length, config.embed_dim) * 0.02
        )

        # Transformer blocks
        self.blocks = nn.ModuleList([
            PredictorBlock(
                config.embed_dim, config.num_heads,
                config.dropout, config.embed_dim
            )
            for _ in range(config.num_layers)
        ])

        self.final_norm = nn.LayerNorm(config.embed_dim)
        self.projection = ProjectionHead(config.embed_dim)

    def _make_causal_mask(self, seq_len: int, device: torch.device) -> torch.Tensor:
        """Create causal attention mask (upper triangular = -inf)."""
        mask = torch.triu(
            torch.ones(seq_len, seq_len, device=device) * float('-inf'),
            diagonal=1
        )
        return mask

    def forward(self, z: torch.Tensor, actions: torch.Tensor) -> torch.Tensor:
        """
        Args:
            z: (B, T, D) latent embeddings (teacher-forced ground truth during training)
            actions: (B, T, A) actions for each timestep
        Returns:
            z_hat: (B, T, D) predicted next-step embeddings
        """
        B, T, D = z.shape

        # Embed actions
        action_emb = self.action_embed(actions)  # (B, T, D)
        # For AdaLN: average action embedding across time for now,
        # or apply per-step (implementation choice)
        # Paper: action injected at each layer via AdaLN
        # Use per-step: flatten to (B*T, D) and process per-position
        # Actually: the predictor sees a sequence; AdaLN conditions each layer
        # The action at each position conditions that position's processing
        # Simplification: use mean action embedding per sequence for conditioning
        # Better: pass action sequence and apply position-wise

        # Add positional embeddings
        x = z + self.pos_embed[:, :T, :]

        # Causal mask
        causal_mask = self._make_causal_mask(T, z.device)

        # Transformer blocks with AdaLN
        # AdaLN needs a single conditioning vector per sample.
        # For action sequence, we pool or use the last action.
        # Most natural: use action embedding per step.
        # Since AdaLN expects (B, D), we need to handle T dimension.
        # Approach: process all T steps, conditioning on aggregated actions.
        # Alternative: condition each layer on the full action sequence.
        # Paper is not fully explicit here. Use mean-pooled action embedding.
        action_cond = action_emb.mean(dim=1)  # (B, D)

        for block in self.blocks:
            x = block(x, action_cond, causal_mask)

        x = self.final_norm(x)

        # Project each position through projection head
        B, T, D = x.shape
        z_hat = self.projection(x.reshape(B * T, D)).reshape(B, T, D)

        return z_hat
```

**Pitfall - AdaLN action conditioning granularity**: The paper states "action conditioning via AdaLN at each layer" but does not specify whether a single action vector conditions the entire sequence or whether each position gets its own action. The most likely interpretation is that the action sequence is embedded and each position in the transformer receives its corresponding action embedding through AdaLN. However, since AdaLN typically takes a single conditioning vector, the implementation may need to either (a) pool actions, (b) apply AdaLN position-wise by reshaping, or (c) concatenate action embeddings with the latent tokens. Start with approach (b): reshape to `(B*T, D)`, apply AdaLN, reshape back. If this causes issues, fall back to (a).

**Revised AdaLN approach for per-step conditioning**:
```python
# In PredictorBlock.forward, change AdaLN to accept per-position conditioning:
# action_emb: (B, T, D) instead of (B, D)
# Reshape: (B*T, 1, D) -> apply AdaLN -> reshape back
# Or modify AdaLN to handle (B, T, D) conditioning by broadcasting
```

---

### Step 6: SIGReg Regularizer

**What**: Implement the SIGReg loss comprising: (a) random projection onto M=1024 unit-norm directions, (b) Epps-Pulley univariate normality test statistic with trapezoid quadrature, (c) averaging over projections, (d) step-wise application per timestep.

**Why**: SIGReg is the core anti-collapse mechanism (LeWM Section 3, Equations 2 and 6). It replaces EMA/stop-gradient by forcing the embedding distribution toward N(0, I) via the Cramer-Wold theorem.

**Files**: `lewm/losses/sigreg.py`

**Acceptance Criteria**:
- Input: `(B, D)` embeddings for a single timestep.
- Output: scalar loss value.
- SIGReg value should decrease during training if the distribution approaches N(0, I).
- Sanity check: SIGReg of samples from `torch.randn(B, D)` should be near zero.
- SIGReg of samples from a collapsed distribution (e.g., all zeros) should be large.
- Differentiable with respect to input embeddings.

**Code Skeleton**:
```python
import torch
import torch.nn.functional as F

class SIGReg(torch.nn.Module):
    """Sketched-Isotropic-Gaussian Regularizer."""

    def __init__(self, embed_dim: int = 192, num_projections: int = 1024,
                 num_knots: int = 17, t_min: float = 0.2, t_max: float = 4.0):
        super().__init__()
        self.embed_dim = embed_dim
        self.M = num_projections
        self.num_knots = num_knots
        self.t_min = t_min
        self.t_max = t_max

        # Pre-compute quadrature points (not trainable)
        t_vals = torch.linspace(t_min, t_max, num_knots)
        self.register_buffer('t_vals', t_vals)

        # Pre-compute Gaussian weight w(t) = exp(-t^2/2)
        # NOTE: The paper uses w(t) = exp(-t^2 / (2*lambda^2)) where lambda
        # is an internal SIGReg parameter. The summary uses lambda=1 for w(t).
        # Use exp(-t^2/2) as the default (lambda_internal=1).
        w = torch.exp(-t_vals ** 2 / 2)
        self.register_buffer('w', w)

    def _sample_directions(self, device: torch.device) -> torch.Tensor:
        """Sample M unit-norm random directions on S^{D-1}."""
        U = torch.randn(self.embed_dim, self.M, device=device)
        U = U / U.norm(dim=0, keepdim=True)  # (D, M)
        return U

    def _epps_pulley(self, h: torch.Tensor) -> torch.Tensor:
        """
        Compute Epps-Pulley test statistic for a 1D sample.

        Args:
            h: (N,) 1D projected embeddings (N = batch size)
        Returns:
            scalar test statistic
        """
        N = h.shape[0]
        t = self.t_vals  # (K,)

        # Expand |phi_N(t) - phi_0(t)|^2 in real form:
        # Term 1: (1/N^2) sum_{j,k} cos(t*(h_j - h_k))
        # Term 2: (2/N) sum_n cos(t*h_n) * exp(-t^2/2)
        # Term 3: exp(-t^2)

        # Vectorized over knots K:
        # t: (K,), h: (N,)
        t_h = t.unsqueeze(1) * h.unsqueeze(0)  # (K, N)

        # Term 1: use the identity: (1/N^2) sum cos(t*(hj-hk))
        #        = |mean(exp(i*t*h))|^2 = (mean cos(t*h))^2 + (mean sin(t*h))^2
        cos_th = torch.cos(t_h)  # (K, N)
        sin_th = torch.sin(t_h)  # (K, N)
        mean_cos = cos_th.mean(dim=1)  # (K,)
        mean_sin = sin_th.mean(dim=1)  # (K,)
        term1 = mean_cos ** 2 + mean_sin ** 2  # (K,) = |phi_N(t)|^2

        # Term 2: (2/N) sum cos(t*h_n) * exp(-t^2/2) = 2 * mean_cos * exp(-t^2/2)
        exp_half_t2 = torch.exp(-t ** 2 / 2)  # (K,)
        term2 = 2 * mean_cos * exp_half_t2  # (K,)

        # Term 3: exp(-t^2) = |phi_0(t)|^2
        term3 = torch.exp(-t ** 2)  # (K,)

        integrand = self.w * (term1 - term2 + term3)  # (K,)

        # Trapezoid integration
        dt = (self.t_max - self.t_min) / (self.num_knots - 1)
        result = torch.trapezoid(integrand, dx=dt)
        return result

    def forward(self, Z: torch.Tensor) -> torch.Tensor:
        """
        Compute SIGReg loss for a batch of embeddings at a single timestep.

        Args:
            Z: (B, D) embeddings for one timestep
        Returns:
            scalar SIGReg loss
        """
        B, D = Z.shape

        # Sample random projection directions
        U = self._sample_directions(Z.device)  # (D, M)

        # Project embeddings onto all directions at once
        H = Z @ U  # (B, M)

        # Compute Epps-Pulley for each projection (vectorize over M)
        # This is the hot loop; vectorize if possible
        total = torch.zeros(1, device=Z.device)
        for m in range(self.M):
            total = total + self._epps_pulley(H[:, m])

        return total / self.M

    def forward_vectorized(self, Z: torch.Tensor) -> torch.Tensor:
        """
        Fully vectorized SIGReg (preferred for speed).

        Args:
            Z: (B, D) embeddings
        Returns:
            scalar loss
        """
        B, D = Z.shape
        U = self._sample_directions(Z.device)  # (D, M)
        H = Z @ U  # (B, M)

        t = self.t_vals  # (K,)
        K = t.shape[0]
        M = self.M

        # t_H: (K, B, M) = t[k] * H[b, m]
        t_H = t.view(K, 1, 1) * H.view(1, B, M)  # (K, B, M)

        cos_tH = torch.cos(t_H)  # (K, B, M)
        sin_tH = torch.sin(t_H)  # (K, B, M)

        mean_cos = cos_tH.mean(dim=1)  # (K, M)
        mean_sin = sin_tH.mean(dim=1)  # (K, M)

        term1 = mean_cos ** 2 + mean_sin ** 2  # (K, M) = |phi_N|^2
        exp_half = torch.exp(-t ** 2 / 2).view(K, 1)
        term2 = 2 * mean_cos * exp_half  # (K, M)
        term3 = torch.exp(-t ** 2).view(K, 1)  # (K, 1)

        integrand = self.w.view(K, 1) * (term1 - term2 + term3)  # (K, M)

        # Trapezoid per projection
        dt = (self.t_max - self.t_min) / (self.num_knots - 1)
        # trapezoid: sum of integrand with endpoint correction
        result = torch.trapezoid(integrand, dx=dt, dim=0)  # (M,)

        return result.mean()  # average over M projections


def compute_sigreg_stepwise(embeddings: torch.Tensor, sigreg_module: SIGReg) -> torch.Tensor:
    """
    Apply SIGReg step-wise across timesteps and average.

    Args:
        embeddings: (B, T, D)
        sigreg_module: SIGReg instance
    Returns:
        scalar averaged SIGReg loss
    """
    B, T, D = embeddings.shape
    total = 0.0
    for t in range(T):
        total = total + sigreg_module.forward_vectorized(embeddings[:, t, :])
    return total / T
```

**Critical implementation notes**:
1. The `forward_vectorized` method avoids the M-iteration loop by computing all projections in parallel. This is essential for GPU efficiency with M=1024.
2. Memory concern: the tensor `t_H` has shape `(K, B, M)` = `(17, 128, 1024)` which is about 9M floats (~36 MB). This is manageable.
3. The `|phi_N(t)|^2 = (mean cos)^2 + (mean sin)^2` identity avoids constructing the `(N, N)` pairwise difference matrix which would be `O(N^2)` in memory.
4. Random directions should be resampled each forward pass (not fixed), following standard practice for sketching methods.

---

### Step 7: Prediction Loss

**What**: Implement the MSE prediction loss with teacher-forcing semantics.

**Why**: Equation 1 from the paper. The prediction loss compares predicted next-step embeddings against ground-truth encoder embeddings.

**Files**: `lewm/losses/prediction_loss.py`

**Acceptance Criteria**:
- Computes MSE between `z_hat[:, :-1, :]` (predictions) and `z[:, 1:, :]` (targets).
- Returns scalar loss.
- Correct alignment: prediction at time t should match embedding at time t+1.

**Code Skeleton**:
```python
import torch
import torch.nn.functional as F

def prediction_loss(z: torch.Tensor, z_hat: torch.Tensor) -> torch.Tensor:
    """
    Teacher-forcing MSE prediction loss.

    Args:
        z: (B, T, D) ground-truth encoder embeddings
        z_hat: (B, T, D) predicted embeddings from predictor
    Returns:
        scalar MSE loss
    """
    # z_hat[:, t] predicts z[:, t+1]
    # So compare z_hat[:, 0:T-1] with z[:, 1:T]
    target = z[:, 1:, :]     # (B, T-1, D)
    pred = z_hat[:, :-1, :]  # (B, T-1, D)
    return F.mse_loss(pred, target)
```

---

### Step 8: Combined LeWM Model

**What**: Create the combined model class that holds the encoder and predictor, with methods for training forward pass and inference rollout.

**Why**: Clean separation of the training forward pass (teacher-forcing) from the inference forward pass (autoregressive rollout). This is the top-level model object.

**Files**: `lewm/models/lewm.py`

**Acceptance Criteria**:
- `forward()` for training: encodes all observations, runs predictor with teacher-forcing, returns embeddings and predictions.
- `rollout()` for inference: encodes initial observation, autoregressively predicts H steps.
- Both encoder and predictor are jointly optimized (single optimizer).

**Code Skeleton**:
```python
import torch
import torch.nn as nn
from .encoder import ViTEncoder
from .predictor import CausalPredictor

class LeWM(nn.Module):
    def __init__(self, encoder_config, predictor_config):
        super().__init__()
        self.encoder = ViTEncoder(encoder_config)
        self.predictor = CausalPredictor(predictor_config)

    def forward(self, obs: torch.Tensor, actions: torch.Tensor):
        """
        Training forward pass with teacher-forcing.

        Args:
            obs: (B, T, 3, 224, 224)
            actions: (B, T, A)
        Returns:
            z: (B, T, D) encoder embeddings (ground truth)
            z_hat: (B, T, D) predictor outputs
        """
        z = self.encoder.encode_trajectory(obs)  # (B, T, D)
        z_hat = self.predictor(z, actions)         # (B, T, D)
        return z, z_hat

    @torch.no_grad()
    def rollout(self, obs_init: torch.Tensor, actions: torch.Tensor) -> torch.Tensor:
        """
        Autoregressive rollout for planning.

        Args:
            obs_init: (B, 3, 224, 224) initial observation
            actions: (B, H, A) action sequence
        Returns:
            z_traj: (B, H+1, D) predicted latent trajectory
        """
        z = self.encoder(obs_init)  # (B, D)
        z_traj = [z]

        for t in range(actions.shape[1]):
            # Build history window for predictor
            history = torch.stack(z_traj[-self.predictor.history_length:], dim=1)
            a_t = actions[:, t:t+1, :]
            z_next = self.predictor(history, a_t)[:, -1, :]  # take last prediction
            z_traj.append(z_next)

        return torch.stack(z_traj, dim=1)  # (B, H+1, D)
```

---

### Step 9: Dataset and DataLoader

**What**: Implement a PyTorch Dataset for offline trajectory data that samples sub-trajectories of length T=4 with frame_skip=5.

**Why**: The training data is offline trajectories of (observation, action) pairs. The paper uses sub-trajectory length T=4 and frame_skip=5, meaning each sub-trajectory spans 20 raw environment steps but uses every 5th frame.

**Files**: `lewm/data/dataset.py`, `lewm/data/transforms.py`

**Acceptance Criteria**:
- Dataset stores full trajectories and samples sub-trajectories on `__getitem__`.
- Frame skip of 5 is applied: if raw trajectory has frames 0,1,2,...,19, a sub-trajectory with skip=5 uses frames 0,5,10,15.
- Images are resized to 224x224 and normalized.
- Actions are sliced to match the frame-skipped sub-trajectory.
- Supports both HDF5 and in-memory formats.
- DataLoader produces batches of shape `obs: (B, T, 3, 224, 224)`, `actions: (B, T, A)`.

**Code Skeleton**:
```python
import torch
from torch.utils.data import Dataset, DataLoader
import numpy as np

class TrajectoryDataset(Dataset):
    """Dataset for offline trajectories with sub-trajectory sampling."""

    def __init__(self, trajectories: list, sub_traj_length: int = 4,
                 frame_skip: int = 5, transform=None):
        """
        Args:
            trajectories: list of dicts, each with:
                'observations': np.ndarray (L, H, W, 3) or (L, 3, H, W)
                'actions': np.ndarray (L, A)
            sub_traj_length: T, number of frames in sub-trajectory
            frame_skip: skip between frames
            transform: torchvision transform for images
        """
        self.trajectories = trajectories
        self.T = sub_traj_length
        self.skip = frame_skip
        self.transform = transform

        # Pre-compute valid (trajectory_idx, start_frame) pairs
        self.valid_indices = []
        required_length = (sub_traj_length - 1) * frame_skip + 1
        for traj_idx, traj in enumerate(trajectories):
            traj_len = len(traj['observations'])
            for start in range(traj_len - required_length + 1):
                self.valid_indices.append((traj_idx, start))

    def __len__(self):
        return len(self.valid_indices)

    def __getitem__(self, idx):
        traj_idx, start = self.valid_indices[idx]
        traj = self.trajectories[traj_idx]

        # Extract frames with skip
        frame_indices = [start + t * self.skip for t in range(self.T)]
        obs = [traj['observations'][i] for i in frame_indices]
        actions = [traj['actions'][i] for i in frame_indices]

        # Apply transforms
        if self.transform:
            obs = [self.transform(o) for o in obs]

        obs = torch.stack(obs)        # (T, 3, 224, 224)
        actions = torch.tensor(np.array(actions), dtype=torch.float32)  # (T, A)

        return {'obs': obs, 'actions': actions}
```

```python
# lewm/data/transforms.py
from torchvision import transforms

def get_default_transform(img_size: int = 224):
    return transforms.Compose([
        transforms.ToPILImage(),
        transforms.Resize((img_size, img_size)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406],
                             std=[0.229, 0.224, 0.225]),
    ])
```

---

### Step 10: Training Loop

**What**: Implement the full training loop combining MSE prediction loss and step-wise SIGReg, with joint optimization of encoder and predictor using AdamW.

**Why**: Algorithm 1 in the paper. This is the core training procedure: teacher-forcing, no EMA, no stop-gradient, single optimizer.

**Files**: `lewm/training/trainer.py`

**Acceptance Criteria**:
- Loss decreases over iterations.
- SIGReg component decreases (embeddings approach Gaussian).
- MSE component decreases (predictor learns dynamics).
- No representation collapse (embedding variance remains > 0).
- Runs for 10 epochs on the dataset.
- Logs both loss components separately.

**Code Skeleton**:
```python
import torch
from torch.utils.data import DataLoader
from lewm.models.lewm import LeWM
from lewm.losses.sigreg import SIGReg, compute_sigreg_stepwise
from lewm.losses.prediction_loss import prediction_loss

class Trainer:
    def __init__(self, model: LeWM, config, device: str = 'cuda'):
        self.model = model.to(device)
        self.config = config
        self.device = device

        self.sigreg = SIGReg(
            embed_dim=config.sigreg.embed_dim if hasattr(config.sigreg, 'embed_dim')
                      else config.encoder.embed_dim,
            num_projections=config.sigreg.num_projections,
            num_knots=config.sigreg.num_knots,
            t_min=config.sigreg.t_min,
            t_max=config.sigreg.t_max,
        ).to(device)

        self.optimizer = torch.optim.AdamW(
            model.parameters(),
            lr=config.training.learning_rate,
            weight_decay=config.training.weight_decay,
        )

    def train_epoch(self, dataloader: DataLoader) -> dict:
        self.model.train()
        epoch_stats = {'pred_loss': 0, 'sigreg_loss': 0, 'total_loss': 0, 'steps': 0}

        for batch in dataloader:
            obs = batch['obs'].to(self.device)        # (B, T, 3, 224, 224)
            actions = batch['actions'].to(self.device) # (B, T, A)

            # Forward pass
            z, z_hat = self.model(obs, actions)

            # Losses
            l_pred = prediction_loss(z, z_hat)
            l_sigreg = compute_sigreg_stepwise(z, self.sigreg)
            loss = l_pred + self.config.sigreg.loss_weight * l_sigreg

            # Backward
            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()

            # Logging
            epoch_stats['pred_loss'] += l_pred.item()
            epoch_stats['sigreg_loss'] += l_sigreg.item()
            epoch_stats['total_loss'] += loss.item()
            epoch_stats['steps'] += 1

        # Average
        for k in ['pred_loss', 'sigreg_loss', 'total_loss']:
            epoch_stats[k] /= max(epoch_stats['steps'], 1)

        return epoch_stats

    def train(self, dataloader: DataLoader):
        for epoch in range(self.config.training.num_epochs):
            stats = self.train_epoch(dataloader)
            print(f"Epoch {epoch+1}/{self.config.training.num_epochs} | "
                  f"Pred: {stats['pred_loss']:.4f} | "
                  f"SIGReg: {stats['sigreg_loss']:.6f} | "
                  f"Total: {stats['total_loss']:.4f}")

            # Monitor for collapse
            self._check_collapse(dataloader)

    @torch.no_grad()
    def _check_collapse(self, dataloader):
        """Check if embeddings have collapsed by measuring variance."""
        self.model.eval()
        batch = next(iter(dataloader))
        obs = batch['obs'].to(self.device)
        z = self.model.encoder.encode_trajectory(obs)
        var = z.var(dim=0).mean().item()
        if var < 1e-4:
            print(f"WARNING: Potential collapse detected! Embedding variance = {var:.6f}")
        self.model.train()
```

---

### Step 11: CEM Planner

**What**: Implement the Cross-Entropy Method planner that optimizes action sequences in latent space to reach a goal.

**Why**: Algorithm 2 in the paper (Equations 4 and 5). CEM is the inference-time planning algorithm that uses the learned world model.

**Files**: `lewm/planning/cem.py`, `lewm/planning/rollout.py`

**Acceptance Criteria**:
- Given initial observation and goal observation, returns an optimized action sequence.
- Cost decreases over CEM iterations.
- Elite selection correctly keeps top-K lowest-cost samples.
- Action sequences are clipped to valid action bounds.
- Works with the LeWM model in eval mode.

**Code Skeleton**:
```python
import torch
from lewm.models.lewm import LeWM

class CEMPlanner:
    def __init__(self, model: LeWM, config, device: str = 'cuda'):
        self.model = model
        self.config = config
        self.device = device

    @torch.no_grad()
    def plan(self, obs_init: torch.Tensor, obs_goal: torch.Tensor,
             action_low: torch.Tensor = None,
             action_high: torch.Tensor = None) -> torch.Tensor:
        """
        Plan an action sequence using CEM.

        Args:
            obs_init: (3, 224, 224) or (1, 3, 224, 224) initial observation
            obs_goal: (3, 224, 224) or (1, 3, 224, 224) goal observation
            action_low: (A,) lower bounds for actions
            action_high: (A,) upper bounds for actions
        Returns:
            best_actions: (H, A) optimized action sequence
        """
        self.model.eval()
        H = self.config.cem.horizon
        A = self.config.predictor.action_dim
        N = self.config.cem.num_samples
        K = self.config.cem.num_elites
        n_iter = self.config.cem.num_iterations

        # Ensure batch dimension
        if obs_init.dim() == 3:
            obs_init = obs_init.unsqueeze(0)
        if obs_goal.dim() == 3:
            obs_goal = obs_goal.unsqueeze(0)

        # Encode initial and goal observations
        z_init = self.model.encoder(obs_init.to(self.device))  # (1, D)
        z_goal = self.model.encoder(obs_goal.to(self.device))  # (1, D)

        # Initialize CEM distribution
        mu = torch.zeros(H, A, device=self.device)
        sigma = torch.ones(H, A, device=self.device) * self.config.cem.sigma_init

        for iteration in range(n_iter):
            # Sample action sequences: (N, H, A)
            noise = torch.randn(N, H, A, device=self.device)
            actions = mu.unsqueeze(0) + sigma.unsqueeze(0) * noise

            # Clip to action bounds
            if action_low is not None and action_high is not None:
                actions = actions.clamp(
                    action_low.to(self.device),
                    action_high.to(self.device)
                )

            # Rollout each action sequence
            # Expand z_init to (N, D)
            z_current = z_init.expand(N, -1)  # (N, D)
            z_traj = self.model.rollout(
                obs_init.expand(N, -1, -1, -1), actions
            )  # (N, H+1, D)

            # Cost: L2 distance at final step to goal
            z_final = z_traj[:, -1, :]  # (N, D)
            costs = ((z_final - z_goal) ** 2).sum(dim=-1)  # (N,)

            # Select elites
            elite_idx = costs.argsort()[:K]
            elite_actions = actions[elite_idx]  # (K, H, A)

            # Update distribution
            mu = elite_actions.mean(dim=0)    # (H, A)
            sigma = elite_actions.std(dim=0)  # (H, A)

        return mu  # (H, A) best action sequence
```

---

### Step 12: Optional Decoder for Visualization

**What**: Implement a simple transformer decoder that reconstructs images from latent embeddings for visualization and debugging purposes.

**Why**: The paper mentions a decoder for diagnostic visualization. Not used in training or planning, but useful for verifying the encoder learns meaningful representations.

**Files**: `lewm/diagnostics/decoder.py`

**Acceptance Criteria**:
- Input: `(B, 192)` latent embedding.
- Output: `(B, 3, 224, 224)` reconstructed image.
- Only used for visualization; not part of the training loop.
- Trained separately with frozen encoder if needed.

**Code Skeleton**:
```python
import torch
import torch.nn as nn

class LatentDecoder(nn.Module):
    """Cross-attention transformer decoder for visualization."""

    def __init__(self, embed_dim: int = 192, img_size: int = 224,
                 patch_size: int = 16, num_layers: int = 4, num_heads: int = 4):
        super().__init__()
        self.num_patches = (img_size // patch_size) ** 2
        self.patch_size = patch_size

        # Learnable query tokens (one per output patch)
        self.query_tokens = nn.Parameter(
            torch.randn(1, self.num_patches, embed_dim) * 0.02
        )
        self.pos_embed = nn.Parameter(
            torch.randn(1, self.num_patches, embed_dim) * 0.02
        )

        decoder_layer = nn.TransformerDecoderLayer(
            d_model=embed_dim, nhead=num_heads, batch_first=True
        )
        self.decoder = nn.TransformerDecoder(decoder_layer, num_layers=num_layers)

        # Project patches back to pixel space
        self.patch_proj = nn.Linear(embed_dim, 3 * patch_size * patch_size)

    def forward(self, z: torch.Tensor) -> torch.Tensor:
        """
        Args:
            z: (B, D) latent embedding
        Returns:
            img: (B, 3, 224, 224)
        """
        B = z.shape[0]
        memory = z.unsqueeze(1)  # (B, 1, D)
        queries = self.query_tokens.expand(B, -1, -1) + self.pos_embed

        out = self.decoder(queries, memory)  # (B, num_patches, D)
        patches = self.patch_proj(out)  # (B, num_patches, 3*P*P)

        # Reshape to image
        P = self.patch_size
        H_patches = W_patches = int(self.num_patches ** 0.5)
        img = patches.reshape(B, H_patches, W_patches, 3, P, P)
        img = img.permute(0, 3, 1, 4, 2, 5).reshape(B, 3, H_patches * P, W_patches * P)
        return img
```

---

### Step 13: Diagnostic Metrics

**What**: Implement the temporal straightness metric and embedding distribution analysis.

**Why**: Equation 9 in the paper. These diagnostics help verify the model is learning meaningful, non-collapsed representations.

**Files**: `lewm/diagnostics/metrics.py`

**Acceptance Criteria**:
- Temporal straightness returns a value in [-1, 1]; closer to 1 = straighter trajectories.
- Embedding variance monitoring detects collapse.
- Can be called during or after training.

**Code Skeleton**:
```python
import torch

def temporal_straightness(z: torch.Tensor) -> float:
    """
    Compute temporal straightness metric (Eq. 9).

    Args:
        z: (B, T, D) latent trajectories
    Returns:
        float: average cosine similarity between consecutive velocity vectors
    """
    v = z[:, 1:, :] - z[:, :-1, :]  # (B, T-1, D) velocity
    v_t = v[:, :-1, :]    # (B, T-2, D)
    v_tp1 = v[:, 1:, :]   # (B, T-2, D)

    cos_sim = torch.nn.functional.cosine_similarity(v_t, v_tp1, dim=-1)  # (B, T-2)
    return cos_sim.mean().item()

def embedding_stats(z: torch.Tensor) -> dict:
    """
    Compute embedding distribution statistics for monitoring.

    Args:
        z: (B, D) embeddings
    Returns:
        dict with mean_norm, std_per_dim, mean_per_dim
    """
    return {
        'mean_norm': z.norm(dim=-1).mean().item(),
        'std_per_dim': z.std(dim=0).mean().item(),
        'mean_per_dim': z.mean(dim=0).abs().mean().item(),
        'max_abs': z.abs().max().item(),
    }
```

---

## 5. Data and Preprocessing

### Expected Input Format

The system expects offline trajectory datasets. Each trajectory is a sequence of (observation, action) pairs.

| Field | Shape | Type | Description |
|-------|-------|------|-------------|
| `observations` | `(L, H, W, 3)` or `(L, 3, H, W)` | uint8 or float32 | RGB images, variable resolution |
| `actions` | `(L, A)` | float32 | Continuous actions, A is environment-specific (e.g., 2 for PushT) |

### Preprocessing Pipeline

```
Raw image (any resolution, uint8)
    -> Resize to 224x224
    -> Convert to float32 tensor [0, 1]
    -> Normalize with ImageNet mean/std
    -> Result: (3, 224, 224) float32 tensor

Sub-trajectory sampling:
    Given trajectory of length L, frame_skip=5, T=4:
    Required span = (T-1) * frame_skip + 1 = 16 raw frames
    Valid start indices: 0 to L-16
    Sampled frames: [start, start+5, start+10, start+15]
    Actions: same indices as observations
```

### Dataset Requirements

- **PushT**: 2D pushing task, action_dim=2. Datasets available via the PushT benchmark.
- **TwoRoom**: Navigation task, action_dim=2. History N=1.
- **OGBench-Cube**: 3D manipulation, action_dim varies.

For initial development, a synthetic dataset can be created:
```python
def make_dummy_dataset(num_trajectories=100, traj_length=50, action_dim=2):
    """Generate random trajectories for testing."""
    trajectories = []
    for _ in range(num_trajectories):
        trajectories.append({
            'observations': np.random.randint(0, 255, (traj_length, 64, 64, 3), dtype=np.uint8),
            'actions': np.random.randn(traj_length, action_dim).astype(np.float32),
        })
    return trajectories
```

---

## 6. Testing Strategy

### Unit Tests

| Test File | Tests | What to Verify |
|-----------|-------|----------------|
| `test_sigreg.py` | `test_gaussian_input_low_loss` | SIGReg of `torch.randn(128, 192)` is near zero (< 0.01) |
| | `test_collapsed_input_high_loss` | SIGReg of `torch.zeros(128, 192)` is large (> 0.1) |
| | `test_gradient_flows` | `loss.backward()` produces non-None gradients on input |
| | `test_vectorized_matches_loop` | `forward_vectorized` matches `forward` within tolerance |
| | `test_num_projections` | Increasing M reduces variance of SIGReg estimate |
| `test_encoder.py` | `test_output_shape` | Input `(4, 3, 224, 224)` -> output `(4, 192)` |
| | `test_trajectory_encoding` | Input `(2, 4, 3, 224, 224)` -> output `(2, 4, 192)` |
| | `test_parameter_count` | Total params approximately 5M (within 20%) |
| | `test_batchnorm_in_head` | Verify projection head contains `BatchNorm1d`, not `LayerNorm` |
| `test_predictor.py` | `test_output_shape` | Input `(B, T, 192)` + `(B, T, 2)` -> output `(B, T, 192)` |
| | `test_causal_masking` | Changing future inputs does not change past outputs |
| | `test_adaln_zero_init` | At init, predictor block output is approximately residual identity |
| | `test_parameter_count` | Total params approximately 10M (within 20%) |
| `test_adaln.py` | `test_zero_init_output` | At init, `AdaLN(x, a) = 0` (so residual block = identity) |
| | `test_gradient_through_condition` | Gradients flow through action conditioning |
| `test_dataset.py` | `test_sub_trajectory_shape` | Output batch has correct shapes |
| | `test_frame_skip` | Frame indices are correctly spaced by `frame_skip` |
| | `test_boundary_conditions` | No index-out-of-bounds at trajectory ends |
| `test_cem.py` | `test_cost_decreases` | Average cost of elites decreases over CEM iterations |
| | `test_action_bounds` | Returned actions respect bounds if specified |
| | `test_deterministic_goal` | With a simple model, CEM finds actions reaching the goal |

### Integration Tests

| Test | Description |
|------|-------------|
| `test_full_forward_pass` | obs -> encoder -> predictor -> loss computation, no errors |
| `test_backward_pass` | Full loss backward propagates to both encoder and predictor parameters |
| `test_single_training_step` | One step of training reduces loss (on fixed synthetic data) |
| `test_no_collapse_10_steps` | After 10 training steps, embedding variance > threshold |
| `test_plan_and_rollout` | Encode obs, plan with CEM, rollout matches plan shape |

### Evaluation Metrics (from paper)

| Metric | Environment | Target |
|--------|-------------|--------|
| Success Rate | PushT | 96% (ViT-S predictor, dropout=0.1) |
| Temporal Straightness | PushT | > 0.8 (emerges naturally) |
| SIGReg value | All | Should decrease to near 0 during training |
| Embedding variance | All | Should remain > 0.1 (no collapse) |

---

## 7. Integration with Existing Agents

### Python API Surface

```python
from lewm.models.lewm import LeWM
from lewm.planning.cem import CEMPlanner
from lewm.config import LeWMConfig

# Initialize
config = LeWMConfig()
model = LeWM(config.encoder, config.predictor)
model.load_state_dict(torch.load("lewm_checkpoint.pt"))

# Plan
planner = CEMPlanner(model, config, device='cuda')
action_sequence = planner.plan(obs_current, obs_goal)

# Get next action
next_action = action_sequence[0]  # first action in planned sequence
```

### Serialization

- Model weights: `torch.save(model.state_dict(), path)` / `torch.load(path)`
- Config: serialize `LeWMConfig` as JSON or YAML using `dataclasses.asdict()`
- Checkpoint format:
  ```python
  checkpoint = {
      'model_state_dict': model.state_dict(),
      'optimizer_state_dict': optimizer.state_dict(),
      'config': dataclasses.asdict(config),
      'epoch': epoch,
      'loss': loss,
  }
  torch.save(checkpoint, 'lewm_checkpoint.pt')
  ```

### Agent Integration

For an OpenAI or Claude agent to use this model:

```python
# Wrapper function for agent tool use
def plan_action(current_image_path: str, goal_image_path: str) -> list[float]:
    """
    Given current and goal images, return the next action.

    Args:
        current_image_path: path to current observation image
        goal_image_path: path to goal observation image
    Returns:
        list of floats representing the action vector
    """
    obs = load_and_preprocess(current_image_path)  # -> (1, 3, 224, 224)
    goal = load_and_preprocess(goal_image_path)     # -> (1, 3, 224, 224)

    action_seq = planner.plan(obs, goal)  # (H, A)
    return action_seq[0].cpu().tolist()   # first action
```

### I/O Format

| Input | Format |
|-------|--------|
| Observation | RGB image, any format (PIL, numpy, path) -> preprocessed to `(1, 3, 224, 224)` tensor |
| Goal | Same as observation |
| Action bounds | `(A,)` tensors for low/high |

| Output | Format |
|--------|--------|
| Action | `(A,)` float32 tensor or Python list |
| Action sequence | `(H, A)` float32 tensor |
| Latent embedding | `(D,)` float32 tensor (for advanced use) |

---

## 8. Risks and Open Questions

### Ambiguities in the Paper

1. **Patch size 14 vs 16**: The paper states patch_size=14, but `timm`'s `vit_tiny_patch14_224` may not exist as a standard variant. `vit_tiny_patch16_224` is standard. If patch=14 is required, the model must be constructed manually or patched. The difference affects the number of tokens (256 for patch=14 vs 196 for patch=16 on 224x224 images) and total parameter count.

2. **AdaLN granularity**: The paper does not specify whether a single pooled action embedding conditions the entire sequence or whether per-position action embeddings are used. The per-position interpretation is more natural for a causal predictor but complicates the AdaLN interface.

3. **SIGReg internal lambda**: The weighting function `w(t) = exp(-t^2 / (2*lambda^2))` uses a lambda that may be distinct from the loss weight lambda=0.1. The summary and pseudocode use `exp(-t^2/2)` (lambda_internal=1). Clarification needed, but lambda_internal=1 appears to be the default.

4. **Optimizer and learning rate**: The paper does not specify the optimizer or learning rate. AdamW with lr=3e-4 is assumed based on common practice for ViT training.

5. **Predictor input**: Whether the predictor takes `[z_{t-N+1}, ..., z_t]` and predicts `z_{t+1}` (sliding window) or processes the full sub-trajectory with causal masking is not fully explicit. The causal mask interpretation (full sub-trajectory) is more consistent with the teacher-forcing description.

6. **Action embedding dimension**: The paper says the predictor has embed_dim=192, but whether action embeddings match this dimension or are projected from a smaller action space is not stated. Projecting action_dim -> 192 via a small MLP is the standard approach.

### Scalability Concerns

1. **SIGReg memory**: The vectorized implementation creates a tensor of shape `(K, B, M) = (17, 128, 1024)`. This is manageable (~36 MB at float32), but increasing batch size or M could become a bottleneck on limited GPU memory.

2. **Encoder through trajectory**: Encoding `B * T = 128 * 4 = 512` images through ViT-Tiny in a single forward pass requires significant memory. Gradient checkpointing or micro-batching may be needed.

3. **CEM planning speed**: At inference, CEM performs 300 rollouts x 30 iterations = 9000 predictor forward passes per planning step. With H=5 steps each, this is 45,000 predictor calls. Batching the 300 samples helps, but real-time planning may require reducing these numbers.

### Implementation Pitfalls

1. **BatchNorm vs LayerNorm**: Using LayerNorm instead of BatchNorm in the projection head will cause SIGReg to fail silently (the loss may decrease but the representation will collapse). This is the single most critical implementation detail.

2. **Teacher-forcing alignment**: The prediction at position t must be compared against the embedding at position t+1. Off-by-one errors here will silently degrade performance.

3. **SIGReg step-wise application**: SIGReg must be applied per-timestep and averaged, not across the full `(B*T, D)` tensor. Applying it across all timesteps conflates temporal structure with batch structure.

4. **AdaLN zero-init verification**: If zero-init is not correctly applied, the predictor may be unstable in early training. Verify by checking that `AdaLN(x, a)` returns approximately zero with freshly initialized parameters.

5. **Causal mask direction**: Ensure the mask prevents attending to future tokens (upper triangular should be masked out), not past tokens. PyTorch `nn.MultiheadAttention` uses `attn_mask` where `True` or `float('-inf')` means "do not attend."

6. **Random projection resampling**: Random directions in SIGReg should be resampled each forward pass for stochastic regularization. Using fixed directions may reduce the regularizer's effectiveness.

7. **BatchNorm in eval mode**: During inference, `BatchNorm1d` uses running statistics. Ensure the model is in `train()` mode during training and `eval()` mode during planning. BatchNorm with very small batch sizes during inference may produce poor estimates -- consider using the running statistics (which is the default in eval mode).

---

## Appendix: Critical Hyperparameters Quick Reference

| Parameter | Value | Source |
|-----------|-------|--------|
| Encoder: ViT-Tiny, 12L/3H/d=192, patch=14 | 5M params | LeWM Table 1 |
| Predictor: 6L/16H/d=192, dropout=0.1 | 10M params | LeWM Table 1 |
| Projection head: Linear(192,192) + BatchNorm1d(192) | Both enc/pred | LeWM Section 4 |
| AdaLN zero-init | gamma=0, beta=0 | LeWM Section 4 |
| SIGReg M=1024 projections | Insensitive above 64 | LeWM ablation |
| SIGReg t in [0.2, 4], 17 knots | Insensitive to knot count | LeWM ablation |
| SIGReg loss weight lambda=0.1 | Range [0.01, 0.2] | LeWM ablation |
| Batch size=128 | Fixed | LeWM Section 5 |
| Sub-trajectory T=4, frame_skip=5 | Fixed | LeWM Section 5 |
| Image resolution 224x224 | Fixed | LeWM Section 5 |
| Training epochs=10 | Fixed | LeWM Section 5 |
| CEM: 300 samples, 30 elites, 30 iters, H=5 | PushT defaults | LeWM Algorithm 2 |
| History length N=3 | PushT/OGBench; N=1 for TwoRoom | LeWM Section 5 |
