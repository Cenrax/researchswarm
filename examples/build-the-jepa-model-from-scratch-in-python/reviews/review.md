# LeWorldModel (LeWM) JEPA — Code Review (Re-Review After Fixes)

**Verdict**: NEEDS_CHANGES
**Date**: 2026-03-26
**Reviewer**: Code Review Specialist (Claude Agent)
**Previous Review Verdict**: NEEDS_CHANGES

---

## 1. Overall Assessment

**NEEDS_CHANGES**

Three of the four requested fixes were applied correctly and completely: gradient clipping is in place, the CEM planner correctly restores model training mode, and per-position action conditioning is properly implemented without mean pooling. However, the patch size fix was only **partially applied**: the `EncoderConfig.patch_size` integer was corrected to 14, but the `model_name` string remains `"vit_tiny_patch16_224"`. Since timm uses the model name string as the primary identifier for architecture construction, this leaves a residual defect that may cause the encoder to silently load a patch-16 architecture and ignore the overriding `patch_size=14` parameter. This issue must be fully resolved before the implementation can be considered a faithful reproduction.

---

## 2. Summary

Significant progress was made since the first review: the two most algorithmically impactful fixes (per-step action conditioning and gradient clipping) are correctly implemented and the CEM mode-restoration bug is fully resolved. The single remaining blocker is a half-applied patch size fix in `config.py` where the model name string `"vit_tiny_patch16_224"` was not updated to match the corrected integer `patch_size = 14`. All other previously raised issues remain at their previous status — no new regressions were introduced.

---

## 3. Verification of Requested Fixes

### Fix 1: Encoder Patch Size Changed to 14
**Status: PARTIALLY APPLIED — still requires action**

`lewm/config.py` line 10 now reads `patch_size: int = 14`. The docstring in `encoder.py` was correctly updated to say `patch=14`. However, `config.py` line 9 still reads:

```python
model_name: str = "vit_tiny_patch16_224"
```

The `ViTEncoder.__init__` calls `timm.create_model(config.model_name, ..., patch_size=config.patch_size)`. The behavior depends on whether timm accepts a `patch_size` override for a model named `"vit_tiny_patch16_224"`. In timm's current API, named models have baked-in defaults; the `patch_size` keyword argument may be accepted for some model families but silently ignored for others, or it may raise an error. In the best case the override works; in the worst case the encoder uses patch 16. The string must be updated to `"vit_tiny_patch14_224"` to eliminate ambiguity.

**Required fix:**
```python
# lewm/config.py line 9
model_name: str = "vit_tiny_patch14_224"
```

Note: run `timm.list_models("vit_tiny*patch14*")` to confirm the exact model name available in the installed timm version before committing.

---

### Fix 2: Per-Position Action Conditioning in Predictor
**Status: CORRECTLY APPLIED**

`predictor.py` line 139 now reads:
```python
action_emb = self.action_embed(actions)  # (B, T, D) — no mean pooling
```

The per-step `action_emb` of shape `(B, T, D)` is passed directly to each `PredictorBlock.forward()`, which passes it to `AdaLN`. The `AdaLN.forward()` was also updated to handle both `(B, D_cond)` broadcast conditioning and `(B, S, D_cond)` per-position conditioning via a `dim` check. The test in `test_predictor.py::TestPredictor::test_adaln_zero_init` now explicitly tests both cases.

**Residual minor issue:** `predict_step` at line 171 does:
```python
z_hat = self.forward(z_history, action.expand(-1, z_history.shape[1], -1))
```
This replicates the single current action `a_t` across all N history positions. In a 3-frame history window, positions t-2 and t-1 are conditioned on `a_t` instead of on their own actions `a_{t-2}` and `a_{t-1}`. This is a known design limitation of the autoregressive rollout path, not a training-path error; the training path correctly receives `(B, T, A)` actions aligned per-step. The severity is **minor** — it only affects the rollout/planning path and the paper does not specify how to handle multi-step history action alignment during rollout.

---

### Fix 3: Gradient Clipping Added in Trainer
**Status: CORRECTLY APPLIED**

`trainer.py` lines 77-80:
```python
torch.nn.utils.clip_grad_norm_(
    self.model.parameters(),
    max_norm=self.config.training.max_grad_norm,
)
```

`TrainingConfig` now includes `max_grad_norm: float = 1.0`, which is a standard and appropriate value for transformer training. The clip is correctly placed after `loss.backward()` and before `optimizer.step()`.

---

### Fix 4: CEM Planning Restores Model Training Mode After Completion
**Status: CORRECTLY APPLIED**

`cem.py` lines 47-107 now use the correct try/finally pattern:
```python
was_training = self.model.training
self.model.eval()
try:
    # ... CEM logic ...
    return mu
finally:
    self.model.train(was_training)
```

The model is guaranteed to be restored to its original mode even if an exception is raised during planning.

---

## 4. Paper Alignment Score: 8 / 10

**Previous score: 7 / 10**

The per-position action conditioning fix elevates the score. The remaining half-point deduction is for the unresolved `model_name` string.

**What matches (unchanged from previous review):**
- SIGReg Epps-Pulley statistic with vectorized `|phi_N|^2 = mean_cos^2 + mean_sin^2` expansion
- BatchNorm1d in projection heads (not LayerNorm)
- AdaLN with zero initialization of both weight and bias
- Teacher-forcing training (no `.detach()` on encoder embeddings)
- No EMA, no stop-gradient
- MSE + 0.1 * SIGReg loss, applied step-wise per timestep
- CEM: 300 samples, 30 iterations, 30 elites, H=5
- Predictor: 6 layers, 16 heads, 0.1 dropout, causal mask
- Encoder: 12 layers, 3 heads, dim=192
- Gradient clipping (new, now present)
- Per-step action conditioning (new, now correct)

**What deviates:**
1. `model_name = "vit_tiny_patch16_224"` with `patch_size = 14` override — string not updated — **major**
2. `predict_step` uses a single action replicated across all history positions — **minor**

---

## 5. Security Findings

Full security report: `/Users/subham/Desktop/codes/agentsclaude/output/20260326_114622_build-the-jepa-model-from-scratch-in-python/reviews/security.md`

No new security issues were introduced by the fixes. The findings from the previous review remain unchanged:

| Severity | Count | Summary |
|----------|-------|---------|
| Critical | 0 | None |
| High | 0 | No checkpoint save/load present; prior advisory stands if added later |
| Medium | 1 | No input validation on public API surface |
| Low | 2 | `make_synthetic_dataset` still uses unseeded global numpy RNG; no `.gitignore` |

No hardcoded secrets, credentials, injection vectors, or unsafe deserialization found.

---

## 6. File-by-File Review

### `lewm/config.py`
**Issues:**
- **(major)** Line 9: `model_name: str = "vit_tiny_patch16_224"` still references patch16 in the string. The `patch_size = 14` integer was correctly updated but the model_name was not. These two must be consistent. See Fix 1 above.

---

### `lewm/models/encoder.py`
**Issues:**
- **(nit)** Docstring correctly updated to say `patch=14`. No issues in the code itself; the bug is upstream in `config.py`.

---

### `lewm/models/adaln.py`
**Issues:** None. The `forward` method correctly handles both `(B, D_cond)` and `(B, S, D_cond)` conditioning shapes via a `dim` check, and the zero-initialization is verified in both the `__main__` block and test suite.

---

### `lewm/models/predictor.py`
**Issues:**
- **(minor)** `predict_step` at line 171 replicates the current action across all history positions:
  ```python
  z_hat = self.forward(z_history, action.expand(-1, z_history.shape[1], -1))
  ```
  Only the last position's prediction is used (`z_hat[:, -1, :]`), so the effect on final output is limited. However, the history positions t-2 and t-1 inside the attention window will be conditioned on the wrong action `a_t` instead of their own historical actions. The training forward path is unaffected and correct.
- **(nit)** The causal mask test in `test_predictor.py::test_causal_mask_attention_only` was not updated; it still uses `torch.ones` actions. Now that per-step conditioning is implemented, this test is no longer needed as a workaround and should use varied actions to test the real conditioning behavior.

---

### `lewm/models/projection_head.py`
**Issues:** None. Unchanged and correct.

---

### `lewm/models/lewm.py`
**Issues:**
- **(minor)** `rollout` and `rollout_from_embedding` share identical action-expansion logic (`a_t.expand(-1, history.shape[1], -1)`) and differ only in whether the initial frame is encoded. Consider extracting `_autoregressive_step` to a shared helper to reduce duplication. This is a code quality concern, not a correctness issue.
- **(nit)** `rollout` is decorated with `@torch.no_grad()` but does not explicitly set `self.eval()`. BatchNorm in the encoder projection head will behave differently in train vs. eval mode. The caller is expected to call `model.eval()` first (as done in `main.py`), but this contract is not enforced or documented in the method signature.

---

### `lewm/losses/sigreg.py`
**Issues:** Unchanged from previous review.
- **(nit)** `compute_sigreg_stepwise` initializes the accumulator with `torch.tensor(0.0, ...)`. This creates a scalar with no attached computation graph until the first addition. In practice `T >= 1` always holds, so this is safe.

---

### `lewm/losses/prediction_loss.py`
**Issues:** None. Unchanged and correct.

---

### `lewm/training/trainer.py`
**Issues:**
- **(minor)** No checkpoint saving. A trained model cannot be persisted across runs. Even a basic `torch.save(model.state_dict(), path)` at epoch end would be needed for practical use.
- **(nit)** `_check_collapse` calls `next(iter(dataloader))` which re-instantiates the DataLoader iterator on every epoch. For large datasets this is wasteful. A single held-out validation batch would be better.

---

### `lewm/planning/cem.py`
**Issues:** None. The try/finally mode-restoration pattern is correctly implemented. Sigma is still clamped only from below (`min=1e-4`) with no upper bound — this is a pre-existing nit that was not introduced by the fix.

---

### `lewm/planning/rollout.py`
**Issues:**
- **(nit)** `batch_latent_rollout` is a one-line wrapper around `latent_rollout` with no added behavior. The naming implies separate batch handling but it does not add any. This can confuse readers. Unchanged from previous review.

---

### `lewm/data/dataset.py`
**Issues:**
- **(medium)** `make_synthetic_dataset` still uses the global numpy random state without a seed. Tests that depend on this function will not be reproducible across platforms or numpy versions. The seed parameter proposed in the previous review was not added.
- **(nit)** `frame_indices` off-by-one analysis is correct (as noted previously); the computation is safe.

---

### `lewm/diagnostics/decoder.py`
**Issues:**
- **(nit)** `patch_size: int = 16` default — still inconsistent with the encoder's patch-14 after the partial fix. Not in the training path, cosmetic only.

---

### `lewm/data/transforms.py`
**Issues:** None. Unchanged and correct.

---

### `lewm/training/logger.py`
**Issues:** None. Unchanged and correct.

---

### `main.py`
**Issues:**
- **(nit)** `config.sigreg.num_projections = 128` is set for CPU speed but is not commented as a demo shortcut. A reader might copy this value for real training, where 1024 is specified by the paper.

---

### `tests/test_sigreg.py`
**Issues:** Unchanged from previous review.
- **(minor)** `test_gaussian_input_low_loss` threshold of `< 0.05` is empirically reasonable but may be brittle across platforms.

---

### `tests/test_predictor.py`
**Issues:**
- **(minor)** `test_causal_mask_attention_only` still uses `torch.ones` for actions as a workaround. Now that per-step conditioning is implemented, the test should use randomly varied actions to actually test the new conditioning. The test passes for the wrong reason — it was testing the old mean-pool approximation, not per-position correctness.
- **(nit)** `test_adaln_zero_init` was correctly added and tests both 2D and 3D conditioning inputs. Well done.

---

### `tests/test_encoder.py`
**Issues:** None. Unchanged and correct.

---

### `tests/test_integration.py`
**Issues:**
- **(minor)** `test_single_training_step` assertion `loss2_val < loss1_val * 2` is too permissive and unchanged from the previous review.

---

## 7. Specific Fix Suggestions

### Fix 1 (Remaining): Correct Encoder Model Name String

**File**: `/Users/subham/Desktop/codes/agentsclaude/output/20260326_114622_build-the-jepa-model-from-scratch-in-python/code/lewm/config.py`

The integer `patch_size` was correctly changed to 14. Now update the string:

```python
@dataclass
class EncoderConfig:
    """ViT-Tiny encoder configuration."""
    model_name: str = "vit_tiny_patch14_224"  # was "vit_tiny_patch16_224"
    patch_size: int = 14
```

Before deploying, confirm the exact timm name:
```python
import timm
print(timm.list_models("vit_tiny*patch14*"))
```

Common candidates are `"vit_tiny_patch14_224"` or `"vit_tiny_patch14_reg4_dinov2"`. Use the base variant without DINOv2 pretraining since the paper trains from scratch.

---

### Fix 2 (Recommended): Update Causal Mask Test to Use Per-Step Actions

**File**: `/Users/subham/Desktop/codes/agentsclaude/output/20260326_114622_build-the-jepa-model-from-scratch-in-python/code/tests/test_predictor.py`

```python
def test_causal_mask_attention_only(self, predictor: CausalPredictor) -> None:
    """Changing future latent inputs should not affect past outputs."""
    B, T, D, A = 4, 4, 192, 2
    torch.manual_seed(42)
    z = torch.randn(B, T, D)
    actions = torch.randn(B, T, A)  # varied per-step actions

    predictor.eval()
    with torch.no_grad():
        z_hat1 = predictor(z, actions)

        z2 = z.clone()
        z2[:, -1, :] = torch.randn(B, D)
        # Keep actions identical to isolate the latent causality
        z_hat2 = predictor(z2, actions)

    diff = (z_hat1[:, 0, :] - z_hat2[:, 0, :]).abs().max().item()
    assert diff < 1e-5, f"Causal mask failed: diff at pos 0 = {diff}"
```

---

### Fix 3 (Recommended): Seed Synthetic Dataset Generator

**File**: `/Users/subham/Desktop/codes/agentsclaude/output/20260326_114622_build-the-jepa-model-from-scratch-in-python/code/lewm/data/dataset.py`

```python
def make_synthetic_dataset(
    num_trajectories: int = 10,
    traj_length: int = 50,
    img_size: int = 64,
    action_dim: int = 2,
    seed: int | None = None,
) -> list[dict]:
    """Generate random trajectories for testing only. Not for production use."""
    rng = np.random.default_rng(seed)
    trajectories = []
    for _ in range(num_trajectories):
        trajectories.append({
            "observations": rng.integers(
                0, 255, (traj_length, img_size, img_size, 3), dtype=np.uint8
            ),
            "actions": rng.standard_normal((traj_length, action_dim)).astype(np.float32),
        })
    return trajectories
```

---

## 8. What Was Done Well

All positives from the previous review still apply:

- **SIGReg vectorization**: The `|phi_N|^2 = mean_cos^2 + mean_sin^2` identity avoids the O(N^2) pairwise sum in the paper's pseudocode with a correct and memory-efficient O(B*M*K) implementation.
- **AdaLN zero-initialization**: Correctly zeroed for both weight and bias. Now extended to support per-position `(B, S, D_cond)` conditioning, which is required for the corrected predictor.
- **Gradient clipping**: The fix was cleanly integrated into both `TrainingConfig` (as `max_grad_norm: float = 1.0`) and the training loop itself.
- **CEM mode restoration**: The try/finally pattern is the correct approach — it handles exceptions during planning without leaving the model stuck in eval mode.
- **Per-step action conditioning**: The AdaLN now correctly receives `(B, T, D)` per-position action embeddings, matching the paper's intent that each sequence position is conditioned on its own action.
- **Configuration hygiene**: All hyperparameters remain centralized in typed dataclasses. The addition of `max_grad_norm` to `TrainingConfig` follows the existing convention correctly.
- **Test coverage**: The addition of `test_adaln_zero_init` that tests both 2D and 3D conditioning inputs provides targeted verification of the per-step conditioning fix.
- **No EMA, no stop-gradient**: Correctly absent throughout. The `z` tensor from the encoder flows directly into both the predictor input and the SIGReg loss without any `.detach()`.
