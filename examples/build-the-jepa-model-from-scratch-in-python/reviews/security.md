# Security Audit: LeWorldModel (LeWM) JEPA

**Code directory**: `output/.../code/`
**Audited**: 2026-03-26

---

## Summary

- Critical: 0
- High: 1
- Medium: 2
- Low: 1

---

## Findings

### [HIGH] Unsafe `torch.load` pattern not present but `torch.save` is absent; no checkpoint save/load implemented
**File**: `lewm/training/trainer.py` (entire file)
**Issue**: The trainer has no checkpoint save or load functionality. When a user inevitably adds one, `torch.load()` without `weights_only=True` (the default prior to PyTorch 2.6) allows arbitrary code execution via crafted `.pt` files. The absence of checkpoint code today means the pattern has not been secured against future addition.
**Fix**: If checkpoint saving is added, always use:
```python
# Saving
torch.save(model.state_dict(), path)  # save state_dict only, never the full model

# Loading
state_dict = torch.load(path, map_location=device, weights_only=True)
model.load_state_dict(state_dict)
```
**Risk**: Currently no active exploit surface; risk is HIGH for the inevitable checkpoint feature.

---

### [MEDIUM] No input shape or dtype validation on public API methods
**Files**: `lewm/models/encoder.py:34`, `lewm/models/predictor.py:127`, `lewm/planning/cem.py:36`
**Issue**: Public-facing methods (`ViTEncoder.forward`, `CausalPredictor.forward`, `CEMPlanner.plan`) accept tensors without validating shape, dtype, or value range. Malformed inputs will produce cryptic internal errors deep in PyTorch or timm rather than clear failure messages.
**Example in `encoder.py`**:
```python
def forward(self, obs: torch.Tensor) -> torch.Tensor:
    # No validation: what if obs.shape == (B, 1, 224, 224) or dtype is int?
    cls_token = self.vit(obs)
```
**Fix**: Add lightweight guards at entry points:
```python
def forward(self, obs: torch.Tensor) -> torch.Tensor:
    if obs.ndim != 4 or obs.shape[1] != 3:
        raise ValueError(f"Expected (B, 3, H, W), got {obs.shape}")
    if obs.dtype not in (torch.float32, torch.float16):
        raise TypeError(f"Expected float tensor, got {obs.dtype}")
    ...
```
**Risk**: Unexpected behavior or silent errors when integrated into larger systems; not directly exploitable.

---

### [MEDIUM] `make_synthetic_dataset` uses `np.random.randint` without a seed
**File**: `lewm/data/dataset.py:98`
**Issue**: `make_synthetic_dataset` is used in tests and the `main.py` demo. Without a fixed seed, results are non-reproducible, making CI tests reliant on it non-deterministic. More importantly, if a user mistakenly passes `make_synthetic_dataset` output as real data, there is no guard. This is a code quality issue with reproducibility and data safety implications.
**Fix**: Accept an optional seed parameter and document clearly that this is for testing only:
```python
def make_synthetic_dataset(
    num_trajectories: int = 10,
    traj_length: int = 50,
    img_size: int = 64,
    action_dim: int = 2,
    seed: int | None = None,
) -> list[dict]:
    rng = np.random.default_rng(seed)
    ...
    "observations": rng.integers(0, 255, (traj_length, img_size, img_size, 3), dtype=np.uint8),
```
**Risk**: Non-reproducible tests; no direct security vulnerability.

---

### [LOW] No `.gitignore` to prevent accidental commit of model weights or data
**File**: Repository root / code directory
**Issue**: There is no `.gitignore` file in the code directory. If a user trains the model and saves checkpoints (e.g., `checkpoint.pt`, `best_model.pt`) alongside the code, those binary files could be accidentally committed, potentially leaking training data characteristics or large binary blobs.
**Fix**: Add a `.gitignore`:
```
# Model checkpoints
*.pt
*.pth
*.ckpt

# Data
data/
*.npy
*.npz

# Python cache
__pycache__/
*.pyc
.pytest_cache/
```
**Risk**: Accidental secret exposure if checkpoints contain embedded data; large repo size.

---

## Checklist

### Secrets & Credentials
- [x] No hardcoded API keys, tokens, or passwords
- [x] No credentials in comments or docstrings
- [x] No `.env` file usage (not needed for this codebase)

### Input Validation
- [x] No path traversal vulnerabilities (no user-supplied file paths)
- [ ] Model inputs not validated for shape and dtype (see MEDIUM finding above)
- [x] No `eval()` or `exec()` on any input

### File Operations
- [x] No file I/O in the main model code (no open/write operations)
- [x] No user-controlled file paths

### Dependencies
- [x] All dependencies from reputable sources (torch, torchvision, timm, numpy, pytest)
- [x] No known CVEs in specified minimum versions
- [x] No unnecessary network access in model code
- [ ] No `torch.load` calls present (but no save/load implemented either -- see HIGH finding)

### Code Execution
- [x] No subprocess calls
- [x] No shell injection vectors
- [x] No arbitrary code download

### Data Handling
- [x] No PII in logs or outputs
- [x] No model weight API exposure
