# Dependencies

## Required

| Package | Version | Purpose | Paper Reference |
|---------|---------|---------|-----------------|
| torch | >=2.1 | Tensor operations, autograd, model building, training loop | All sections - core framework |
| torchvision | >=0.16 | Image transforms (Resize, Normalize, ToTensor) for 224x224 preprocessing | Encoder input pipeline |
| timm | >=0.9.12 | ViT-Tiny pretrained/scratch model (`vit_tiny_patch16_224`); patch embedding, transformer blocks | LeWM Section 4: Encoder architecture |
| numpy | >=1.24 | Array manipulation, dataset handling | Data preprocessing |

## Optional

| Package | Version | Purpose |
|---------|---------|---------|
| tensorboard | >=2.14 | Training visualization (loss curves, SIGReg values, embedding histograms) |
| matplotlib | >=3.7 | Plotting decoded images, latent space visualizations, trajectory plots |
| tqdm | >=4.65 | Progress bars for training and CEM planning loops |
| gymnasium | >=0.29 | Environment interface for online evaluation (PushT, TwoRoom) |
| h5py | >=3.9 | Loading offline trajectory datasets stored in HDF5 format |
| einops | >=0.7 | Readable tensor reshaping (optional, can use native torch) |
| pytest | >=7.4 | Unit and integration testing |

## Notes on Key Choices

### Why `timm` over HuggingFace `transformers`
- `timm` provides direct access to ViT-Tiny with clean separation of patch embedding, transformer blocks, and classification head.
- Easier to extract `[CLS]` token and replace the head with a custom `Linear + BatchNorm1d` projection.
- Smaller install footprint than `transformers` + `accelerate`.
- The paper mentions "HuggingFace library" but `timm` ViT-Tiny is architecturally identical and more Pythonic for custom modifications.

### Why not `flash-attn`
- The predictor uses causal masking which `flash-attn` supports, but adding it as a hard dependency complicates installation.
- PyTorch 2.1+ includes `torch.nn.functional.scaled_dot_product_attention` with built-in flash attention support via `attn_mask` parameter.
- Recommended: use PyTorch native SDPA. Optional: install `flash-attn` for speed on supported GPUs.

### `torch` size warning
- PyTorch with CUDA is ~2.5 GB. CPU-only variant is ~200 MB.
- For development/testing, CPU-only is sufficient. For training, CUDA is required.

## Conflicts
- None identified between listed packages.
- `timm` depends on `torch` and `torchvision`, so versions must be compatible.

## Install Commands

```bash
# Core (required)
pip install torch>=2.1 torchvision>=0.16 timm>=0.9.12 numpy>=1.24

# Development (optional)
pip install tensorboard>=2.14 matplotlib>=3.7 tqdm>=4.65 pytest>=7.4

# Environment interaction (optional, for online evaluation)
pip install gymnasium>=0.29 h5py>=3.9

# Full install
pip install torch>=2.1 torchvision>=0.16 timm>=0.9.12 numpy>=1.24 tensorboard>=2.14 matplotlib>=3.7 tqdm>=4.65 pytest>=7.4 gymnasium>=0.29 h5py>=3.9
```
