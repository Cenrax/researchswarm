---
description: Generate unit tests for implemented modules to verify correctness of individual components. Use this after writing code to create a test suite.
---

# Test Generation

Write pytest-compatible unit tests for each implemented module.

## Test Structure

Create `output/code/tests/` directory with:

```
output/code/tests/
├── __init__.py
├── conftest.py          # Shared fixtures
├── test_<module1>.py
├── test_<module2>.py
└── ...
```

## For Each Module, Test:

### 1. Input/Output Shapes
```python
def test_module_output_shape():
    model = Module(config)
    x = torch.randn(batch_size, seq_len, d_model)
    out = model(x)
    assert out.shape == (batch_size, seq_len, d_output)
```

### 2. Numerical Correctness
- Known input → expected output (from paper examples if available)
- Edge cases: zero input, single element, max values

### 3. Gradient Flow
```python
def test_gradients_flow():
    model = Module(config)
    x = torch.randn(2, 10, 64, requires_grad=True)
    out = model(x)
    loss = out.sum()
    loss.backward()
    assert x.grad is not None
    assert not torch.isnan(x.grad).any()
```

### 4. Configuration
- Default config produces valid model
- Invalid config raises appropriate errors

### 5. Determinism
```python
def test_deterministic():
    torch.manual_seed(42)
    out1 = model(x)
    torch.manual_seed(42)
    out2 = model(x)
    assert torch.allclose(out1, out2)
```

## Fixtures in conftest.py
```python
@pytest.fixture
def sample_config():
    return {"d_model": 64, "n_heads": 4, "n_layers": 2}

@pytest.fixture
def sample_input():
    return torch.randn(2, 10, 64)
```

## Rules
- Every public function/class gets at least one test.
- Use small dimensions (d_model=64, seq_len=10) for speed.
- Tests must run in <10 seconds total on CPU.
- Run the test suite after writing: `python -m pytest output/code/tests/ -v`
