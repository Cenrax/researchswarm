---
description: Execute code in an isolated sandbox to verify correctness. Use this whenever you write or modify code to ensure it runs without errors before moving on.
---

# Sandbox Code Execution

Run code safely and verify outputs match expected behavior.

## Execution Protocol

### For each code file:

1. **Install dependencies first**:
   ```bash
   pip install -q <packages>
   ```

2. **Run the module independently**:
   ```bash
   python output/code/<file>.py
   ```

3. **Capture and verify output**:
   - Exit code must be 0
   - No unhandled exceptions
   - Print key values for verification:
     - Tensor shapes: `print(f"Shape: {tensor.shape}")`
     - Sample outputs: `print(f"Output sample: {output[:5]}")`
     - Timing: `print(f"Execution time: {elapsed:.2f}s")`

4. **Handle failures**:
   - Read the full traceback
   - Identify root cause (import error, shape mismatch, etc.)
   - Fix the code
   - Re-run (max 3 retries per file)

### For integration testing:

1. Run `output/code/main.py` after all modules pass individually.
2. Verify end-to-end data flow.
3. Check memory usage stays reasonable:
   ```bash
   python -c "import tracemalloc; tracemalloc.start(); exec(open('output/code/main.py').read()); print(f'Peak memory: {tracemalloc.get_traced_memory()[1]/1e6:.1f}MB')"
   ```

## GPU Fallback
If CUDA is unavailable, ensure all code runs on CPU:
```python
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
```

## Rules
- NEVER skip execution — every file must be tested.
- Use synthetic/small data — do not download large datasets.
- Print shapes and sample values, not full tensors.
- If a file takes >60 seconds, flag it as a performance concern.
