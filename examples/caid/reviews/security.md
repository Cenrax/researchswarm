# Security Audit — CAID Framework (Post-Fix Re-Review)

**Audited**: `output/20260412_004603_build-the-caid-framework-in-python/code/`
**Date**: 2026-04-12
**Review type**: Post-fix re-review

---

## Summary

| Severity | Count |
|---|---|
| CRITICAL | 0 |
| HIGH | 0 |
| MEDIUM | 1 |
| LOW | 2 |

All previously reported HIGH and MEDIUM severity issues have been resolved. No new issues were introduced.

---

## Resolved Findings (Verified Fixed)

### [FIXED - was HIGH] Path traversal in engineer file-write operations

**Previous issue**: LLM-controlled `# filename:` headers could write outside the worktree.

**Verification**: `engineer.py:55-67` now contains `_safe_resolve_path()` which calls `.resolve()` and checks that the resolved path starts with `str(wt_base) + "/"`. This is used at lines 269, 281, and 299 for all file writes. Test coverage added in `tests/test_engineer.py:41-93` covering `../` traversal, absolute paths, mid-path `..`, and symlink escape attempts.

**Status**: FIXED and tested.

---

### [FIXED - was MEDIUM] `FileNotFoundError` returns `passed=True`

**Previous issue**: Missing pytest silently reported test success.

**Verification**: `engineer.py:332-333` now returns `{"passed": False, "output": "pytest not found. Install: pip install pytest"}`.

**Status**: FIXED.

---

### [FIXED - was MEDIUM] Subprocess uses system `python` instead of `sys.executable`

**Previous issue**: `["python", "-m", "pytest", ...]` could use wrong interpreter.

**Verification**: `engineer.py:314` now uses `[sys.executable, "-m", "pytest", "-x", "--tb=short", "-q"]`. `sys` is imported at line 13.

**Status**: FIXED.

---

### [FIXED - was LOW] `gitpython` listed as unused dependency

**Previous issue**: Unused dependency with historical CVEs.

**Verification**: `requirements.txt` no longer contains `gitpython`. `pyproject.toml` dependencies section also does not include it.

**Status**: FIXED.

---

### [FIXED - was LOW] `pyproject.toml` used deprecated build backend

**Previous issue**: `setuptools.backends._legacy:_Backend` is deprecated.

**Verification**: `pyproject.toml:3` now reads `build-backend = "setuptools.build_meta"`.

**Status**: FIXED.

---

### [FIXED - was MINOR] `assert` used for control flow in `manager.py`

**Previous issue**: `assert self.graph is not None` stripped by `-O` flag.

**Verification**: `manager.py:161-162`, `241-242`, `343-344` all use `if self.graph is None: raise RuntimeError("Graph not initialized")`.

**Status**: FIXED.

---

### [FIXED - was MINOR] Branch names not sanitized

**Previous issue**: Task IDs with `/` and `::` would produce invalid git branch names.

**Verification**: `engineer.py:85-86` now sanitizes: `safe_task_id = task.task_id.replace("/", "-").replace("::", "-").replace(".", "-")`.

**Status**: FIXED.

---

## Remaining Findings

### [MEDIUM] API key can be stored in plaintext YAML — `config.py:69-70`, `caid.yaml.example`

**Issue**: `LLMConfig.api_key` accepts a value directly from the YAML file. A user who puts `api_key: sk-...` in `caid.yaml` and commits it would leak the secret. The code does not warn when a key is loaded from YAML rather than from an environment variable.

**Fix**: Add a warning when the API key comes from YAML:
```python
# In from_yaml(), after processing llm_data:
if llm_data.get("api_key") and not os.environ.get("CAID_LLM_API_KEY"):
    import warnings
    warnings.warn(
        "API key loaded from YAML file. "
        "Prefer CAID_LLM_API_KEY environment variable to avoid accidental secret commits.",
        UserWarning,
        stacklevel=2,
    )
```

**Mitigating factors**: `caid.yaml` is already excluded in `.gitignore` (line 1-2). The `caid.yaml.example` has the `api_key` line commented out with a note recommending the env variable. This is LOW risk in practice.

**Risk**: Accidental secret exposure via version control if a user copies the example and adds their key without checking `.gitignore`.

---

### [LOW] `worktree_base_dir` defaults to world-readable `/tmp/caid-worktrees` — `config.py:58`

**Issue**: On shared systems, `/tmp` is world-readable. Worktree directories under `/tmp/caid-worktrees` contain full repository source code, LLM responses, and partial implementations. Another local user could read these during a run.

**Fix**: Use a mode-0700 private temp directory:
```python
import tempfile
worktree_base_dir: Path = Field(
    default_factory=lambda: Path(tempfile.mkdtemp(prefix="caid-worktrees-"))
)
```
Or explicitly set restrictive permissions:
```python
self.config.worktree_base_dir.mkdir(parents=True, exist_ok=True, mode=0o700)
```

**Risk**: Low in single-user environments (most development machines). Moderate on shared compute (HPC clusters, shared CI runners).

---

### [LOW] Two `read_text()` calls without encoding in `engineer.py` — lines 184 and 228

**Issue**:
- `engineer.py:184`: `conflict_contents[f] = fpath.read_text()` — no encoding specified.
- `engineer.py:228`: `existing = target.read_text()` — no encoding specified.

Both rely on the system default locale encoding. On systems where `locale.getpreferredencoding()` is not UTF-8, reading files with UTF-8 content (e.g., docstrings with non-ASCII characters) will raise `UnicodeDecodeError`, causing the engineer to fail silently.

**Fix**:
```python
conflict_contents[f] = fpath.read_text(encoding="utf-8", errors="replace")
existing = target.read_text(encoding="utf-8", errors="replace")
```

**Risk**: Low — operational robustness issue on non-UTF-8 systems. Not exploitable, but can cause silent engineer crashes.

---

## Full Checklist Results

### 1. Secrets and Credentials
- [x] No hardcoded API keys or tokens in source code
- [x] Credentials loaded from environment variables (`CAID_LLM_API_KEY`)
- [x] `caid.yaml` listed in `.gitignore` (line 1-2 of `.gitignore`)
- [x] `.env` files listed in `.gitignore`
- [x] No credentials in comments or docstrings
- [ ] No warning emitted when API key loaded from YAML (MEDIUM finding above)

### 2. Input Validation
- [x] File paths from LLM validated by `_safe_resolve_path()` — path traversal prevented
- [x] YAML parsed with `yaml.safe_load()` (not `yaml.load()`)
- [x] No `eval()` or `exec()` on user or LLM input
- [x] No `shell=True` subprocess calls anywhere
- [x] Pydantic models validate all JSON schemas with type enforcement

### 3. File Operations
- [x] YAML file opened with `with open(path)` context manager (`config.py:69`)
- [x] All worktree writes go through `_safe_resolve_path()` guard
- [x] Temporary directories cleaned up in test fixtures (`conftest.py:70`, `test_engineer.py:37`)
- [ ] Two `read_text()` calls lack explicit encoding (LOW finding above)

### 4. Dependencies
- [x] All dependencies from reputable PyPI sources
- [x] `gitpython` removed — no longer an unnecessary dependency
- [x] No pickle or unsafe deserialization anywhere in codebase
- [x] `pyyaml` uses `safe_load` throughout
- [x] No pinned versions with known CVEs (minimum-version pinning used)

### 5. Code Execution
- [x] No shell injection — all subprocess calls use list form
- [x] All git subprocess calls have 120s timeout (`git_ops.py:68`)
- [x] pytest subprocess has 300s timeout (`engineer.py:324`)
- [x] `sys.executable` used for pytest (not bare `"python"`)
- [x] No dynamic code download or remote execution
- [x] No `os.system()` calls

### 6. Data Handling
- [x] No PII in log statements — only task IDs, engineer IDs, file paths
- [x] No model weights — CAID is a coordination framework, not an ML model
- [x] Token usage tracked internally but not written to disk or logged at high verbosity
- [x] No unsafe serialization formats (JSON and YAML only, via Pydantic and `safe_load`)
