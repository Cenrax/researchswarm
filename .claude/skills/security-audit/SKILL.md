---
description: Perform a security audit on generated code checking for vulnerabilities, unsafe practices, and exposed secrets. Use this during code review to ensure the implementation is safe for production.
---

# Security Audit

Check generated code for security vulnerabilities and unsafe practices.

## Audit Checklist

### 1. Secrets & Credentials
- [ ] No hardcoded API keys, tokens, or passwords
- [ ] No credentials in comments or docstrings
- [ ] Sensitive values loaded from environment variables
- [ ] `.env` files listed in `.gitignore`

### 2. Input Validation
- [ ] File paths validated (no path traversal: `../`)
- [ ] User inputs sanitized before use
- [ ] Model inputs validated for shape and dtype
- [ ] No `eval()` or `exec()` on untrusted input

### 3. File Operations
- [ ] Files opened with context managers (`with open(...)`)
- [ ] Write operations use safe paths (no user-controlled paths)
- [ ] Temporary files cleaned up
- [ ] Permissions not overly broad

### 4. Dependencies
- [ ] All dependencies from reputable sources
- [ ] No known CVEs in pinned versions
- [ ] No unnecessary network access
- [ ] Pickle/deserialization uses safe loading

### 5. Code Execution
- [ ] No shell injection in subprocess calls
- [ ] Subprocess calls use list form, not string
- [ ] No arbitrary code download and execution
- [ ] Timeout set on network/subprocess calls

### 6. Data Handling
- [ ] No PII in logs or outputs
- [ ] Model weights not exposed via API without auth
- [ ] Safe tensor serialization (prefer safetensors over pickle)

## Severity Levels
- **CRITICAL**: Exploitable vulnerability (e.g., code injection, exposed secrets)
- **HIGH**: Unsafe practice that could lead to exploitation
- **MEDIUM**: Best practice violation with limited risk
- **LOW**: Code quality issue with security implications

## Output

Write findings to `output/reviews/security.md`:

```markdown
# Security Audit

## Summary
- Critical: 0
- High: 1
- Medium: 2
- Low: 1

## Findings

### [HIGH] Unsafe pickle loading in model.py:45
**Issue**: `torch.load()` without `weights_only=True`
**Fix**: `torch.load(path, weights_only=True)`
**Risk**: Arbitrary code execution via crafted model file
```

## Rules
- Every finding must include a fix suggestion.
- CRITICAL findings must be fixed before code ships.
- Check EVERY file, not just main modules.
