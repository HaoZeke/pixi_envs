# CI Testing Notes

## Critical: Environment Variables for UV/PIP

### Problem

`PIP_EXTRA_INDEX_URL=https://download.pytorch.org/whl/cpu` in the environment causes `uv sync` to fail.

### Solution

Always unset extra index URLs before running `uv sync` in CI:

```bash
env -u PIP_EXTRA_INDEX_URL -u UV_EXTRA_INDEX_URL -u UV_PIP_EXTRA_INDEX_URL uv sync --all-extras
```

### Test Results

- chemparseplot: 72 passed
- pychum: 26 passed (after adding rgpycrumbs dependency)
