# Configuration Issues Found - copilot/fix-30

## Overview
Issues identified in the `copilot/fix-30` branch that need to be resolved before merging.

## ❌ **Issues Found**

### 1. **Test Patching Error** - CRITICAL
- **File**: `backend/tests/test_config_paths_simple.py:115`
- **Issue**: Test patches `shared.utils.get_settings` but function is in `shared.config`
- **Fix**: Change patch target to `shared.config.get_settings`
### 2. **pyproject.toml README Path Error** - CRITICAL
- **File**: `backend/pyproject.toml`
- **Issue**: CI build failing due to invalid README path reference
- **Error**: `Cannot access '/home/runner/work/docrag/docrag/backend/../README.md' (or anything outside '/home/runner/work/docrag/docrag/backend')`
- **Root Cause**: pyproject.toml has `readme = "../README.md"` which tries to access parent directory
- **Impact**: CI builds failing, package installation broken
- **Fix**: Change to `readme = "README.md"` or create a README.md in backend/ directory

### 3. **Missing CI Validation in Development Workflow** - MEDIUM
- **File**: `.github/copilot-instructions.md`
- **Issue**: GitHub Copilot instructions don't require CI validation before PR creation
- **Impact**: PRs being created with failing CI builds, wasting review time
- **Root Cause**: No explicit requirement for CI checks in development workflow
- **Fix**: Add mandatory CI validation step to copilot instructions

## 📋 **Priority Fixes Required**

1. **HIGH PRIORITY**: Fix pyproject.toml README path to resolve CI build failures
2. **MEDIUM PRIORITY**: Fix test patch target to make all tests pass
3. **MEDIUM PRIORITY**: Update copilot instructions to require CI validation

## 🔧 **Required Fixes**

### Fix 1: README Path in pyproject.toml
**File**: `backend/pyproject.toml`
**Current**: `readme = "../README.md"`
**Fix**: `readme = "README.md"` (and copy/create README.md in backend/)

### Fix 2: Test Patch Target
**File**: `backend/tests/test_config_paths_simple.py`
**Line 115**: Change from:
```python
with patch('shared.utils.get_settings', return_value=custom_settings):
```
To:
### Fix 3: Add CI Validation to Copilot Instructions
**File**: `.github/copilot-instructions.md`
**Addition**: Add new section requiring CI validation:
```markdown
## CI Validation Requirements
- **ALWAYS run and validate CI checks before creating pull requests**
- Check GitHub Actions status on branch before submitting PR
- Fix any failing tests, linting errors, or build issues
- Only create PRs with green CI status
- Include CI status verification in PR description
```
