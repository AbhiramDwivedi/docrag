# VS Code Copilot Instructions

## Base Instructions
This file extends the repository's GitHub Copilot instructions. **Always follow the guidelines in `.github/copilot-instructions.md` first**, then apply the VS Code-specific additions below.

**Reference**: See `.github/copilot-instructions.md` for:
- General coding guidelines (Python ≥ 3.11, type hints, Black formatting)
- Backend structure and import paths (`backend.src.*`)
- Testing requirements (`backend/tests/` only)
- Demo file placement (`backend/examples/` only)
- Architecture documentation requirements
- Dependency management and TODOs

---

## VS Code Specific Guidelines

### **Environment Management**
1. **Virtual Environment Required**
   - **Always activate virtual environment** before running Python commands: `.venv\Scripts\activate`
   - Never run `python`, `pytest`, or `pip` commands without activating venv first
   - Use `python -m pytest` instead of direct `pytest` command
   - Working directory should be `c:\sw\code\docrag\backend` for most operations

### **File Operations**
2. **Temporary Files for Testing**
   - **For any code/script longer than 2-3 lines**, create a temporary file first
   - Pattern: `temp_<purpose>.py` (e.g., `temp_ci_validation.py`, `temp_import_test.py`)
   - **Always remove temporary files** after use: `Remove-Item temp_*.py -Force`
   - Use temporary files for:
     - Testing import structures
     - Validating CI compatibility
     - Running complex validation scripts
     - Testing multi-line command sequences

### **Terminal Operations**
3. **Windows PowerShell Considerations**
   - Use `;` to join commands on single line if needed
   - Use `cd "c:\sw\code\docrag\backend" ; command` syntax for directory changes
   - Handle Windows file permission errors gracefully (SQLite database locks are common)
   - Use `Remove-Item` instead of `rm` for file deletion

### **Testing Strategy**
4. **Comprehensive Test Validation**
   - Run specific test suites before full test runs: `python -m pytest tests/test_specific.py -v`
   - Use virtual environment for all pytest commands
   - Create temporary validation scripts for complex test scenarios
   - Handle Windows-specific test teardown issues (file locking) as expected behavior

### **Code Analysis**
5. **Large File Handling**
   - **When analyzing files >50 lines or running complex validation**, use temporary files
   - Break large validations into smaller, focused temporary scripts
   - Use `read_file` with appropriate line ranges instead of reading entire large files
   - Create focused validation scripts rather than running large inline commands

### **Git Operations**
6. **Branch and Commit Management**
   - Stage changes carefully: `git add -A` then review `git status`
   - Remove temporary files before committing: `git reset HEAD temp_*.py` if needed
   - Use descriptive commit messages following conventional commits format
   - Push changes after validation: `git push origin <branch-name>`

---

## VS Code Tool Usage Patterns

### **For CI/Test Validation:**
```python
# Create temp_ci_validation.py
# Run comprehensive checks
# Remove temporary file
```

### **For Import Structure Testing:**
```python
# Create temp_import_test.py
# Test all critical imports
# Validate structure compatibility
# Clean up
```

### **For Large Script Execution:**
```python
# Create temp_<feature>_test.py
# Include error handling
# Provide clear output
# Remove after completion
```

---

## Error Handling Patterns

### **Common Windows Issues:**
- **SQLite file locking**: Expected in test teardown, not a code issue
- **Permission errors on temp file cleanup**: Use `-Force` flag with `Remove-Item`
- **Virtual environment activation**: Always check `.venv` exists and activate

### **Test Failure Analysis:**
- **Missing dependencies**: Ensure virtual environment is activated
- **Import errors**: Create temporary test file to validate import structure
- **Threading issues**: Often test environment specific, not code issues

---

## Workflow Examples

### **Validating CI Compatibility:**
1. Create `temp_ci_validation.py` with syntax checks
2. Run in virtual environment: `.venv\Scripts\activate ; python temp_ci_validation.py`
3. Review output and fix issues
4. Clean up: `Remove-Item temp_ci_validation.py -Force`

### **Testing Complex Changes:**
1. Create focused temporary test file
2. Activate virtual environment
3. Run validation with clear output
4. Make necessary fixes
5. Re-run validation
6. Clean up temporary files
7. Commit changes

### **Large File Analysis:**
1. Use `read_file` with focused line ranges
2. Create temporary analysis script if needed
3. Run analysis in virtual environment
4. Document findings
5. Clean up temporary files

---

## Integration with GitHub Copilot Instructions

This file **supplements** the GitHub Copilot instructions and should be used together:

1. **Start with** `.github/copilot-instructions.md` for project structure and coding standards
2. **Apply** VS Code-specific patterns from this file for environment and tool usage
3. **Follow** the same file organization rules (tests in `backend/tests/`, demos in `backend/examples/`)
4. **Maintain** the same import structure (`backend.src.*`) and architecture requirements

**When in doubt, prioritize the GitHub Copilot instructions for code structure and this file for VS Code operational patterns.**

---

## Quick Reference

**Before any Python operation:**
```powershell
cd "c:\sw\code\docrag\backend" ; .venv\Scripts\activate
```

**For testing complex logic:**
```powershell
# Create temp file, run in venv, clean up
New-Item temp_test.py ; python temp_test.py ; Remove-Item temp_test.py -Force
```

**For committing changes:**
```powershell
git add -A ; git status ; git commit -m "message" ; git push origin branch-name
```

Follow these patterns consistently for optimal VS Code Copilot assistance while maintaining compatibility with the broader repository guidelines.
