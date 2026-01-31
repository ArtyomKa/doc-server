# Workflow Reference Guide

## Phase 1: Alignment Verification Template

### Input Patterns
- User: "I'm going to start implementation of phase X.Y"
- Extract phase number (e.g., "2.3")

### Alignment Check Matrix
| Requirement | Tasks Spec | Product Spec | Plan Spec | Status |
|------------|-------------|-------------|-----------|---------|
| Feature X | ✓ present | ✓ included | ✓ planned | ✅ ALIGNED |
| Feature Y | ✓ present | ❌ missing | ✓ planned | ⚠️ GAP |

### Common Alignment Issues
1. **Scope Creep**: Tasks include features not in product spec
2. **Missing Requirements**: Product spec has requirements not in tasks
3. **Implementation Gaps**: Plan doesn't cover all tasks
4. **Technical Contradictions**: Plan conflicts with product decisions

---

## Phase 3: Acceptance Verification Template

### AC Verification Pattern
For each AC-X.Y.Z criterion:
1. **Code Check**: Is feature implemented in source code?
2. **Test Check**: Is feature covered by tests?
3. **Run Test**: Does test pass?
4. **Coverage Check**: Is line covered by test coverage?

### Status Indicators
- ✅ **Fully Met**: All checks pass
- ⚠️ **Partially Met**: Some checks pass, others fail
- ❌ **Not Met**: No checks pass

### Test Commands
```bash
# Run specific module tests
pytest tests/test_<module>.py -v

# Coverage check
pytest --cov=doc_server.<module> --cov-report=term-missing

# Full test suite
pytest tests/ -v --cov=doc_server
```

### Report Structure
```
## Phase X.Y Verification Report

### Acceptance Criteria Status
| AC | Requirement | Status | Notes |
|----|-------------|---------|-------|
| 2.3.1 | .gitignore parsing | ✅ | Fully implemented and tested |
| 2.3.2 | Allowlist enforcement | ⚠️ | Missing .json extension |

### Test Results
- Tests Run: 15/15 passed
- Coverage: 92.3%
- Issues: 1 partial implementation

### Recommendations
[ ] Add .json extension to allowlist
[ ] Add test cases for JSON file handling
```

---

## Git Commands Reference

### Branch Management
```bash
# Create phase branch
git checkout -b phase-2.4

# Commit phase work
git add .
git commit -m "Phase 2.4: file_filter - Implement .gitignore parsing and allowlist"

# Merge to main when complete
git checkout main
git merge phase-2.4
git branch -d phase-2.4
```

### Commit Message Patterns
- Feature work: "Phase X.Y: module_name - Brief description"
- Bug fixes: "Phase X.Y: module_name - Fix issue description"
- Documentation: "Phase X.Y: README - Update installation guide"