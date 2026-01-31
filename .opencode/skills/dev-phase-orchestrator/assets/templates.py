# Test Coverage Template
# coverage:skip

"""Template for test coverage reports."""


def generate_coverage_report(
    module_name: str, coverage_percentage: float, missing_lines: list[str]
) -> str:
    """Generate standardized coverage report."""

    status = "✅ PASS" if coverage_percentage > 90 else "❌ FAIL"

    report = f"""
## Test Coverage Report - {module_name}

### Overall Coverage
- **Percentage**: {coverage_percentage:.1f}%
- **Status**: {status}
- **Target**: >90%

### Missing Coverage ({len(missing_lines)} lines)
"""

    if missing_lines:
        for i, line in enumerate(missing_lines[:10], 1):
            report += f"{i}. {line}\n"

        if len(missing_lines) > 10:
            report += f"... and {len(missing_lines) - 10} more lines\n"
    else:
        report += "None\n"

    return report


# Acceptance Criteria Verification Template
def generate_ac_report(phase: str, ac_results: list[dict]) -> str:
    """Generate standardized AC verification report."""

    report = f"""
## Phase {phase} Verification Report

### Acceptance Criteria Status
| AC | Requirement | Status | Notes |
|----|-------------|---------|-------|
"""

    for result in ac_results:
        status_icon = result.get("status", "❌")
        report += f"| {result['ac_id']} | {result['requirement']} | {status_icon} | {result.get('notes', '')} |\n"

    return report
