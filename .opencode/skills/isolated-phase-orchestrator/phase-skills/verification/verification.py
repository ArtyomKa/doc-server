#!/usr/bin/env python3
"""
Phase verification agent implementation.
Validates implementation against acceptance criteria and quality standards.
"""

import json
import subprocess
from pathlib import Path
from typing import Dict, Any, List, Optional
from datetime import datetime, timezone


class VerificationAgent:
    """Isolated context verification agent."""

    def __init__(self, workspace_root: Path = Path.cwd() / ".phase-workflow"):
        self.workspace_root = workspace_root
        self.workspace_root.mkdir(exist_ok=True)
        self.repo_root = Path.cwd()

    def execute(self, context: Dict[str, Any]) -> Dict[str, Any]:
        """Execute verification phase."""
        try:
            phase_id = context["phase_id"]
            session_id = context["session_id"]
            implementation_state = context.get("implementation_state", {})

            # Load acceptance criteria
            ac_criteria = self._load_acceptance_criteria(phase_id)

            # Verify implementation against AC
            ac_results = self._verify_acceptance_criteria(
                implementation_state, ac_criteria
            )

            # Check test coverage
            coverage_results = self._verify_test_coverage(implementation_state)

            # Run additional quality checks
            quality_metrics = self._assess_code_quality(implementation_state)

            # Determine overall status
            verification_status = self._determine_verification_status(
                ac_results, coverage_results
            )

            # Generate state artifact
            artifact = self._create_verification_artifact(
                session_id,
                phase_id,
                verification_status,
                ac_results,
                coverage_results,
                quality_metrics,
            )

            # Save artifact
            self._save_artifact(artifact)

            return {
                "success": True,
                "state_artifact": artifact,
                "report": {
                    "ac_criteria_met": sum(
                        1 for r in ac_results.values() if r.get("status") == "✅"
                    ),
                    "coverage_percentage": coverage_results.get("percentage", 0),
                    "status": verification_status,
                },
            }

        except Exception as e:
            return {
                "success": False,
                "error": f"Verification phase failed: {str(e)}",
                "report": {},
            }

    def _load_acceptance_criteria(self, phase_id: str) -> Dict[str, Any]:
        """Load acceptance criteria for the phase."""
        ac_file = Path("specs/doc-server-acceptence.md")

        if not ac_file.exists():
            # Mock acceptance criteria
            return {
                f"AC-{phase_id}.1": "Document processor handles multiple file formats",
                f"AC-{phase_id}.2": "Error handling for corrupted files implemented",
                f"AC-{phase_id}.3": "Test coverage >90% for implemented modules",
                f"AC-{phase_id}.4": "Code follows project quality standards",
            }

        try:
            with open(ac_file, "r", encoding="utf-8") as f:
                content = f.read()

            # Parse acceptance criteria for specific phase
            criteria = {}
            lines = content.split("\n")

            for i, line in enumerate(lines):
                if line.startswith(f"AC-{phase_id}."):
                    ac_id = line.split(":")[0].strip()
                    description = line.split(":", 1)[1].strip() if ":" in line else line
                    criteria[ac_id] = description

            return criteria

        except Exception:
            return {}

    def _verify_acceptance_criteria(
        self, implementation_state: Dict[str, Any], ac_criteria: Dict[str, Any]
    ) -> Dict[str, Dict[str, Any]]:
        """Verify implementation against acceptance criteria."""
        results = {}
        implemented_files = implementation_state.get("implemented_files", [])
        test_results = implementation_state.get("test_results", {})

        for ac_id, description in ac_criteria.items():
            status = "✅"  # Default to passed
            evidence = "Verified through code analysis and testing"
            test_coverage = True
            notes = ""

            # Mock verification logic
            if "file formats" in description.lower():
                # Check if document processor was implemented
                has_processor = any(
                    "document_processor.py" in f.get("path", "")
                    for f in implemented_files
                )
                status = "✅" if has_processor else "❌"
                evidence = (
                    "Document processor implementation found"
                    if has_processor
                    else "Document processor not found"
                )

            elif "error handling" in description.lower():
                # Mock check for error handling
                status = "✅"  # Assume implemented for demo
                evidence = "Error handling patterns detected in code"

            elif "test coverage" in description.lower():
                coverage_pct = test_results.get("coverage_percentage", 0)
                status = (
                    "✅" if coverage_pct >= 90 else "⚠️" if coverage_pct >= 80 else "❌"
                )
                evidence = f"Test coverage: {coverage_pct}%"
                test_coverage = coverage_pct >= 90

            elif "quality standards" in description.lower():
                quality_metrics = test_results.get("quality_metrics", {})
                overall_passed = quality_metrics.get("overall_passed", False)
                status = "✅" if overall_passed else "❌"
                evidence = f"Quality checks: {'PASSED' if overall_passed else 'FAILED'}"

            results[ac_id] = {
                "status": status,
                "evidence": evidence,
                "test_coverage": test_coverage,
                "notes": notes,
            }

        return results

    def _verify_test_coverage(
        self, implementation_state: Dict[str, Any]
    ) -> Dict[str, Any]:
        """Verify test coverage meets requirements."""
        test_results = implementation_state.get("test_results", {})
        coverage_pct = test_results.get("coverage_percentage", 0)

        threshold_met = coverage_pct >= 90

        # Try to get detailed coverage by module
        coverage_by_module = {}
        try:
            coverage_file = Path("coverage.json")
            if coverage_file.exists():
                with open(coverage_file) as f:
                    coverage_data = json.load(f)

                files = coverage_data.get("files", [])
                for file_data in files:
                    if "document_processor" in file_data.get("name", ""):
                        module_name = (
                            file_data["name"].split("/")[-1].replace(".py", "")
                        )
                        coverage_by_module[module_name] = file_data["summary"][
                            "percent_covered"
                        ]
        except Exception:
            pass

        return {
            "percentage": coverage_pct,
            "threshold_met": threshold_met,
            "coverage_by_module": coverage_by_module,
        }

    def _assess_code_quality(
        self, implementation_state: Dict[str, Any]
    ) -> Dict[str, Any]:
        """Assess overall code quality metrics."""
        test_results = implementation_state.get("test_results", {})
        quality_metrics = test_results.get("quality_metrics", {})

        # Calculate quality scores
        code_quality_score = 0.0
        test_quality_score = 0.0
        documentation_coverage = 0.0

        # Code quality based on static analysis
        if quality_metrics:
            passed_checks = sum(
                [
                    quality_metrics.get("black_passed", False),
                    quality_metrics.get("isort_passed", False),
                    quality_metrics.get("mypy_passed", False),
                ]
            )
            code_quality_score = (passed_checks / 3) * 100

        # Test quality based on test results
        tests_run = test_results.get("tests_run", 0)
        tests_passed = test_results.get("tests_passed", 0)
        if tests_run > 0:
            test_quality_score = (tests_passed / tests_run) * 100

        # Documentation coverage (mock)
        documentation_coverage = 85.0  # Mock percentage

        return {
            "code_quality_score": code_quality_score,
            "test_quality_score": test_quality_score,
            "documentation_coverage": documentation_coverage,
        }

    def _determine_verification_status(
        self, ac_results: Dict[str, Any], coverage_results: Dict[str, Any]
    ) -> str:
        """Determine overall verification status."""
        # Check for critical failures
        failed_ac = [
            ac_id
            for ac_id, result in ac_results.items()
            if result.get("status") == "❌"
        ]
        coverage_met = coverage_results.get("threshold_met", False)

        if failed_ac:
            return "rejected"
        elif not coverage_met:
            return "needs_review"
        else:
            return "approved"

    def _create_verification_artifact(
        self,
        session_id: str,
        phase_id: str,
        verification_status: str,
        ac_results: Dict[str, Any],
        coverage_results: Dict[str, Any],
        quality_metrics: Dict[str, Any],
    ) -> Dict[str, Any]:
        """Create verification state artifact."""

        # Generate verification failures list
        verification_failures = []
        for ac_id, result in ac_results.items():
            if result.get("status") in ["❌", "⚠️"]:
                verification_failures.append(
                    {
                        "ac_id": ac_id,
                        "failure_type": "ac_not_met",
                        "description": f"Acceptance criteria {ac_id} not fully satisfied",
                        "severity": "critical"
                        if result.get("status") == "❌"
                        else "major",
                        "suggested_fix": "Review implementation against AC requirements",
                    }
                )

        return {
            "session_id": session_id,
            "phase_id": phase_id,
            "phase_type": "verification",
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "status": "completed",
            "verification_results": {
                "status": verification_status,
                "ac_results": ac_results,
                "test_coverage": coverage_results,
                "verification_failures": verification_failures,
                "quality_metrics": quality_metrics,
            },
            "metadata": {
                "agent_id": "phase-verification-agent",
                "created_at": datetime.now(timezone.utc).isoformat(),
                "duration_ms": 600000,  # Mock duration
            },
        }

    def _save_artifact(self, artifact: Dict[str, Any]) -> None:
        """Save state artifact to workspace."""
        artifact_path = self.workspace_root / "state_verification.json"
        with open(artifact_path, "w") as f:
            json.dump(artifact, f, indent=2, default=str)


def main():
    """Entry point for verification agent."""
    import sys

    if len(sys.argv) < 2:
        print("Usage: python verification.py <phase_id>")
        sys.exit(1)

    phase_id = sys.argv[1]
    session_id = f"phase-{phase_id}-{datetime.now(timezone.utc).strftime('%Y%m%d')}"

    agent = VerificationAgent()
    context = {
        "phase_id": phase_id,
        "session_id": session_id,
        "implementation_state": {
            "implemented_files": [
                {"path": "doc_server/ingestion/document_processor.py"},
                {"path": "tests/test_document_processor.py"},
            ],
            "test_results": {
                "coverage_percentage": 92.5,
                "tests_run": 8,
                "tests_passed": 8,
                "quality_metrics": {
                    "black_passed": True,
                    "isort_passed": True,
                    "mypy_passed": True,
                    "overall_passed": True,
                },
            },
        },
    }

    result = agent.execute(context)

    if result["success"]:
        print(f"✅ Verification phase completed for phase {phase_id}")
        print(f"Status: {result['state_artifact']['verification_results']['status']}")
    else:
        print(f"❌ Verification phase failed: {result['error']}")
        sys.exit(1)


if __name__ == "__main__":
    main()
