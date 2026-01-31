#!/usr/bin/env python3
"""
Phase alignment agent implementation.
Verifies alignment between tasks, product specs, and implementation plans.
"""

import json
import re
from pathlib import Path
from typing import Dict, Any, List, Optional, Tuple
from datetime import datetime, timezone


class AlignmentAgent:
    """Isolated context alignment verification agent."""

    def __init__(self, workspace_root: Path = Path.cwd() / ".phase-workflow"):
        self.workspace_root = workspace_root
        self.workspace_root.mkdir(exist_ok=True)
        self.specs_dir = Path.cwd() / "specs"

    def execute(self, context: Dict[str, Any]) -> Dict[str, Any]:
        """Execute alignment verification phase."""
        try:
            phase_id = context["phase_id"]
            session_id = context["session_id"]

            # Extract specification content
            spec_content = self._extract_specifications(phase_id)

            # Perform alignment analysis
            alignment_result = self._analyze_alignment(phase_id, spec_content)

            # Generate state artifact
            artifact = self._create_alignment_artifact(
                session_id, phase_id, alignment_result
            )

            # Save artifact
            self._save_artifact(artifact)

            return {
                "success": True,
                "state_artifact": artifact,
                "report": {
                    "total_tasks": len(alignment_result.get("approved_tasks", [])),
                    "issues_found": len(alignment_result.get("alignment_issues", [])),
                    "status": alignment_result.get("status", "completed"),
                },
            }

        except Exception as e:
            return {
                "success": False,
                "error": f"Alignment phase failed: {str(e)}",
                "report": {},
            }

    def _extract_specifications(self, phase_id: str) -> Dict[str, Any]:
        """Extract relevant content from specification files."""
        content = {}

        # Extract phase tasks
        tasks_file = self.specs_dir / "doc-server-tasks.md"
        if tasks_file.exists():
            content["tasks"] = self._extract_phase_section(tasks_file, phase_id)
        else:
            content["tasks"] = ""

        # Extract product spec requirements
        product_file = self.specs_dir / "doc-server-product-spec.md"
        if product_file.exists():
            content["product_spec"] = self._extract_relevant_requirements(
                product_file, phase_id
            )
        else:
            content["product_spec"] = ""

        # Extract implementation plan
        plan_file = self.specs_dir / "doc-server-plan.md"
        if plan_file.exists():
            content["implementation_plan"] = self._extract_phase_plan(
                plan_file, phase_id
            )
        else:
            content["implementation_plan"] = ""

        return content

    def _extract_phase_section(self, file_path: Path, phase_id: str) -> str:
        """Extract specific phase section from markdown file."""
        try:
            with open(file_path, "r", encoding="utf-8") as f:
                content = f.read()

            # Look for phase section
            phase_pattern = rf"## Phase {re.escape(phase_id)}.*?(?=## Phase |\Z)"
            match = re.search(phase_pattern, content, re.DOTALL | re.IGNORECASE)

            if match:
                return match.group(0).strip()

            return ""

        except Exception:
            return ""

    def _extract_relevant_requirements(self, file_path: Path, phase_id: str) -> str:
        """Extract requirements relevant to specific phase."""
        try:
            with open(file_path, "r", encoding="utf-8") as f:
                content = f.read()

            # Simple extraction - look for phase-related content
            lines = content.split("\n")
            relevant_lines = []

            for line in lines:
                if any(
                    keyword in line.lower()
                    for keyword in [f"phase {phase_id}", "requirement", "specification"]
                ):
                    relevant_lines.append(line)

            return "\n".join(relevant_lines)

        except Exception:
            return ""

    def _extract_phase_plan(self, file_path: Path, phase_id: str) -> str:
        """Extract implementation plan for specific phase."""
        return self._extract_phase_section(file_path, phase_id)

    def _analyze_alignment(
        self, phase_id: str, content: Dict[str, Any]
    ) -> Dict[str, Any]:
        """Analyze alignment between specifications."""
        issues = []
        approved_tasks = []

        # Mock alignment analysis - in real implementation this would be more sophisticated
        tasks_text = content.get("tasks", "")
        product_text = content.get("product_spec", "")
        plan_text = content.get("implementation_plan", "")

        # Check for missing content
        if not tasks_text:
            issues.append(
                {
                    "type": "missing_tasks",
                    "description": f"No tasks found for phase {phase_id}",
                    "severity": "critical",
                    "requires_resolution": True,
                }
            )

        if not product_text:
            issues.append(
                {
                    "type": "missing_product_spec",
                    "description": f"No product spec requirements found for phase {phase_id}",
                    "severity": "critical",
                    "requires_resolution": True,
                }
            )

        # If we have basic content, approve mock tasks
        if tasks_text and not any(issue["severity"] == "critical" for issue in issues):
            approved_tasks = [f"{phase_id}.1", f"{phase_id}.2"]

        status = (
            "approved"
            if not any(issue["severity"] == "critical" for issue in issues)
            else "needs_review"
        )

        return {
            "status": status,
            "alignment_issues": issues,
            "approved_tasks": approved_tasks,
            "scope_boundaries": {
                "included_modules": [f"doc_server/ingestion/*"],
                "excluded_modules": [],
                "file_patterns": ["*.py", "test_*.py"],
            },
            "decision_log": [
                {
                    "decision": "Mock alignment analysis completed",
                    "rationale": "Basic content validation performed",
                    "timestamp": datetime.now(timezone.utc).isoformat(),
                }
            ],
        }

    def _create_alignment_artifact(
        self, session_id: str, phase_id: str, result: Dict[str, Any]
    ) -> Dict[str, Any]:
        """Create alignment state artifact."""
        return {
            "session_id": session_id,
            "phase_id": phase_id,
            "phase_type": "alignment",
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "status": "completed",
            "alignment_results": result,
            "metadata": {
                "agent_id": "phase-alignment-agent",
                "created_at": datetime.now(timezone.utc).isoformat(),
                "duration_ms": 600000,  # Mock duration
            },
        }

    def _save_artifact(self, artifact: Dict[str, Any]) -> None:
        """Save state artifact to workspace."""
        artifact_path = self.workspace_root / "state_alignment.json"
        with open(artifact_path, "w") as f:
            json.dump(artifact, f, indent=2, default=str)


def main():
    """Entry point for alignment agent."""
    import sys

    if len(sys.argv) < 2:
        print("Usage: python alignment.py <phase_id>")
        sys.exit(1)

    phase_id = sys.argv[1]
    session_id = f"phase-{phase_id}-{datetime.now(timezone.utc).strftime('%Y%m%d')}"

    agent = AlignmentAgent()
    context = {"phase_id": phase_id, "session_id": session_id}

    result = agent.execute(context)

    if result["success"]:
        print(f"✅ Alignment phase completed for phase {phase_id}")
        print(
            f"Approved tasks: {result['state_artifact']['alignment_results']['approved_tasks']}"
        )
    else:
        print(f"❌ Alignment phase failed: {result['error']}")
        sys.exit(1)


if __name__ == "__main__":
    main()
