#!/usr/bin/env python3
"""
State management for isolated phase contexts.
Handles artifact creation, validation, and persistence between phase transitions.
"""

import json
import uuid
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict, Any, Optional, List
import jsonschema


class StateManager:
    """Manages state artifacts for isolated phase handoffs."""

    def __init__(self, workspace_root: Path = Path.cwd() / ".phase-workflow"):
        self.workspace_root = workspace_root
        self.workspace_root.mkdir(exist_ok=True)
        self.schema_path = Path(__file__).parent / "schemas.json"
        self.schemas = self._load_schemas()

    def _load_schemas(self) -> Dict[str, Any]:
        """Load JSON schemas for state validation."""
        try:
            with open(self.schema_path, "r", encoding="utf-8") as f:
                content = f.read()
                if not content.strip():
                    # Handle empty schema file
                    return {"oneOf": [], "definitions": {}}
                return json.loads(content)
        except FileNotFoundError:
            # Handle missing schema file
            print(
                f"Warning: Schema file not found at {self.schema_path}, using empty schema"
            )
            return {"oneOf": [], "definitions": {}}
        except json.JSONDecodeError as e:
            # Handle invalid JSON in schema file
            print(f"Warning: Invalid JSON in schema file {self.schema_path}: {e}")
            return {"oneOf": [], "definitions": {}}
        except Exception as e:
            # Handle any other file reading errors
            print(f"Warning: Failed to load schema file {self.schema_path}: {e}")
            return {"oneOf": [], "definitions": {}}

    def create_session_id(self, phase_id: str) -> str:
        """Generate unique session identifier."""
        today = datetime.now(timezone.utc).strftime("%Y%m%d")
        return f"phase-{phase_id}-{today}"

    def create_artifact(
        self,
        phase_id: str,
        phase_type: str,
        data: Dict[str, Any],
        session_id: Optional[str] = None,
    ) -> Dict[str, Any]:
        """Create and validate a state artifact."""
        if session_id is None:
            session_id = self.create_session_id(phase_id)

        # Base artifact structure
        artifact = {
            "session_id": session_id,
            "phase_id": phase_id,
            "phase_type": phase_type,
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "status": "completed",
            "metadata": {
                "agent_id": f"phase-{phase_type}-agent",
                "created_at": datetime.now(timezone.utc).isoformat(),
            },
        }

        # Merge phase-specific data
        artifact.update(data)

        # Validate against schema
        self.validate_artifact(artifact)

        # Persist artifact
        artifact_path = self.workspace_root / f"state_{phase_type}.json"
        with open(artifact_path, "w") as f:
            json.dump(artifact, f, indent=2, default=str)

        return artifact

    def validate_artifact(self, artifact: Dict[str, Any]) -> None:
        """Validate artifact against appropriate schema."""
        phase_type = artifact.get("phase_type")

        # Find matching schema (oneOf)
        for schema in self.schemas.get("oneOf", []):
            # Check for phase_type in different possible locations
            schema_phase_type = (
                schema.get("properties", {}).get("phase_type", {}).get("const")
                or schema.get("allOf", [{}])[-1]
                .get("properties", {})
                .get("phase_type", {})
                .get("const")
                or schema.get("title", "").lower().replace(" state artifact", "")
            )

            if schema_phase_type == phase_type:
                try:
                    jsonschema.validate(artifact, schema)
                    return
                except jsonschema.ValidationError as e:
                    raise ValueError(f"Artifact validation failed: {e.message}")

        raise ValueError(f"No schema found for phase type: {phase_type}")

    def load_previous_state(self, current_phase_type: str) -> Optional[Dict[str, Any]]:
        """Load state artifact from previous phase."""
        phase_order = ["alignment", "implementation", "verification", "commit"]

        try:
            current_index = phase_order.index(current_phase_type)
            if current_index == 0:
                return None  # No previous phase for alignment

            previous_phase = phase_order[current_index - 1]
            artifact_path = self.workspace_root / f"state_{previous_phase}.json"

            if artifact_path.exists():
                with open(artifact_path, "r") as f:
                    return json.load(f)

        except (ValueError, IndexError, json.JSONDecodeError):
            pass

        return None

    def cleanup_phase_context(self, phase_type: str) -> None:
        """Clean up context and temporary files for completed phase."""
        # In a real implementation, this would clean up any temporary context files
        # For now, we just verify the state artifact exists
        artifact_path = self.workspace_root / f"state_{phase_type}.json"
        if not artifact_path.exists():
            raise FileNotFoundError(f"No state artifact found for phase: {phase_type}")

    def get_session_status(self, session_id: str) -> Dict[str, Any]:
        """Get overall status of workflow session."""
        phase_types = ["alignment", "implementation", "verification", "commit"]
        status = {
            "session_id": session_id,
            "phases": {},
            "overall_status": "pending",
        }

        completed_phases = 0
        for phase_type in phase_types:
            artifact_path = self.workspace_root / f"state_{phase_type}.json"
            if artifact_path.exists():
                try:
                    with open(artifact_path, "r") as f:
                        artifact = json.load(f)
                        status["phases"][phase_type] = {
                            "status": artifact.get("status"),
                            "timestamp": artifact.get("timestamp"),
                        }
                        if artifact.get("status") == "completed":
                            completed_phases += 1
                except json.JSONDecodeError:
                    status["phases"][phase_type] = {"status": "error"}
            else:
                status["phases"][phase_type] = {"status": "pending"}

        # Determine overall status
        if completed_phases == len(phase_types):
            status["overall_status"] = "completed"
        elif completed_phases > 0:
            status["overall_status"] = "in_progress"

        return status


# Utility functions for creating specific artifact types
def create_alignment_artifact(
    session_id: str,
    phase_id: str,
    alignment_status: str,
    approved_tasks: List[str],
    scope_boundaries: Dict[str, List[str]],
    alignment_issues: Optional[List[Dict[str, Any]]] = None,
) -> Dict[str, Any]:
    """Create alignment phase state artifact."""
    state_manager = StateManager()

    return state_manager.create_artifact(
        phase_id=phase_id,
        phase_type="alignment",
        data={
            "alignment_results": {
                "status": alignment_status,
                "alignment_issues": alignment_issues or [],
                "approved_tasks": approved_tasks,
                "scope_boundaries": scope_boundaries,
                "decision_log": [],
            }
        },
        session_id=session_id,
    )


def create_implementation_artifact(
    session_id: str,
    phase_id: str,
    branch_name: str,
    implemented_files: List[Dict[str, Any]],
    build_status: str = "success",
    test_results: Optional[Dict[str, Any]] = None,
) -> Dict[str, Any]:
    """Create implementation phase state artifact."""
    state_manager = StateManager()

    return state_manager.create_artifact(
        phase_id=phase_id,
        phase_type="implementation",
        data={
            "implementation_results": {
                "branch_name": branch_name,
                "implemented_files": implemented_files,
                "build_status": build_status,
                "test_results": test_results or {},
                "implementation_notes": [],
            }
        },
        session_id=session_id,
    )


def create_verification_artifact(
    session_id: str,
    phase_id: str,
    verification_status: str,
    ac_results: Dict[str, Dict[str, Any]],
    test_coverage: Dict[str, Any],
    verification_failures: Optional[List[Dict[str, Any]]] = None,
) -> Dict[str, Any]:
    """Create verification phase state artifact."""
    state_manager = StateManager()

    return state_manager.create_artifact(
        phase_id=phase_id,
        phase_type="verification",
        data={
            "verification_results": {
                "status": verification_status,
                "ac_results": ac_results,
                "test_coverage": test_coverage,
                "verification_failures": verification_failures or [],
                "quality_metrics": {},
            }
        },
        session_id=session_id,
    )


def create_commit_artifact(
    session_id: str,
    phase_id: str,
    commit_hash: str,
    files_committed: List[Dict[str, Any]],
    commit_message: str,
    branch_name: str,
) -> Dict[str, Any]:
    """Create commit phase state artifact."""
    state_manager = StateManager()

    return state_manager.create_artifact(
        phase_id=phase_id,
        phase_type="commit",
        data={
            "commit_results": {
                "commit_hash": commit_hash,
                "branch_name": branch_name,
                "commit_message": commit_message,
                "files_committed": files_committed,
                "remote_status": {
                    "pushed": False,
                    "remote_url": "",
                    "branch_created": True,
                },
            }
        },
        session_id=session_id,
    )
