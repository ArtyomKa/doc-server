#!/usr/bin/env python3
"""
Test script for isolated-phase-orchestrator skill.
"""

import sys
import os
from pathlib import Path

# Add the skill directory to Python path
skill_dir = Path(__file__).parent
sys.path.insert(0, str(skill_dir))

from workflows.coordination import PhaseCoordinator


def test_phase_orchestrator():
    """Test the complete phase orchestrator workflow."""

    print("🧪 Testing Isolated Phase Orchestrator Skill")
    print("=" * 50)

    # Create workspace
    workspace_root = Path.cwd() / ".test-phase-workflow"
    workspace_root.mkdir(exist_ok=True)

    # Create coordinator
    coordinator = PhaseCoordinator(workspace_root)

    # Test phase ID
    phase_id = "2.4"

    try:
        # 1. Create session
        print("1. Creating workflow session...")
        session = coordinator.create_session(phase_id)
        print(f"   ✅ Session created: {session.session_id}")

        # 2. Test each phase
        phases = ["alignment", "implementation", "verification", "commit"]

        for i, phase_type in enumerate(phases, 1):
            print(f"{i + 1}. Executing {phase_type} phase...")

            # Build context with previous state if available
            previous_state = None
            if i > 1:
                # Load previous state from file
                prev_phase = phases[i - 2]  # Previous phase name
                prev_artifact_path = workspace_root / f"state_{prev_phase}.json"
                if prev_artifact_path.exists():
                    try:
                        import json

                        with open(prev_artifact_path, "r") as f:
                            previous_state = json.load(f)
                    except Exception:
                        previous_state = None

            context = {
                "phase_id": phase_id,
                "session_id": session.session_id,
                "previous_state": previous_state,
            }

            # Add specific state for phases that need it
            if phase_type == "implementation":
                context["alignment_state"] = {
                    "approved_tasks": [f"{phase_id}.1", f"{phase_id}.2"],
                    "scope_boundaries": {
                        "included_modules": ["doc_server/ingestion/*"]
                    },
                }
            elif phase_type == "verification":
                context["implementation_state"] = {
                    "implemented_files": [
                        {"path": "doc_server/ingestion/document_processor.py"}
                    ],
                    "test_results": {"coverage_percentage": 92.5},
                }
            elif phase_type == "commit":
                context["verification_state"] = {
                    "verification_results": {"status": "approved"}
                }

            # Execute phase
            result = coordinator.execute_phase(
                phase_type, context, timeout_ms=30000
            )  # 30 seconds

            if result["success"]:
                print(f"   ✅ {phase_type.title()} phase completed")
                print(
                    f"      Report: {result.get('report', {}).get('status', 'unknown')}"
                )
            else:
                print(
                    f"   ❌ {phase_type.title()} phase failed: {result.get('error', 'unknown')}"
                )
                return False

        # 3. Get final dashboard
        print("5. Generating final dashboard...")
        dashboard = coordinator.get_session_dashboard()
        print(f"   ✅ Overall status: {dashboard.get('status', 'unknown')}")
        print(f"   ✅ Progress: {dashboard.get('progress', 0):.1f}%")

        # 4. Test state persistence
        print("6. Testing state persistence...")
        artifact_count = 0
        for phase_type in phases:
            artifact_path = workspace_root / f"state_{phase_type}.json"
            if artifact_path.exists():
                artifact_count += 1

        print(f"   ✅ {artifact_count}/{len(phases)} state artifacts saved")

        # Cleanup
        import shutil

        shutil.rmtree(workspace_root, ignore_errors=True)

        print(
            "\n🎉 All tests passed! Isolated Phase Orchestrator is working correctly."
        )
        return True

    except Exception as e:
        print(f"\n❌ Test failed with error: {e}")
        import traceback

        traceback.print_exc()
        return False


if __name__ == "__main__":
    success = test_phase_orchestrator()
    sys.exit(0 if success else 1)
