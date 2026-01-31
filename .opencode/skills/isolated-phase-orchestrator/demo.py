#!/usr/bin/env python3
"""
Simple demonstration of isolated-phase-orchestrator skill.
Simulates user saying "I'm going to start phase 2.4"
"""

import sys
import re
from pathlib import Path

# Add skill directory to Python path
skill_dir = Path(__file__).parent
sys.path.insert(0, str(skill_dir))

from workflows.coordination import PhaseCoordinator


def simulate_user_interaction():
    """Simulate user saying 'I'm going to start phase X.Y'"""

    print('👤 User: "I\'m going to start phase 2.4"')
    print()

    # Parse phase ID from user message
    user_message = "I'm going to start phase 2.4"
    match = re.search(r"phase\s+(\d+\.\d+)", user_message.lower())

    if not match:
        print("❌ Could not parse phase ID from user message")
        return False

    phase_id = match.group(1)
    print(f"🤖 Detected phase: {phase_id}")
    print()

    # Create coordinator and run workflow
    print("🚀 Starting isolated phase workflow...")
    print("=" * 50)

    workspace_root = Path.cwd() / ".demo-workflow"
    workspace_root.mkdir(exist_ok=True)

    coordinator = PhaseCoordinator(workspace_root)

    try:
        # Create session
        session = coordinator.create_session(phase_id)
        print(f"✅ Created session: {session.session_id}")

        # Simulate the 4-phase workflow
        phases = [
            ("alignment", "Verifying task-product spec alignment..."),
            ("implementation", "Implementing approved tasks in isolated branch..."),
            (
                "verification",
                "Validating implementation against acceptance criteria...",
            ),
            ("commit", "Committing verified changes with metadata..."),
        ]

        for i, (phase_type, description) in enumerate(phases, 1):
            print(f"\n{i}. {description}")

            # Build context
            context = {
                "phase_id": phase_id,
                "session_id": session.session_id,
                f"{phase_type}_state": {"status": "approved"} if i > 1 else None,
            }

            # Execute phase
            result = coordinator.execute_phase(phase_type, context, timeout_ms=10000)

            if result["success"]:
                print(f"   ✅ {phase_type.title()} phase completed successfully")
                report = result.get("report", {})
                if report.get("status") != "unknown":
                    print(f"   📊 {report}")
            else:
                print(f"   ❌ {phase_type.title()} phase failed: {result.get('error')}")
                return False

        # Show final dashboard
        print("\n" + "=" * 50)
        print("📊 Final Workflow Dashboard")
        print("=" * 50)

        dashboard = coordinator.get_session_dashboard()
        print(f"Session ID: {dashboard['session_id']}")
        print(f"Phase: {dashboard['phase_id']}")
        print(f"Overall Status: {dashboard['status']}")
        print(f"Progress: {dashboard['progress']:.1f}%")

        print("\nPhase Details:")
        for phase_name, details in dashboard.get("phases", {}).items():
            print(f"  {phase_name.title()}: {details['status'].upper()}")

        print("\n🎉 Phase 2.4 workflow completed successfully!")

        # Cleanup
        import shutil

        shutil.rmtree(workspace_root, ignore_errors=True)

        return True

    except Exception as e:
        print(f"❌ Workflow failed: {e}")
        import traceback

        traceback.print_exc()
        return False


if __name__ == "__main__":
    success = simulate_user_interaction()
    sys.exit(0 if success else 1)
