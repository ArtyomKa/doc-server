#!/usr/bin/env python3
"""
Utility script for development phase workflow automation.
"""

import re
import sys
from pathlib import Path


def extract_phase_from_message(message: str) -> str | None:
    """Extract phase number from user message."""
    match = re.search(r"phase\s+(\d+\.\d+)", message.lower())
    return match.group(1) if match else None


def extract_tasks_from_file(tasks_file: Path, phase: str) -> dict:
    """Extract tasks for specific phase from tasks file."""
    content = tasks_file.read_text()

    # Find the phase section
    phase_pattern = rf"## Phase\s+{phase.split('\.')[0]}[^#]*?(?=## Phase|$)"
    phase_match = re.search(phase_pattern, content, re.DOTALL)

    if not phase_match:
        return {}

    phase_content = phase_match.group(0)

    # Extract subsections for the specific sub-phase
    subphase_pattern = rf"### {phase}\.[^#]*?(?=\n###|\Z)"
    subphase_match = re.search(subphase_pattern, phase_content, re.DOTALL)

    if not subphase_match:
        return {}

    tasks = {}
    current_task = None

    for line in subphase_match.group(0).split("\n"):
        line = line.strip()
        if line.startswith("- ["):
            task_text = line[2:].strip()
            status = "pending"
            if "x" in task_text[1]:
                status = "completed"
            tasks[task_text] = status

    return tasks


def create_branch_name(phase: str) -> str:
    """Create git branch name for phase."""
    return f"phase-{phase}"


def create_commit_message(phase: str, module_name: str, summary: str) -> str:
    """Create commit message for phase."""
    return f"Phase {phase}: {module_name} - {summary}"


if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python dev_utils.py <command> [args...]")
        print("Commands:")
        print("  extract-phase <message>")
        print("  create-branch <phase>")
        print("  commit-msg <phase> <module> <summary>")
        sys.exit(1)

    command = sys.argv[1]

    if command == "extract-phase":
        message = " ".join(sys.argv[2:])
        phase = extract_phase_from_message(message)
        print(f"Phase: {phase}")

    elif command == "create-branch":
        phase = sys.argv[2]
        print(f"Branch: {create_branch_name(phase)}")

    elif command == "commit-msg":
        phase = sys.argv[2]
        module = sys.argv[3]
        summary = " ".join(sys.argv[4:])
        print(f"Commit: {create_commit_message(phase, module, summary)}")

    else:
        print(f"Unknown command: {command}")
        sys.exit(1)
