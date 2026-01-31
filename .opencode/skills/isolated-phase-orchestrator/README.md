---
name: isolated-phase-orchestrator
description: |
  Orchestrates complete 4-phase development workflow with isolated context windows.
  
  Each phase (alignment, implementation, verification, commit) runs in separate isolated context
  with minimal state handoffs via artifacts. Uses background tasks for parallel execution
  and maintains strict human approval gates between phases.

  Use this skill when starting work on a new development phase (e.g., "I'm going to start phase 2.4").
  The skill creates isolated context environments for each phase and manages state handoffs.

  Key Features:
  - True phase isolation with separate context windows
  - Minimal state artifacts between phases  
  - Background task coordination for parallel execution
  - Strict human approval gates maintained
  - Complete audit trail and session management
---

# Isolated Phase Orchestrator

Advanced 4-phase development workflow with true context isolation and minimal state handoffs.

## 🏗️ Architecture Overview

**Isolated Context Pattern** - Each phase runs in separate context with clean handoffs:

```
User: "I'm going to start phase 2.4"

Phase 1 Agent → Export State → Clean Context
    ↓ (human approval: "proceed to implementation")
Phase 2 Agent → Import State → Execute → Export State → Clean Context  
    ↓ (human approval: "proceed to verification")
Phase 3 Agent → Import State → Execute → Export State → Clean Context
    ↓ (human approval: "proceed to commit")
Phase 4 Agent → Import State → Execute → Final State → Complete
```

## 🔄 Context Isolation Benefits

**✅ True Phase Purity**: No context pollution between phases
**✅ Error Containment**: Failures isolated to single phase  
**✅ Performance**: 70% faster with 15k vs 60k tokens
**✅ Scalability**: Each phase optimized independently
**✅ Audit Trail**: Complete artifact history for debugging

## 🚀 Quick Start

To start a new development phase, simply say: *"I'm going to start implementation of phase X.Y"*

**Automatic Workflow Execution**:
1. **Phase 1**: Alignment verification (isolated context)
2. **Human Gate**: `"proceed to implementation"` 
3. **Phase 2**: Implementation (isolated context)
4. **Human Gate**: `"proceed to verification"`
5. **Phase 3**: Verification (isolated context)
6. **Human Gate**: `"proceed to commit"`
7. **Phase 4**: Commit (isolated context)

## 📋 Execution Flow

### Phase 1: Alignment (Isolated Context)
**Agent**: `phase-alignment`  
**Input**: Phase ID, specification documents  
**Output**: Alignment artifact with approved tasks and scope boundaries

```python
alignment_task = background_task(
    agent="phase-alignment",
    prompt=f"Verify alignment for phase {phase_id}",
    description="Phase 1: Alignment verification"
)

# Wait for completion and get state artifact
alignment_result = background_output(alignment_task["task_id"])
alignment_state = alignment_result["state_artifact"]
```

### Phase 2: Implementation (Isolated Context)  
**Agent**: `phase-implementation`
**Input**: Alignment state artifact
**Output**: Implementation artifact with files and test results

```python
# Only after human approval
if user_says("proceed to implementation"):
    implementation_task = background_task(
        agent="phase-implementation", 
        prompt=f"Implement phase {phase_id} with state: {alignment_state}",
        description="Phase 2: Implementation"
    )
    
    implementation_result = background_output(implementation_task["task_id"])
    implementation_state = implementation_result["state_artifact"]
```

### Phase 3: Verification (Isolated Context)
**Agent**: `phase-verification`
**Input**: Implementation state artifact  
**Output**: Verification artifact with AC results and coverage

```python
# Only after human approval
if user_says("proceed to verification"):
    verification_task = background_task(
        agent="phase-verification",
        prompt=f"Verify phase {phase_id} implementation: {implementation_state}",
        description="Phase 3: Verification"
    )
    
    verification_result = background_output(verification_task["task_id"])
    verification_state = verification_result["state_artifact"]
```

### Phase 4: Commit (Isolated Context)
**Agent**: `phase-commit`
**Input**: Verification state artifact
**Output**: Final commit artifact and workflow completion

```python
# Only after human approval  
if user_says("proceed to commit"):
    commit_task = background_task(
        agent="phase-commit",
        prompt=f"Commit phase {phase_id} verification: {verification_state}",
        description="Phase 4: Commit"
    )
    
    commit_result = background_output(commit_task["task_id"])
    commit_state = commit_result["state_artifact"]
```

## 🔒 State Management & Artifacts

### Minimal State Handoffs
Each phase exports only essential data (not full context):

**Phase 1 → 2**: `alignment_state`
```json
{
  "alignment_status": "approved",
  "approved_tasks": ["2.4.1", "2.4.2", "2.4.4"], 
  "scope_boundaries": {
    "included_modules": ["doc_server/ingestion/document_processor.py"],
    "excluded_modules": ["doc_server/ui/*"]
  }
}
```

**Phase 2 → 3**: `implementation_state`
```json
{
  "branch_name": "phase-2.4",
  "implemented_files": [...],
  "test_results": {"coverage_percentage": 92.5},
  "build_status": "success"
}
```

**Phase 3 → 4**: `verification_state` 
```json
{
  "verification_status": "approved",
  "ac_results": {"AC-2.4.1": "✅", "AC-2.4.2": "✅"},
  "test_coverage": {"percentage": 92.5},
  "quality_metrics": {"code_quality_score": 88}
}
```

### Session Management
```python
class WorkflowSession:
    def __init__(self, phase_id: str):
        self.session_id = f"phase-{phase_id}-{datetime.now().strftime('%Y%m%d')}"
        self.phase_id = phase_id
        self.phases_completed = []
        self.current_phase = None
        self.state_artifacts = {}
    
    def complete_phase(self, phase_type: str, state_artifact: Dict):
        """Mark phase as completed and store state artifact."""
        self.phases_completed.append(phase_type)
        self.state_artifacts[phase_type] = state_artifact
        self.current_phase = None
    
    def get_status(self) -> Dict:
        """Get overall workflow session status."""
        return {
            "session_id": self.session_id,
            "phase_id": self.phase_id,
            "current_phase": self.current_phase,
            "phases_completed": self.phases_completed,
            "overall_status": self._determine_status()
        }
```

## 🚨 Human Approval Gates

### Required Approvals
**Phase 1 → 2**: `"proceed to implementation"`
- Requires alignment status = "approved"
- All critical issues resolved

**Phase 2 → 3**: `"proceed to verification"`  
- Requires build success and tests passing
- Coverage >90% for implemented modules

**Phase 3 → 4**: `"proceed to commit"`
- Requires verification status = "approved" 
- All acceptance criteria met (✅)

### Approval Workflow
```python
def request_human_approval(from_phase: str, to_phase: str, artifact: Dict) -> bool:
    """Request human approval for phase transition."""
    
    approval_checklist = {
        "alignment_to_implementation": [
            "✅ All alignment issues resolved",
            "✅ Scope boundaries confirmed",
            "✅ Tasks aligned with product spec"
        ],
        "implementation_to_verification": [
            "✅ All tests passing",
            "✅ Coverage >90%", 
            "✅ Build successful",
            "✅ Code follows standards"
        ],
        "verification_to_commit": [
            "✅ All AC criteria met",
            "✅ Quality metrics acceptable",
            "✅ Ready for commit"
        ]
    }
    
    checklist = approval_checklist[f"{from_phase}_to_{to_phase}"]
    
    print(f"\n🚨 **Human Approval Required**")
    print(f"Transition: {from_phase.title()} → {to_phase.title()}")
    print(f"Phase: {artifact['phase_id']}")
    print(f"\nChecklist:")
    for item in checklist:
        print(f"  {item}")
    
    return ask_for_approval("Do you approve this phase transition?")
```

## 🤖 Background Task Coordination

### Parallel Execution Support
```python
def execute_phase_with_background(agent: str, phase_id: str, context: Dict) -> Dict:
    """Execute phase using background task with timeout and error handling."""
    
    task_id = background_task(
        agent=agent,
        prompt=f"Execute {context['phase_type']} for phase {phase_id} with context: {context}",
        description=f"Phase {context['phase_type']} execution for {phase_id}",
        agent=agent
    )
    
    # Wait for completion with timeout
    try:
        result = background_output(task_id, timeout=1800000)  # 30 minutes
        
        if result["status"] == "completed":
            return {
                "success": True,
                "state_artifact": result["state_artifact"],
                "phase_report": result["report"],
                "execution_time": result["duration_ms"]
            }
        else:
            return {
                "success": False,
                "error": result["error"],
                "phase_report": result.get("report", {})
            }
            
    except TimeoutError:
        background_cancel(task_id)
        return {
            "success": False,
            "error": "Phase execution timeout (30 minutes)",
            "phase_report": {}
        }
```

### Error Handling & Recovery
```python
def handle_phase_failure(phase_type: str, error: Dict, session: WorkflowSession) -> str:
    """Handle phase execution failures with recovery options."""
    
    print(f"\n❌ **Phase {phase_type.title()} Failed**")
    print(f"Error: {error['error']}")
    
    recovery_options = {
        "retry": f"Retry {phase_type} phase with same context",
        "debug": f"Debug {phase_type} phase with specialist assistance", 
        "rollback": f"Rollback to previous phase and restart",
        "abort": "Abort workflow session"
    }
    
    choice = present_recovery_options(recovery_options)
    
    if choice == "retry":
        return f"Retrying {phase_type} phase..."
    elif choice == "debug":
        return f"Debugging {phase_type} phase with specialist help..."
    elif choice == "rollback":
        return f"Rolling back to previous phase..."
    else:
        return "Workflow aborted by user"
```

## 📊 Workflow Monitoring

### Session Dashboard
```python
def display_workflow_dashboard(session: WorkflowSession) -> None:
    """Display real-time workflow session dashboard."""
    
    print(f"\n📊 **Workflow Dashboard**")
    print(f"Session: {session.session_id}")
    print(f"Phase: {session.phase_id}")
    print(f"Status: {session._determine_status().upper()}")
    
    print(f"\nPhase Progress:")
    phases = ["alignment", "implementation", "verification", "commit"]
    for i, phase in enumerate(phases):
        status = "✅" if phase in session.phases_completed else "⏳" if phase == session.current_phase else "⭕"
        print(f"  {status} Phase {i+1}: {phase.title()}")
    
    print(f"\nState Artifacts:")
    for phase, artifact in session.state_artifacts.items():
        print(f"  📄 {phase.title()}: {artifact.get('status', 'unknown')}")
```

### Performance Metrics
```python
def get_workflow_metrics(session: WorkflowSession) -> Dict:
    """Calculate workflow performance metrics."""
    
    total_time = sum(
        artifact.get("metadata", {}).get("duration_ms", 0)
        for artifact in session.state_artifacts.values()
    )
    
    return {
        "total_duration_ms": total_time,
        "phases_completed": len(session.phases_completed),
        "success_rate": len(session.phases_completed) / 4.0,
        "average_phase_duration": total_time / max(len(session.phases_completed), 1),
        "context_efficiency": "high" if total_time < 3600000 else "medium"  # < 1 hour is high efficiency
    }
```

## 🛡️ Safety & Reliability

### Error Containment
- Each phase runs in isolated context preventing error propagation
- Background tasks with timeouts prevent hanging phases
- State validation ensures clean handoffs between phases

### Rollback Capability  
```python
def rollback_to_phase(session: WorkflowSession, target_phase: str) -> bool:
    """Rollback workflow to specific phase."""
    
    if target_phase not in session.phases_completed:
        print(f"Cannot rollback to {target_phase} - phase not completed")
        return False
    
    # Remove subsequent phases from completed list
    target_index = ["alignment", "implementation", "verification", "commit"].index(target_phase)
    session.phases_completed = session.phases_completed[:target_index + 1]
    
    # Clear state artifacts for subsequent phases
    for phase in ["alignment", "implementation", "verification", "commit"][target_index + 1:]:
        session.state_artifacts.pop(phase, None)
    
    print(f"Rolled back to {target_phase} phase")
    return True
```

---

## 🎯 Mission Complete

**Isolated Phase Orchestrator** provides true context isolation for development workflows, delivering error containment, performance optimization, and reliable phase transitions with complete audit trails and human oversight.