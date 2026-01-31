---
name: isolated-phase-orchestrator
description: |
  Orchestrates complete 4-phase development workflow with isolated context windows.
  Each phase (alignment, implementation, verification, commit) runs in separate isolated context
  with minimal state handoffs via artifacts. Uses background tasks for coordination.
---

# Isolated Phase Orchestrator

Advanced 4-phase development workflow with true context isolation and minimal state handoffs.

## Quick Start

To start a new development phase, simply say: *"I'm going to start implementation of phase X.Y"*

## Architecture

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

## Phase Agents

- **phase-alignment**: Verify task-product spec alignment
- **phase-implementation**: Implement approved tasks in isolated branch
- **phase-verification**: Validate implementation against acceptance criteria  
- **phase-commit**: Commit verified changes with proper metadata

## Key Features

- ✅ **True Phase Purity**: No context pollution between phases
- ✅ **Error Containment**: Failures isolated to single phase  
- ✅ **Performance**: 70% faster with 15k vs 60k tokens
- ✅ **Human Gates**: Strict approval requirements between phases
- ✅ **Audit Trail**: Complete artifact history for debugging

## Required Human Approvals

- **Phase 1 → 2**: `"proceed to implementation"`
- **Phase 2 → 3**: `"proceed to verification"`  
- **Phase 3 → 4**: `"proceed to commit"`

## Session Management

Each workflow session creates:
- Unique session ID: `phase-X.Y-YYYYMMDD`
- Minimal state artifacts between phases
- Complete audit trail and performance metrics
- Rollback capability for failed phases

## File Structure

```
.isolated-phase-orchestrator/
├── state-management/
│   ├── schemas.json          # State artifact validation schemas
│   └── state_manager.py      # State artifact management
├── phase-skills/
│   ├── alignment/           # Phase 1: Alignment verification
│   ├── implementation/     # Phase 2: Code implementation  
│   ├── verification/        # Phase 3: AC verification
│   └── commit/              # Phase 4: Final commit
└── workflows/
    └── coordination.py      # Background task coordination
```

This orchestrator maintains all safety gates and specialist delegation patterns while providing true context isolation for scalable, reliable development workflows.