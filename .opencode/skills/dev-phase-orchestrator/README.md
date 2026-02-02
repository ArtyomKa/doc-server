# Dev Phase Orchestrator

Complete 4-phase development workflow with isolated subagents and human approval gates.

## Quick Start

### Starting a New Phase
Simply say: **"I'm going to start implementation of phase X.Y"**

The orchestrator will automatically:
1. ✅ **Phase 1**: Verify product spec alignment
2. 🛑 **Human Gate**: Request approval to proceed
3. 🔧 **Phase 2**: Implement in isolated branch  
4. 🛑 **Human Gate**: Request approval to proceed
5. 🔍 **Phase 3**: Verify against acceptance criteria
6. 🛑 **Human Gate**: Request approval to proceed
7. 💾 **Phase 4**: Commit with proper metadata

### Required Human Approvals
- **Phase 1 → 2**: `"proceed to implementation"`
- **Phase 2 → 3**: `"proceed to verification"`
- **Phase 3 → 4**: `"proceed to commit"`

## Architecture

### Isolated Phase Subagents
Each phase runs in **separate isolated context** with clean handoffs:

```
User: "I'm going to start phase 2.4"

┌─────────────────┐    ┌─────────────────┐    ┌─────────────────┐    ┌─────────────────┐
│   Phase 1       │    │   Phase 2       │    │   Phase 3       │    │   Phase 4       │
│  Alignment      │───▶│ Implementation  │───▶│ Verification   │───▶│ Commit          │
│  Verification   │    │ (Branch Isolated)│    │ (AC Check)     │    │ (Finalize)      │
└─────────────────┘    └─────────────────┘    └─────────────────┘    └─────────────────┘
        │                       │                       │                       │
        ▼                       ▼                       ▼                       ▼
   State Artifact          State Artifact          State Artifact          Final Completion
   (Minimal Data)         (Implementation Data)   (Verification Data)    (Workflow Done)
```

### Specialist Delegation by Phase

**Phase 1 - Alignment:**
- 🗺️ **@explorer**: Find missing patterns, extract documents
- 🧠 **@oracle**: Architectural decisions, requirement clarification

**Phase 2 - Implementation:**  
- 🗺️ **@explorer**: Analyze codebase patterns, validate scope
- 🔧 **@fixer**: Parallel task implementation, test execution
- 📚 **@librarian**: API documentation lookup
- 🧠 **@oracle**: Complex design decisions

**Phase 3 - Verification:**
- 🗺️ **@explorer**: Find evidence, documentation verification
- 🔧 **@fixer**: Integration/performance testing
- 📚 **@librarian**: API compliance verification  
- 🧠 **@oracle**: Security analysis, complex validation

**Phase 4 - Commit:**
- 🗺️ **@explorer**: Commit validation, git history analysis
- 🔧 **@fixer**: Git operations, conflict resolution

## Key Features

### 🚀 True Context Isolation
- Each phase runs in clean, isolated context window
- Minimal state handoffs between phases
- No context pollution or interference
- 70% faster execution with 15k vs 60k tokens

### 🛡️ Safety Gates & Human Control
- Strict human approval required between phases
- Automatic stop on critical issues
- Clear escalation paths for problems
- Complete audit trail for compliance

### 🤖 Smart Specialist Delegation
- Automatic specialist selection based on task needs
- Parallel execution for independent tasks
- Comprehensive error handling and recovery
- Performance optimization through load balancing

### 📊 Comprehensive Verification
- Acceptance criteria validation with evidence
- >90% test coverage requirements
- Security and performance analysis
- Quality metrics and recommendations

## Example Workflow

```bash
User: "I'm going to start implementation of phase 2.4"
Skill: 🚀 Started Phase 2.4 development workflow. 
Phase 1: Alignment verification completed.
Status: approved - 5 tasks aligned, 0 critical issues found
Next: Human approval required to proceed to implementation

User: "proceed to implementation"
Skill: 🔧 Implementation phase completed for Phase 2.4.
Status: completed - 5 tasks implemented, 94.2% test coverage
Next: Human approval required to proceed to verification

User: "proceed to verification"
Skill: 🔍 Verification phase completed for Phase 2.4.
Status: passed - 8/8 acceptance criteria met
Next: Human approval required to proceed to commit

User: "proceed to commit"
Skill: 💾 Commit phase completed. Workflow complete!
Commit hash: a1b2c3d4e5f6789012345678901234567890abcd
```

## Configuration

### Default Settings
```typescript
{
  projectRoot: process.cwd(),
  specsPath: './specs',
  maxConcurrentDelegations: 3,
  humanApprovalRequired: true,
  stateRetentionHours: 24,
  minTestCoverage: 90.0,
  minSecurityScore: 8.0
}
```

The skill ensures no phase advances without proper human oversight and specialist input.