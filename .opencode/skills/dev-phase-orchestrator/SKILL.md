---
name: dev-phase-orchestrator
description: |
  Orchestrates complete 4-phase development workflow for doc-server: product spec alignment verification, implementation, acceptance criteria verification, and commit.

  Use this skill when starting work on a new development phase (e.g., "I'm going to start phase 2.4"). The skill ensures proper alignment verification before implementation and thorough verification after implementation.
---

# Development Phase Orchestrator

Automates complete 4-phase development workflow for doc-server project.

## ⚠️ CRITICAL SAFETY PRINCIPLE

**NEVER ADVANCE AUTOMATICALLY TO NEXT WORKFLOW STEP** - Each phase requires explicit human approval before proceeding. The skill will STOP and request guidance if any issues are found that require human attention.

**Human Approval Required Between Each Phase**:
- Phase 1 → 2: "proceed to implementation" 
- Phase 2 → 3: "proceed to verification"
- Phase 3 → 4: "proceed to commit"

## Quick Start

To start a new development phase, simply say: *"I'm going to start implementation of phase X.Y"*

The skill will automatically:
1. Run product spec alignment verification
2. Guide you through implementation  
3. Run acceptance criteria verification
4. Prepare commit

**🤖 Specialist Delegation Strategy**:
- **@explorer**: For discovering unknown patterns, searching codebase, parallel file discovery
- **@librarian**: For API/docs lookup (FastMCP, ChromaDB, sentence-transformers)  
- **@oracle**: For architectural decisions, scope questions, complex debugging
- **@designer**: For UI/UX polish (not applicable for backend phases)
- **@fixer**: For well-defined, parallel implementation tasks

**Delegation Triggers**:
- "Research X": Use @librarian for library/API documentation
- "Find all Y": Use @explorer for codebase-wide searches
- "Decide between A/B": Use @oracle for strategic decisions
- "Implement these N tasks": Use @fixer for parallel execution
- "Complex issue in X": Use @oracle for deep debugging

**⚠️ CRITICAL SAFETY**: Each phase requires explicit human approval before advancing to the next step. The skill will STOP and request guidance if any issues are found.

## Workflow Phases

### Phase 1: Product Spec Alignment Verification (PRE-IMPLEMENTATION GATE)

**Trigger**: User says "I'm going to start implementation of phase X.Y"

**Purpose**: Verify that phase X.Y tasks, product spec, and implementation plan are aligned before any code changes begin.

**Steps**:
1. Read `@specs/doc-server-tasks.md` - Extract phase X.Y tasks
2. Read `@specs/doc-server-product-spec.md` - Extract product-level requirements
3. Read `@specs/doc-server-plan.md` - Extract implementation plan details
4. Cross-reference alignment between all three documents:
   - Do tasks match product spec?
   - Does plan match product spec?
   - Are there any gaps or contradictions?
5. Generate alignment report with findings
6. **⚠️ CRITICAL**: If alignment issues found, **STOP** and request human decision before proceeding

**Output**: Alignment report with status for each requirement and any issues found

**Human Gate**: Must get explicit approval to proceed to Phase 2

---

### Phase 2: Implementation

**Trigger**: Only after alignment verified AND human explicitly confirms "proceed to implementation"

**Steps**:
1. **🔒 BRANCH ISOLATION**: Create dedicated git branch `phase-X.Z` to isolate phase work
   ```bash
   git checkout -b phase-2.4
   ```
2. Implement module(s) according to plan
3. Write tests following project patterns
4. Run tests locally to verify basic functionality
5. Ensure test coverage >90%
6. **⚠️ CRITICAL**: If implementation issues arise, **STOP** and request human guidance

**Output**: Completed implementation ready for verification

**Human Gate**: Must get explicit approval to proceed to Phase 3

---

### Phase 3: Acceptance Criteria Verification

**Trigger**: After implementation is complete AND human explicitly requests verification

**User Request**: *"verify that phase X.Y in @specs/doc-server-tasks.md is implemented according to @specs/doc-server-acceptence.md"*

**Steps**:
1. Read `@specs/doc-server-tasks.md` - Extract phase X.Y tasks
2. Read `@specs/doc-server-acceptence.md` - Extract AC criteria
3. Read implementation files and test files
4. Verify each AC criterion against code:
   - Check if implemented in code
   - Check if tested
   - Mark as ✅ (fully met), ⚠️ (partially met), or ❌ (not met)
5. Run pytest on module
6. Check test coverage (>90%)
7. Generate detailed verification report (✅/⚠️/❌ per AC)
8. **⚠️ CRITICAL**: If verification failures found, **STOP** and request human approval for fixes

**Output**: Detailed verification report with acceptance criteria status, test results, and coverage metrics

**Human Gate**: Must get explicit approval to proceed to Phase 4

---

### Phase 4: Commit

**Trigger**: Only after verification passes AND human explicitly confirms "proceed to commit"

**Steps**:
1. Stage all changed files
2. Create commit message following pattern: "Phase X.Y: [module name] - [summary]"
3. Commit changes
4. Optionally push to remote

**Output**: Committed changes ready for next phase

---

## File References

The skill references these specification files:

- `@specs/doc-server-tasks.md` - Task breakdown and status
- `@specs/doc-server-product-spec.md` - Product requirements and scope
- `@specs/doc-server-plan.md` - Implementation plan details
- `@specs/doc-server-acceptence.md` - Acceptance criteria and test specifications
- `@CODING_STANDARDS.md` - Code style and quality guidelines
- `@AGENTS.md` - Agent instructions and workflow

---

## Implementation Details

### Extract Phase Tasks Pattern

From `@specs/doc-server-tasks.md`, extract tasks for phase X.Y:

```python
def extract_phase_tasks(tasks_content, phase_number):
    # Look for "## Phase {phase_number}:"
    # Extract subsections and task lists
    # Return structured tasks with status
```

### Alignment Verification Pattern

Cross-reference three specification files:

1. **Tasks → Product Spec**: Do tasks fulfill product requirements?
2. **Plan → Product Spec**: Does implementation plan match product decisions?
3. **Tasks → Plan**: Are task breakdowns consistent with implementation approach?

### Acceptance Criteria Pattern

From `@specs/doc-server-acceptence.md`, extract AC-X.Y.Z criteria:

```python
def extract_acceptance_criteria(acceptence_content, phase_number):
    # Look for "#### X.Y.Z" patterns
    # Extract requirements and test specifications
    # Return structured AC list
```

### Verification Status Legend

- ✅ **Fully Met**: Requirement completely implemented and tested
- ⚠️ **Partially Met**: Requirement mostly implemented but missing some aspects
- ❌ **Not Met**: Requirement not implemented or completely broken

### Test Coverage Target

Aim for >90% coverage for critical path components as specified in acceptance criteria.

---

## 🎯 Skill Complete

**`dev-phase-orchestrator` is now ready for use!**

This skill enforces strict human-gated workflow progression and provides specialist delegation guidance for efficient, error-free development phases.