# Dev-Phase-Orchestrator Skill

## Usage

This skill automatically triggers when you say: *"I'm going to start implementation of phase X.Y"*

## What It Does

1. **Phase 1**: Product spec alignment verification (STOPS if issues found)
2. **Phase 2**: Implementation in isolated git branch (STOPS for issues)  
3. **Phase 3**: Acceptance criteria verification (STOPS for failures)
4. **Phase 4**: Commit (only after explicit approval)

## Safety Features

- ✅ **Human Gates**: Never advances without explicit approval
- 🛑 **Auto-Stop**: Halts on any issues requiring attention
- 🌿 **Branch Isolation**: Work done in `phase-X.Z` branches
- 🤖 **Smart Delegation**: Routes tasks to appropriate specialists

## Required Human Commands

- `"proceed to implementation"` - After alignment verification
- `"proceed to verification"` - After implementation complete
- `"proceed to commit"` - After verification passes

## Example Workflow

```bash
User: "I'm going to start implementation of phase 2.4"
Skill: → Runs Phase 1 alignment verification
User: "proceed to implementation"  
Skill: → Creates phase-2.4 branch, guides implementation
User: "proceed to verification"
Skill: → Runs AC verification, generates report
User: "proceed to commit"
Skill: → Stages and commits changes
```

The skill ensures no phase advances without proper human oversight and specialist input.