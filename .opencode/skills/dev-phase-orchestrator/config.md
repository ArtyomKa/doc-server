# Development Phase Orchestrator - Configuration

## Environment Configuration

### Required Environment Variables

```bash
# Core Configuration
ORCHESTRATOR_PROJECT_ROOT=/path/to/project
ORCHESTRATOR_SPECS_PATH=/path/to/specs

# Delegation Settings
ORCHESTRATOR_MAX_DELEGATIONS=3
ORCHESTRATOR_DELEGATION_TIMEOUT=30000

# Quality Gates
ORCHESTRATOR_MIN_TEST_COVERAGE=90.0
ORCHESTRATOR_MIN_SECURITY_SCORE=8.0

# Session Management
ORCHESTRATOR_STATE_RETENTION_HOURS=24
ORCHESTRATOR_ENABLE_METRICS=true

# Git Settings
ORCHESTRATOR_DEFAULT_BRANCH=main
ORCHESTRATOR_COMMIT_MESSAGE_TEMPLATE="Phase {phaseId}: {module} - {summary}"
ORCHESTRATOR_BRANCH_ACTION=keep
```

### Configuration File (orchestrator.config.json)

```json
{
  "projectRoot": "/home/user/project",
  "specsPath": "/home/user/project/specs",
  "maxConcurrentDelegations": 3,
  "humanApprovalRequired": true,
  "stateRetentionHours": 24,
  "enableMetrics": true,
  "minTestCoverage": 90.0,
  "minSecurityScore": 8.0,
  "delegation": {
    "timeout": 30000,
    "retryAttempts": 3,
    "retryDelay": 1000
  },
  "git": {
    "defaultBranch": "main",
    "commitMessageTemplate": "Phase {phaseId}: {module} - {summary}",
    "defaultBranchAction": "keep",
    "pushToRemote": false,
    "createTag": false
  },
  "phases": {
    "alignment": {
      "enabled": true,
      "timeout": 300000,
      "requiredApprovals": ["proceed_to_implementation"]
    },
    "implementation": {
      "enabled": true,
      "timeout": 1800000,
      "requiredApprovals": ["proceed_to_verification"],
      "minTestCoverage": 90.0
    },
    "verification": {
      "enabled": true,
      "timeout": 600000,
      "requiredApprovals": ["proceed_to_commit"],
      "minTestCoverage": 85.0
    },
    "commit": {
      "enabled": true,
      "timeout": 120000,
      "requiredApprovals": []
    }
  },
  "specialists": {
    "explorer": {
      "enabled": true,
      "maxConcurrentTasks": 2,
      "timeout": 15000
    },
    "librarian": {
      "enabled": true,
      "maxConcurrentTasks": 1,
      "timeout": 10000
    },
    "oracle": {
      "enabled": true,
      "maxConcurrentTasks": 1,
      "timeout": 30000
    },
    "designer": {
      "enabled": false,
      "maxConcurrentTasks": 1,
      "timeout": 20000
    },
    "fixer": {
      "enabled": true,
      "maxConcurrentTasks": 3,
      "timeout": 25000
    }
  },
  "errorHandling": {
    "maxRetries": 3,
    "retryDelay": 1000,
    "enableEscalation": true,
    "logLevel": "info"
  }
}
```

## Specialist Configuration

### @explorer Configuration
```json
{
  "specialist": "explorer",
  "capabilities": [
    "find_missing_patterns",
    "extract_phase_tasks",
    "analyze_codebase_patterns",
    "validate_scope_boundaries"
  ],
  "triggers": {
    "alignment": ["find_missing_patterns", "extract_phase_tasks"],
    "implementation": ["analyze_codebase_patterns"],
    "verification": ["find_criterion_implementation"],
    "commit": ["validate_commit_message"]
  },
  "timeout": 15000,
  "maxConcurrentTasks": 2
}
```

### @fixer Configuration
```json
{
  "specialist": "fixer",
  "capabilities": [
    "implement_task",
    "run_tests",
    "execute_git_operations",
    "parallel_implementation"
  ],
  "triggers": {
    "alignment": [],
    "implementation": ["implement_task", "run_tests"],
    "verification": ["run_integration_tests"],
    "commit": ["execute_git_commit"]
  },
  "timeout": 25000,
  "maxConcurrentTasks": 3
}
```

### @oracle Configuration
```json
{
  "specialist": "oracle",
  "capabilities": [
    "architectural_decisions",
    "requirement_clarification",
    "security_analysis",
    "complex_validation"
  ],
  "triggers": {
    "alignment": ["architectural_decisions"],
    "implementation": ["architectural_decisions"],
    "verification": ["security_analysis"],
    "commit": []
  },
  "timeout": 30000,
  "maxConcurrentTasks": 1
}
```

## Phase-Specific Configuration

### Phase 1: Alignment
```json
{
  "phase": "alignment",
  "description": "Product spec alignment verification",
  "enabled": true,
  "timeout": 300000,
  "requiredApprovals": ["proceed_to_implementation"],
  "specialists": {
    "explorer": ["find_missing_patterns", "extract_phase_tasks"],
    "oracle": ["architectural_decisions"]
  },
  "outputs": {
    "alignmentReport": true,
    "approvedTasks": true,
    "scopeBoundaries": true,
    "criticalIssues": true
  },
  "errorHandling": {
    "stopOnCritical": true,
    "allowMajorIssues": false,
    "retryCount": 2
  }
}
```

### Phase 2: Implementation
```json
{
  "phase": "implementation",
  "description": "Implementation with branch isolation",
  "enabled": true,
  "timeout": 1800000,
  "requiredApprovals": ["proceed_to_verification"],
  "specialists": {
    "explorer": ["analyze_codebase_patterns"],
    "fixer": ["implement_task", "run_tests"],
    "librarian": ["get_api_docs"],
    "oracle": ["architectural_decisions"]
  },
  "outputs": {
    "implementationReport": true,
    "testResults": true,
    "modifiedFiles": true,
    "qualityMetrics": true
  },
  "qualityGates": {
    "minTestCoverage": 90.0,
    "maxLintErrors": 0,
    "requireAllTestsPass": true
  },
  "git": {
    "createBranch": true,
    "branchPrefix": "phase-",
    "autoStage": true
  }
}
```

### Phase 3: Verification
```json
{
  "phase": "verification",
  "description": "Acceptance criteria verification",
  "enabled": true,
  "timeout": 600000,
  "requiredApprovals": ["proceed_to_commit"],
  "specialists": {
    "explorer": ["find_criterion_implementation"],
    "fixer": ["run_integration_tests"],
    "librarian": ["verify_api_compliance"],
    "oracle": ["security_analysis"]
  },
  "outputs": {
    "verificationReport": true,
    "acceptanceCriteria": true,
    "evidence": true,
    "outstandingIssues": true
  },
  "qualityGates": {
    "minTestCoverage": 85.0,
    "minSecurityScore": 8.0,
    "requireAllACMet": false,
    "allowPartialMet": true
  }
}
```

### Phase 4: Commit
```json
{
  "phase": "commit",
  "description": "Final commit and workflow completion",
  "enabled": true,
  "timeout": 120000,
  "requiredApprovals": [],
  "specialists": {
    "explorer": ["validate_commit_message"],
    "fixer": ["execute_git_commit", "manage_branch"]
  },
  "outputs": {
    "commitReport": true,
    "completionArtifact": true,
    "workflowSummary": true
  },
  "git": {
    "commitMessageTemplate": "Phase {phaseId}: {module} - {summary}",
    "defaultBranchAction": "keep",
    "pushToRemote": false,
    "createTag": false
  }
}
```

## Logging Configuration

```json
{
  "logging": {
    "level": "info",
    "format": "json",
    "outputs": ["console", "file"],
    "file": {
      "path": "./logs/orchestrator.log",
      "maxSize": "10MB",
      "maxFiles": 5
    },
    "categories": {
      "orchestrator": "info",
      "delegation": "debug",
      "phases": "info",
      "errors": "error"
    }
  }
}
```

## Metrics Configuration

```json
{
  "metrics": {
    "enabled": true,
    "interval": 60000,
    "outputs": ["prometheus", "file"],
    "prometheus": {
      "port": 9090,
      "path": "/metrics"
    },
    "file": {
      "path": "./metrics/orchestrator-metrics.json"
    },
    "track": [
      "phase_duration",
      "delegation_success_rate",
      "error_count",
      "human_approval_time"
    ]
  }
}
```

## Security Configuration

```json
{
  "security": {
    "enableAuditLog": true,
    "auditLogPath": "./logs/audit.log",
    "sessionTimeout": 3600000,
    "maxSessionDuration": 86400000,
    "requireAuthentication": false,
    "allowedOrigins": ["*"],
    "rateLimiting": {
      "enabled": true,
      "maxRequests": 100,
      "windowMs": 60000
    }
  }
}
```