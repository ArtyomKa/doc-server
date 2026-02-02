/**
 * Core type definitions for the dev-phase-orchestrator skill
 */

import { z } from 'zod';

// Phase identifiers and status
export const PhaseIdSchema = z.string().regex(/^\d+\.\d+$/);
export type PhaseId = z.infer<typeof PhaseIdSchema>;

export const PhaseStatusSchema = z.enum(['pending', 'in_progress', 'completed', 'failed', 'cancelled']);
export type PhaseStatus = z.infer<typeof PhaseStatusSchema>;

// Session management
export const SessionIdSchema = z.string().regex(/^phase-\d+\.\d+-\d{8}$/);
export type SessionId = z.infer<typeof SessionIdSchema>;

export const SessionContextSchema = z.object({
  sessionId: SessionIdSchema,
  phaseId: PhaseIdSchema,
  startTime: z.string().datetime(),
  currentPhase: z.enum(['alignment', 'implementation', 'verification', 'commit']),
  status: PhaseStatusSchema,
  metadata: z.record(z.any()).optional(),
});
export type SessionContext = z.infer<typeof SessionContextSchema>;

// Alignment verification types
export const AlignmentIssueSchema = z.object({
  type: z.enum(['task_product_gap', 'plan_product_mismatch', 'task_plan_inconsistency', 'missing_requirement', 'contradiction']),
  description: z.string(),
  severity: z.enum(['critical', 'major', 'minor']),
  affectedComponents: z.array(z.string()),
  suggestedResolution: z.string().optional(),
});
export type AlignmentIssue = z.infer<typeof AlignmentIssueSchema>;

export const AlignmentReportSchema = z.object({
  status: z.enum(['approved', 'rejected', 'needs_review']),
  phaseId: PhaseIdSchema,
  analysisSummary: z.object({
    totalTasks: z.number(),
    alignedTasks: z.number(),
    criticalIssues: z.number(),
    majorIssues: z.number(),
    minorIssues: z.number(),
  }),
  alignmentIssues: z.array(AlignmentIssueSchema),
  approvedTasks: z.array(z.string()),
  scopeBoundaries: z.object({
    includedModules: z.array(z.string()),
    excludedModules: z.array(z.string()),
    filePatterns: z.array(z.string()),
  }),
  decisionLog: z.array(z.object({
    decision: z.string(),
    rationale: z.string(),
    timestamp: z.string().datetime(),
  })),
});
export type AlignmentReport = z.infer<typeof AlignmentReportSchema>;

// Implementation types
export const ImplementationTaskSchema = z.object({
  taskId: z.string(),
  description: z.string(),
  targetFiles: z.array(z.string()),
  testFiles: z.array(z.string()),
  dependencies: z.array(z.string()).optional(),
});
export type ImplementationTask = z.infer<typeof ImplementationTaskSchema>;

export const ImplementationReportSchema = z.object({
  status: z.enum(['completed', 'partial', 'failed']),
  phaseId: PhaseIdSchema,
  branchName: z.string(),
  completedTasks: z.array(ImplementationTaskSchema),
  failedTasks: z.array(ImplementationTaskSchema),
  testResults: z.object({
    totalTests: z.number(),
    passedTests: z.number(),
    failedTests: z.number(),
    coverage: z.number(),
  }),
  implementationNotes: z.array(z.string()).optional(),
});
export type ImplementationReport = z.infer<typeof ImplementationReportSchema>;

// Verification types
export const AcceptanceCriterionSchema = z.object({
  criterionId: z.string(),
  description: z.string(),
  requirements: z.array(z.string()),
  status: z.enum(['met', 'partially_met', 'not_met']),
  evidence: z.string().optional(),
  gapReason: z.string().optional(),
});
export type AcceptanceCriterion = z.infer<typeof AcceptanceCriterionSchema>;

export const VerificationReportSchema = z.object({
  status: z.enum(['passed', 'failed', 'partial']),
  phaseId: PhaseIdSchema,
  acceptanceCriteria: z.array(AcceptanceCriterionSchema),
  testResults: z.object({
    totalTests: z.number(),
    passedTests: z.number(),
    failedTests: z.number(),
    coverage: z.number(),
  }),
  verificationSummary: z.object({
    totalCriteria: z.number(),
    fullyMet: z.number(),
    partiallyMet: z.number(),
    notMet: z.number(),
  }),
  outstandingIssues: z.array(z.string()),
});
export type VerificationReport = z.infer<typeof VerificationReportSchema>;

// Commit types
export const CommitReportSchema = z.object({
  status: z.enum(['committed', 'failed']),
  phaseId: PhaseIdSchema,
  commitHash: z.string().optional(),
  commitMessage: z.string(),
  filesCommitted: z.array(z.string()),
  branchName: z.string(),
  timestamp: z.string().datetime(),
});
export type CommitReport = z.infer<typeof CommitReportSchema>;

// State handoff artifacts
export const StateArtifactSchema = z.object({
  sessionId: SessionIdSchema,
  phaseId: PhaseIdSchema,
  fromPhase: z.enum(['alignment', 'implementation', 'verification', 'commit']),
  toPhase: z.enum(['alignment', 'implementation', 'verification', 'commit']),
  timestamp: z.string().datetime(),
  data: z.record(z.any()),
});
export type StateArtifact = z.infer<typeof StateArtifactSchema>;

// Specialist delegation
export const SpecialistTypeSchema = z.enum(['explorer', 'librarian', 'oracle', 'designer', 'fixer']);
export type SpecialistType = z.infer<typeof SpecialistTypeSchema>;

export const DelegationRequestSchema = z.object({
  specialist: SpecialistTypeSchema,
  task: z.string(),
  context: z.record(z.any()).optional(),
  priority: z.enum(['low', 'medium', 'high']).default('medium'),
  expectedOutput: z.string().optional(),
});
export type DelegationRequest = z.infer<typeof DelegationRequestSchema>;

export const DelegationResultSchema = z.object({
  specialist: SpecialistTypeSchema,
  task: z.string(),
  result: z.any(),
  success: z.boolean(),
  error: z.string().optional(),
  executionTime: z.number(),
});
export type DelegationResult = z.infer<typeof DelegationResultSchema>;

// Human approval gates
export const HumanApprovalSchema = z.object({
  phase: z.enum(['alignment', 'implementation', 'verification', 'commit']),
  action: z.enum(['proceed_to_implementation', 'proceed_to_verification', 'proceed_to_commit', 'proceed_to_completion']),
  timestamp: z.string().datetime(),
  comments: z.string().optional(),
  conditions: z.array(z.string()).optional(),
});
export type HumanApproval = z.infer<typeof HumanApprovalSchema>;

// Configuration
export const OrchestratorConfigSchema = z.object({
  projectRoot: z.string(),
  specsPath: z.string(),
  maxConcurrentDelegations: z.number().default(3),
  humanApprovalRequired: z.boolean().default(true),
  stateRetentionHours: z.number().default(24),
  enableMetrics: z.boolean().default(true),
});
export type OrchestratorConfig = z.infer<typeof OrchestratorConfigSchema>;

// Error handling
export const OrchestratorErrorSchema = z.object({
  phase: z.enum(['alignment', 'implementation', 'verification', 'commit', 'orchestration']),
  errorType: z.enum(['validation_error', 'delegation_error', 'file_error', 'approval_error', 'system_error']),
  message: z.string(),
  details: z.record(z.any()).optional(),
  recoverable: z.boolean(),
  suggestedAction: z.string().optional(),
});
export type OrchestratorError = z.infer<typeof OrchestratorErrorSchema>;

// Export all types for easy importing
export type {
  SessionContext,
  AlignmentIssue,
  AlignmentReport,
  ImplementationTask,
  ImplementationReport,
  AcceptanceCriterion,
  VerificationReport,
  CommitReport,
  StateArtifact,
  DelegationRequest,
  DelegationResult,
  HumanApproval,
  OrchestratorConfig,
  OrchestratorError,
};