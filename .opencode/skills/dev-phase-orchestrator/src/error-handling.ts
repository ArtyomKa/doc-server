/**
 * Comprehensive Error Handling and Validation System
 * Provides robust error handling, validation, and recovery strategies for the orchestrator
 */

import {
  OrchestratorError,
  PhaseId,
  SessionId,
  OrchestratorConfig,
  AlignmentReport,
  ImplementationReport,
  VerificationReport,
  CommitReport,
} from '../types';

/**
 * Error severity levels
 */
export enum ErrorSeverity {
  LOW = 'low',
  MEDIUM = 'medium',
  HIGH = 'high',
  CRITICAL = 'critical'
}

/**
 * Error recovery strategies
 */
export enum RecoveryStrategy {
  RETRY = 'retry',
  FALLBACK = 'fallback',
  ESCALATE = 'escalate',
  ABORT = 'abort',
  MANUAL_INTERVENTION = 'manual_intervention'
}

/**
 * Enhanced error information
 */
export interface EnhancedError extends OrchestratorError {
  id: string;
  timestamp: string;
  severity: ErrorSeverity;
  recoveryStrategy: RecoveryStrategy;
  retryCount: number;
  maxRetries: number;
  context?: any;
  stackTrace?: string;
  suggestedActions?: string[];
  relatedErrors?: string[];
}

/**
 * Validation result
 */
export interface ValidationResult {
  valid: boolean;
  errors: ValidationError[];
  warnings: ValidationWarning[];
  metadata?: any;
}

export interface ValidationError {
  field: string;
  message: string;
  code: string;
  severity: ErrorSeverity;
}

export interface ValidationWarning {
  field: string;
  message: string;
  code: string;
  recommendation?: string;
}

/**
 * Error Handler Class
 */
export class OrchestratorErrorHandler {
  private config: OrchestratorConfig;
  private errorLog: Map<string, EnhancedError[]> = new Map();
  private recoveryHandlers: Map<string, (error: EnhancedError) => Promise<boolean>> = new Map();

  constructor(config: OrchestratorConfig) {
    this.config = config;
    this.initializeRecoveryHandlers();
  }

  /**
   * Handle and process errors
   */
  async handleError(error: Error, phase: string, context?: any): Promise<EnhancedError> {
    const enhancedError = this.enhanceError(error, phase, context);
    
    // Log error
    this.logError(enhancedError);
    
    // Attempt recovery
    const recovered = await this.attemptRecovery(enhancedError);
    
    if (!recovered) {
      // If recovery failed, escalate based on severity
      await this.escalateError(enhancedError);
    }
    
    return enhancedError;
  }

  /**
   * Validate phase inputs
   */
  validatePhaseInput(phase: string, input: any): ValidationResult {
    const errors: ValidationError[] = [];
    const warnings: ValidationWarning[] = [];

    switch (phase) {
      case 'alignment':
        this.validateAlignmentInput(input, errors, warnings);
        break;
      case 'implementation':
        this.validateImplementationInput(input, errors, warnings);
        break;
      case 'verification':
        this.validateVerificationInput(input, errors, warnings);
        break;
      case 'commit':
        this.validateCommitInput(input, errors, warnings);
        break;
      default:
        errors.push({
          field: 'phase',
          message: `Unknown phase: ${phase}`,
          code: 'UNKNOWN_PHASE',
          severity: ErrorSeverity.HIGH
        });
    }

    return {
      valid: errors.length === 0,
      errors,
      warnings
    };
  }

  /**
   * Validate phase outputs
   */
  validatePhaseOutput(phase: string, output: any): ValidationResult {
    const errors: ValidationError[] = [];
    const warnings: ValidationWarning[] = [];

    switch (phase) {
      case 'alignment':
        this.validateAlignmentOutput(output, errors, warnings);
        break;
      case 'implementation':
        this.validateImplementationOutput(output, errors, warnings);
        break;
      case 'verification':
        this.validateVerificationOutput(output, errors, warnings);
        break;
      case 'commit':
        this.validateCommitOutput(output, errors, warnings);
        break;
    }

    return {
      valid: errors.length === 0,
      errors,
      warnings
    };
  }

  /**
   * Get error statistics
   */
  getErrorStatistics(): {
    totalErrors: number;
    errorsByPhase: Record<string, number>;
    errorsBySeverity: Record<ErrorSeverity, number>;
    recoverySuccessRate: number;
  } {
    const allErrors = Array.from(this.errorLog.values()).flat();
    
    const errorsByPhase: Record<string, number> = {};
    const errorsBySeverity: Record<ErrorSeverity, number> = {
      [ErrorSeverity.LOW]: 0,
      [ErrorSeverity.MEDIUM]: 0,
      [ErrorSeverity.HIGH]: 0,
      [ErrorSeverity.CRITICAL]: 0
    };

    allErrors.forEach(error => {
      errorsByPhase[error.phase] = (errorsByPhase[error.phase] || 0) + 1;
      errorsBySeverity[error.severity]++;
    });

    const recoveredErrors = allErrors.filter(e => e.retryCount < e.maxRetries);
    const recoverySuccessRate = allErrors.length > 0 ? recoveredErrors.length / allErrors.length : 0;

    return {
      totalErrors: allErrors.length,
      errorsByPhase,
      errorsBySeverity,
      recoverySuccessRate
    };
  }

  /**
   * Get errors for session
   */
  getSessionErrors(sessionId: SessionId): EnhancedError[] {
    return Array.from(this.errorLog.entries())
      .filter(([key]) => key.includes(sessionId))
      .flatMap(([, errors]) => errors);
  }

  /**
   * Private methods
   */

  private enhanceError(error: Error, phase: string, context?: any): EnhancedError {
    const errorId = this.generateErrorId();
    const severity = this.determineSeverity(error, phase);
    const recoveryStrategy = this.determineRecoveryStrategy(error, severity);
    
    return {
      id: errorId,
      phase: phase as any,
      errorType: this.mapErrorType(error),
      message: error.message,
      timestamp: new Date().toISOString(),
      severity,
      recoveryStrategy,
      retryCount: 0,
      maxRetries: this.getMaxRetries(recoveryStrategy),
      details: context,
      stackTrace: error.stack,
      recoverable: recoveryStrategy !== RecoveryStrategy.ABORT,
      suggestedAction: this.getSuggestedAction(error, phase),
      relatedErrors: []
    };
  }

  private generateErrorId(): string {
    return `err-${Date.now()}-${Math.random().toString(36).substr(2, 9)}`;
  }

  private determineSeverity(error: Error, phase: string): ErrorSeverity {
    const message = error.message.toLowerCase();
    
    // Critical errors
    if (message.includes('critical') || message.includes('security') || message.includes('data loss')) {
      return ErrorSeverity.CRITICAL;
    }
    
    // High severity errors
    if (message.includes('failed') || message.includes('timeout') || message.includes('corruption')) {
      return ErrorSeverity.HIGH;
    }
    
    // Medium severity errors
    if (message.includes('warning') || message.includes('partial') || message.includes('incomplete')) {
      return ErrorSeverity.MEDIUM;
    }
    
    // Low severity errors
    return ErrorSeverity.LOW;
  }

  private determineRecoveryStrategy(error: Error, severity: ErrorSeverity): RecoveryStrategy {
    const message = error.message.toLowerCase();
    
    // Network/temporary issues - retry
    if (message.includes('timeout') || message.includes('network') || message.includes('temporary')) {
      return RecoveryStrategy.RETRY;
    }
    
    // Critical issues - manual intervention
    if (severity === ErrorSeverity.CRITICAL) {
      return RecoveryStrategy.MANUAL_INTERVENTION;
    }
    
    // System failures - escalate
    if (message.includes('system') || message.includes('infrastructure')) {
      return RecoveryStrategy.ESCALATE;
    }
    
    // Feature issues - fallback
    if (message.includes('feature') || message.includes('capability')) {
      return RecoveryStrategy.FALLBACK;
    }
    
    // Default: manual intervention
    return RecoveryStrategy.MANUAL_INTERVENTION;
  }

  private mapErrorType(error: Error): OrchestratorError['errorType'] {
    const message = error.message.toLowerCase();
    
    if (message.includes('validation')) return 'validation_error';
    if (message.includes('git')) return 'file_error';
    if (message.includes('approval')) return 'approval_error';
    if (message.includes('delegation')) return 'delegation_error';
    
    return 'system_error';
  }

  private getMaxRetries(strategy: RecoveryStrategy): number {
    switch (strategy) {
      case RecoveryStrategy.RETRY:
        return 3;
      case RecoveryStrategy.FALLBACK:
        return 1;
      case RecoveryStrategy.ESCALATE:
        return 0;
      case RecoveryStrategy.MANUAL_INTERVENTION:
        return 0;
      case RecoveryStrategy.ABORT:
        return 0;
      default:
        return 1;
    }
  }

  private getSuggestedAction(error: Error, phase: string): string {
    const message = error.message.toLowerCase();
    
    if (message.includes('timeout')) {
      return 'Increase timeout values or check network connectivity';
    }
    
    if (message.includes('permission')) {
      return 'Check file permissions and access rights';
    }
    
    if (message.includes('validation')) {
      return 'Review input data and format requirements';
    }
    
    return 'Review error details and context for resolution guidance';
  }

  private logError(error: EnhancedError): void {
    const sessionKey = error.details?.sessionId || 'global';
    
    if (!this.errorLog.has(sessionKey)) {
      this.errorLog.set(sessionKey, []);
    }
    
    this.errorLog.get(sessionKey)!.push(error);
    
    // Log to console with appropriate level
    const logMessage = `[${error.severity.toUpperCase()}] ${error.phase}: ${error.message}`;
    
    switch (error.severity) {
      case ErrorSeverity.CRITICAL:
        console.error(logMessage, error);
        break;
      case ErrorSeverity.HIGH:
        console.error(logMessage);
        break;
      case ErrorSeverity.MEDIUM:
        console.warn(logMessage);
        break;
      case ErrorSeverity.LOW:
        console.log(logMessage);
        break;
    }
  }

  private async attemptRecovery(error: EnhancedError): Promise<boolean> {
    const handler = this.recoveryHandlers.get(error.recoveryStrategy);
    
    if (!handler) {
      console.warn(`No recovery handler for strategy: ${error.recoveryStrategy}`);
      return false;
    }
    
    try {
      return await handler(error);
    } catch (recoveryError) {
      console.error(`Recovery failed for error ${error.id}:`, recoveryError);
      return false;
    }
  }

  private async escalateError(error: EnhancedError): Promise<void> {
    console.error(`🚨 Escalating error ${error.id}: ${error.message}`);
    
    if (error.severity === ErrorSeverity.CRITICAL) {
      // Critical errors require immediate attention
      console.error('🚨 CRITICAL ERROR - IMMEDIATE ATTENTION REQUIRED');
      console.error('Phase:', error.phase);
      console.error('Error:', error.message);
      console.error('Context:', error.details);
      
      // In a real implementation, this would trigger alerts, notifications, etc.
    }
  }

  /**
   * Validation methods
   */
  private validateAlignmentInput(input: any, errors: ValidationError[], warnings: ValidationWarning[]): void {
    if (!input.phaseId) {
      errors.push({
        field: 'phaseId',
        message: 'Phase ID is required',
        code: 'REQUIRED_FIELD',
        severity: ErrorSeverity.HIGH
      });
    } else if (!/^\d+\.\d+$/.test(input.phaseId)) {
      errors.push({
        field: 'phaseId',
        message: 'Phase ID must be in format X.Y',
        code: 'INVALID_FORMAT',
        severity: ErrorSeverity.HIGH
      });
    }

    if (!input.sessionId) {
      errors.push({
        field: 'sessionId',
        message: 'Session ID is required',
        code: 'REQUIRED_FIELD',
        severity: ErrorSeverity.HIGH
      });
    }
  }

  private validateImplementationInput(input: any, errors: ValidationError[], warnings: ValidationWarning[]): void {
    if (!input.phaseId) {
      errors.push({
        field: 'phaseId',
        message: 'Phase ID is required',
        code: 'REQUIRED_FIELD',
        severity: ErrorSeverity.HIGH
      });
    }

    if (!input.alignmentState) {
      errors.push({
        field: 'alignmentState',
        message: 'Alignment state from Phase 1 is required',
        code: 'REQUIRED_FIELD',
        severity: ErrorSeverity.HIGH
      });
    }

    if (!input.approvedTasks || input.approvedTasks.length === 0) {
      warnings.push({
        field: 'approvedTasks',
        message: 'No approved tasks to implement',
        code: 'NO_TASKS',
        recommendation: 'Check alignment phase results'
      });
    }
  }

  private validateVerificationInput(input: any, errors: ValidationError[], warnings: ValidationWarning[]): void {
    if (!input.implementationState) {
      errors.push({
        field: 'implementationState',
        message: 'Implementation state from Phase 2 is required',
        code: 'REQUIRED_FIELD',
        severity: ErrorSeverity.HIGH
      });
    }

    if (!input.branchName) {
      errors.push({
        field: 'branchName',
        message: 'Branch name is required',
        code: 'REQUIRED_FIELD',
        severity: ErrorSeverity.HIGH
      });
    }

    if (!input.modifiedFiles || input.modifiedFiles.length === 0) {
      warnings.push({
        field: 'modifiedFiles',
        message: 'No modified files found for verification',
        code: 'NO_FILES',
        recommendation: 'Check implementation phase results'
      });
    }
  }

  private validateCommitInput(input: any, errors: ValidationError[], warnings: ValidationWarning[]): void {
    if (!input.verificationState) {
      errors.push({
        field: 'verificationState',
        message: 'Verification state from Phase 3 is required',
        code: 'REQUIRED_FIELD',
        severity: ErrorSeverity.HIGH
      });
    }

    if (!input.branchName) {
      errors.push({
        field: 'branchName',
        message: 'Branch name is required',
        code: 'REQUIRED_FIELD',
        severity: ErrorSeverity.HIGH
      });
    }

    const verificationResults = input.verificationState?.data;
    if (verificationResults && verificationResults.status === 'failed') {
      errors.push({
        field: 'verificationResults',
        message: 'Cannot commit failed verification results',
        code: 'VERIFICATION_FAILED',
        severity: ErrorSeverity.CRITICAL
      });
    }
  }

  private validateAlignmentOutput(output: AlignmentReport, errors: ValidationError[], warnings: ValidationWarning[]): void {
    if (!output.status) {
      errors.push({
        field: 'status',
        message: 'Alignment status is required',
        code: 'REQUIRED_FIELD',
        severity: ErrorSeverity.HIGH
      });
    }

    if (output.status === 'rejected') {
      errors.push({
        field: 'status',
        message: 'Alignment rejected - critical issues found',
        code: 'ALIGNMENT_REJECTED',
        severity: ErrorSeverity.CRITICAL
      });
    }

    if (output.analysisSummary?.criticalIssues && output.analysisSummary.criticalIssues > 0) {
      errors.push({
        field: 'analysisSummary',
        message: `${output.analysisSummary.criticalIssues} critical alignment issues found`,
        code: 'CRITICAL_ISSUES',
        severity: ErrorSeverity.CRITICAL
      });
    }
  }

  private validateImplementationOutput(output: ImplementationReport, errors: ValidationError[], warnings: ValidationWarning[]): void {
    if (output.testResults.coverage < 90) {
      if (output.testResults.coverage < 80) {
        errors.push({
          field: 'testResults.coverage',
          message: `Test coverage ${output.testResults.coverage}% is critically low`,
          code: 'LOW_COVERAGE',
          severity: ErrorSeverity.CRITICAL
        });
      } else {
        warnings.push({
          field: 'testResults.coverage',
          message: `Test coverage ${output.testResults.coverage}% is below recommended 90%`,
          code: 'LOW_COVERAGE',
          recommendation: 'Add more tests to improve coverage'
        });
      }
    }

    if (output.failedTasks && output.failedTasks.length > 0) {
      errors.push({
        field: 'failedTasks',
        message: `${output.failedTasks.length} tasks failed to implement`,
        code: 'FAILED_TASKS',
        severity: ErrorSeverity.HIGH
      });
    }
  }

  private validateVerificationOutput(output: VerificationReport, errors: ValidationError[], warnings: ValidationWarning[]): void {
    if (output.verificationSummary.notMet > 0) {
      errors.push({
        field: 'verificationSummary.notMet',
        message: `${output.verificationSummary.notMet} acceptance criteria not met`,
        code: 'UNMET_CRITERIA',
        severity: ErrorSeverity.HIGH
      });
    }

    if (output.testResults.coverage < 90) {
      warnings.push({
        field: 'testResults.coverage',
        message: `Test coverage ${output.testResults.coverage}% could be improved`,
        code: 'COVERAGE_IMPROVEMENT',
        recommendation: 'Consider adding more edge case tests'
      });
    }

    if (output.outstandingIssues && output.outstandingIssues.length > 5) {
      warnings.push({
        field: 'outstandingIssues',
        message: `${output.outstandingIssues.length} outstanding issues found`,
        code: 'MANY_ISSUES',
        recommendation: 'Consider addressing issues before commit'
      });
    }
  }

  private validateCommitOutput(output: CommitReport, errors: ValidationError[], warnings: ValidationWarning[]): void {
    if (output.status === 'failed') {
      errors.push({
        field: 'status',
        message: 'Commit operation failed',
        code: 'COMMIT_FAILED',
        severity: ErrorSeverity.HIGH
      });
    }

    if (!output.commitHash) {
      errors.push({
        field: 'commitHash',
        message: 'Commit hash is missing',
        code: 'MISSING_HASH',
        severity: ErrorSeverity.HIGH
      });
    }

    if (!output.filesCommitted || output.filesCommitted.length === 0) {
      warnings.push({
        field: 'filesCommitted',
        message: 'No files were committed',
        code: 'NO_FILES',
        recommendation: 'Verify files were staged before commit'
      });
    }
  }

  /**
   * Initialize recovery handlers
   */
  private initializeRecoveryHandlers(): void {
    // Retry handler
    this.recoveryHandlers.set(RecoveryStrategy.RETRY, async (error) => {
      if (error.retryCount >= error.maxRetries) {
        return false;
      }
      
      console.log(`🔄 Retrying operation for error ${error.id} (attempt ${error.retryCount + 1}/${error.maxRetries})`);
      error.retryCount++;
      
      // Simulate retry delay
      await new Promise(resolve => setTimeout(resolve, 1000 * error.retryCount));
      
      // In a real implementation, this would retry the actual failed operation
      return Math.random() > 0.3; // 70% success rate for demo
    });

    // Fallback handler
    this.recoveryHandlers.set(RecoveryStrategy.FALLBACK, async (error) => {
      console.log(`🔄 Applying fallback strategy for error ${error.id}`);
      
      // In a real implementation, this would apply fallback logic
      return true;
    });

    // Escalate handler
    this.recoveryHandlers.set(RecoveryStrategy.ESCALATE, async (error) => {
      console.log(`📢 Escalating error ${error.id} to higher authority`);
      
      // In a real implementation, this would trigger escalation procedures
      return false; // Escalation means automatic recovery failed
    });

    // Manual intervention handler
    this.recoveryHandlers.set(RecoveryStrategy.MANUAL_INTERVENTION, async (error) => {
      console.log(`👤 Manual intervention required for error ${error.id}`);
      
      // In a real implementation, this would notify human operators
      return false;
    });

    // Abort handler
    this.recoveryHandlers.set(RecoveryStrategy.ABORT, async (error) => {
      console.log(`🛑 Aborting operation due to error ${error.id}`);
      
      // In a real implementation, this would clean up and abort
      return false;
    });
  }
}

/**
 * Create global error handler instance
 */
export let globalErrorHandler: OrchestratorErrorHandler;

export function initializeErrorHandler(config: OrchestratorConfig): void {
  globalErrorHandler = new OrchestratorErrorHandler(config);
}