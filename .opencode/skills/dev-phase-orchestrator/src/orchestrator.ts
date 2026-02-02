/**
 * Main Dev Phase Orchestrator
 * Coordinates 4-phase development workflow with isolated subagents and state handoffs
 */

import * as fs from 'fs/promises';
import * as path from 'path';
import { z } from 'zod';
import {
  PhaseId,
  SessionId,
  SessionContext,
  HumanApproval,
  StateArtifact,
  OrchestratorConfig,
  OrchestratorError,
  AlignmentReport,
  ImplementationReport,
  VerificationReport,
  CommitReport,
  ScopeBoundaries,
} from './types';
import { PhaseAlignmentAgent, AlignmentContext } from './agents/alignment';
import { PhaseImplementationAgent, ImplementationContext } from './agents/implementation';
import { PhaseVerificationAgent, VerificationContext } from './agents/verification';
import { PhaseCommitAgent, CommitContext } from './agents/commit';

/**
 * Orchestrator state management
 */
interface OrchestratorState {
  currentSession: SessionContext | null;
  phaseHistory: StateArtifact[];
  pendingApprovals: HumanApproval[];
}

/**
 * Main Dev Phase Orchestrator Class
 */
export class DevPhaseOrchestrator {
  private config: OrchestratorConfig;
  private state: OrchestratorState;
  
  // Phase agents
  private alignmentAgent: PhaseAlignmentAgent;
  private implementationAgent: PhaseImplementationAgent;
  private verificationAgent: PhaseVerificationAgent;
  private commitAgent: PhaseCommitAgent;

  constructor(config: OrchestratorConfig) {
    this.config = config;
    this.state = {
      currentSession: null,
      phaseHistory: [],
      pendingApprovals: []
    };

    // Initialize phase agents
    this.alignmentAgent = new PhaseAlignmentAgent({
      specsPath: this.config.specsPath,
      projectRoot: this.config.projectRoot
    });

    this.implementationAgent = new PhaseImplementationAgent({
      projectRoot: this.config.projectRoot,
      specsPath: this.config.specsPath,
      minTestCoverage: 90.0
    });

    this.verificationAgent = new PhaseVerificationAgent({
      projectRoot: this.config.projectRoot,
      specsPath: this.config.specsPath,
      minTestCoverage: 90.0,
      minSecurityScore: 8.0
    });

    this.commitAgent = new PhaseCommitAgent({
      projectRoot: this.config.projectRoot,
      commitMessageTemplate: 'Phase {phaseId}: {module} - {summary}',
      defaultBranchAction: 'keep'
    });
  }

  /**
   * Start a new development phase workflow
   * Triggered by user saying: "I'm going to start implementation of phase X.Y"
   */
  async startPhase(phaseId: PhaseId, userIntent?: string): Promise<{
    sessionId: SessionId;
    phase: string;
    status: string;
    nextAction: string;
  }> {
    console.log(`🚀 Starting development workflow for Phase ${phaseId}`);

    try {
      // Validate phase ID format
      if (!/^\d+\.\d+$/.test(phaseId)) {
        throw new Error(`Invalid phase ID format: ${phaseId}. Expected format: X.Y`);
      }

      // Create session
      const sessionId = this.generateSessionId(phaseId);
      const sessionContext: SessionContext = {
        sessionId,
        phaseId,
        startTime: new Date().toISOString(),
        currentPhase: 'alignment',
        status: 'pending'
      };

      this.state.currentSession = sessionContext;

      // Start Phase 1: Alignment
      console.log(`📋 Phase 1: Starting product spec alignment verification`);
      
      const alignmentResult = await this.executePhase1Alignment(phaseId, sessionId);

      // Store phase result
      const alignmentArtifact: StateArtifact = {
        sessionId,
        phaseId,
        fromPhase: 'alignment',
        toPhase: 'implementation',
        timestamp: new Date().toISOString(),
        data: alignmentResult
      };
      
      this.state.phaseHistory.push(alignmentArtifact);

      // Determine next action based on alignment result
      const nextAction = this.determineNextAction(alignmentResult);

      return {
        sessionId,
        phase: 'alignment',
        status: alignmentResult.status,
        nextAction
      };

    } catch (error) {
      this.handleError(error as Error, 'orchestration', { phaseId });
      throw error;
    }
  }

  /**
   * Proceed to implementation phase
   * Requires human approval: "proceed to implementation"
   */
  async proceedToImplementation(sessionId: SessionId, approval: HumanApproval): Promise<{
    phase: string;
    status: string;
    result: ImplementationReport;
    nextAction: string;
  }> {
    console.log(`🔧 Phase 2: Starting implementation for session ${sessionId}`);

    try {
      // Validate approval
      this.validateHumanApproval(approval, 'proceed_to_implementation');

      // Get alignment state
      const alignmentState = this.getLatestPhaseState(sessionId, 'alignment');
      if (!alignmentState) {
        throw new Error(`No alignment state found for session ${sessionId}`);
      }

      const alignmentResult = alignmentState.data as AlignmentReport;

      // Execute Phase 2: Implementation
      const implementationResult = await this.executePhase2Implementation(
        sessionId,
        this.state.currentSession!.phaseId,
        alignmentResult
      );

      // Store phase result
      const implementationArtifact: StateArtifact = {
        sessionId,
        phaseId: this.state.currentSession!.phaseId,
        fromPhase: 'implementation',
        toPhase: 'verification',
        timestamp: new Date().toISOString(),
        data: implementationResult
      };
      
      this.state.phaseHistory.push(implementationArtifact);

      // Determine next action
      const nextAction = this.determineNextAction(implementationResult);

      return {
        phase: 'implementation',
        status: implementationResult.status,
        result: implementationResult,
        nextAction
      };

    } catch (error) {
      this.handleError(error as Error, 'implementation', { sessionId });
      throw error;
    }
  }

  /**
   * Proceed to verification phase
   * Requires human approval: "proceed to verification"
   */
  async proceedToVerification(sessionId: SessionId, approval: HumanApproval): Promise<{
    phase: string;
    status: string;
    result: VerificationReport;
    nextAction: string;
  }> {
    console.log(`🔍 Phase 3: Starting verification for session ${sessionId}`);

    try {
      // Validate approval
      this.validateHumanApproval(approval, 'proceed_to_verification');

      // Get implementation state
      const implementationState = this.getLatestPhaseState(sessionId, 'implementation');
      if (!implementationState) {
        throw new Error(`No implementation state found for session ${sessionId}`);
      }

      const implementationResult = implementationState.data as ImplementationReport;

      // Execute Phase 3: Verification
      const verificationResult = await this.executePhase3Verification(
        sessionId,
        this.state.currentSession!.phaseId,
        implementationResult
      );

      // Store phase result
      const verificationArtifact: StateArtifact = {
        sessionId,
        phaseId: this.state.currentSession!.phaseId,
        fromPhase: 'verification',
        toPhase: 'commit',
        timestamp: new Date().toISOString(),
        data: verificationResult
      };
      
      this.state.phaseHistory.push(verificationArtifact);

      // Determine next action
      const nextAction = this.determineNextAction(verificationResult);

      return {
        phase: 'verification',
        status: verificationResult.status,
        result: verificationResult,
        nextAction
      };

    } catch (error) {
      this.handleError(error as Error, 'verification', { sessionId });
      throw error;
    }
  }

  /**
   * Proceed to commit phase
   * Requires human approval: "proceed to commit"
   */
  async proceedToCommit(sessionId: SessionId, approval: HumanApproval, commitOptions?: any): Promise<{
    phase: string;
    status: string;
    result: CommitReport;
    workflowComplete: boolean;
  }> {
    console.log(`💾 Phase 4: Starting commit for session ${sessionId}`);

    try {
      // Validate approval
      this.validateHumanApproval(approval, 'proceed_to_commit');

      // Get verification state
      const verificationState = this.getLatestPhaseState(sessionId, 'verification');
      if (!verificationState) {
        throw new Error(`No verification state found for session ${sessionId}`);
      }

      const verificationResult = verificationState.data as VerificationReport;

      // Execute Phase 4: Commit
      const commitResult = await this.executePhase4Commit(
        sessionId,
        this.state.currentSession!.phaseId,
        verificationResult,
        commitOptions
      );

      // Store phase result
      const commitArtifact: StateArtifact = {
        sessionId,
        phaseId: this.state.currentSession!.phaseId,
        fromPhase: 'commit',
        toPhase: 'completed',
        timestamp: new Date().toISOString(),
        data: commitResult
      };
      
      this.state.phaseHistory.push(commitArtifact);

      // Mark workflow as complete
      if (this.state.currentSession) {
        this.state.currentSession.status = 'completed';
      }

      return {
        phase: 'commit',
        status: commitResult.status,
        result: commitResult,
        workflowComplete: true
      };

    } catch (error) {
      this.handleError(error as Error, 'commit', { sessionId });
      throw error;
    }
  }

  /**
   * Execute Phase 1: Alignment in isolated context
   */
  private async executePhase1Alignment(phaseId: PhaseId, sessionId: SessionId): Promise<AlignmentReport> {
    const context: AlignmentContext = {
      phaseId,
      sessionId
    };

    // Execute in isolated context (background task)
    const result = await this.executeInIsolatedContext(
      'alignment',
      () => this.alignmentAgent.execute(context)
    );

    return result;
  }

  /**
   * Execute Phase 2: Implementation in isolated context
   */
  private async executePhase2Implementation(
    sessionId: SessionId,
    phaseId: PhaseId,
    alignmentResult: AlignmentReport
  ): Promise<ImplementationReport> {
    const context: ImplementationContext = {
      phaseId,
      sessionId,
      alignmentState: this.getLatestPhaseState(sessionId, 'alignment')!,
      approvedTasks: alignmentResult.approvedTasks,
      scopeBoundaries: alignmentResult.scopeBoundaries
    };

    // Execute in isolated context (background task)
    const result = await this.executeInIsolatedContext(
      'implementation',
      () => this.implementationAgent.execute(context)
    );

    return result;
  }

  /**
   * Execute Phase 3: Verification in isolated context
   */
  private async executePhase3Verification(
    sessionId: SessionId,
    phaseId: PhaseId,
    implementationResult: ImplementationReport
  ): Promise<VerificationReport> {
    const context: VerificationContext = {
      phaseId,
      sessionId,
      implementationState: this.getLatestPhaseState(sessionId, 'implementation')!,
      branchName: implementationResult.branchName,
      modifiedFiles: implementationResult.completedTasks.flatMap(t => t.targetFiles),
      implementationTasks: implementationResult.completedTasks
    };

    // Execute in isolated context (background task)
    const result = await this.executeInIsolatedContext(
      'verification',
      () => this.verificationAgent.execute(context)
    );

    return result;
  }

  /**
   * Execute Phase 4: Commit in isolated context
   */
  private async executePhase4Commit(
    sessionId: SessionId,
    phaseId: PhaseId,
    verificationResult: VerificationReport,
    commitOptions?: any
  ): Promise<CommitReport> {
    const context: CommitContext = {
      phaseId,
      sessionId,
      verificationState: this.getLatestPhaseState(sessionId, 'verification')!,
      branchName: `phase-${phaseId}`,
      verificationResults: verificationResult,
      modifiedFiles: this.extractModifiedFiles(verificationResult)
    };

    // Execute in isolated context (background task)
    const result = await this.executeInIsolatedContext(
      'commit',
      () => this.commitAgent.execute(context, commitOptions)
    );

    return result;
  }

  /**
   * Execute function in isolated context using background task
   */
  private async executeInIsolatedContext<T>(
    phase: string,
    executeFn: () => Promise<T>
  ): Promise<T> {
    console.log(`🔄 Executing ${phase} in isolated context`);
    
    // In a real implementation, this would use background tasks for true isolation
    // For now, we'll simulate isolation by executing the function directly
    
    const startTime = Date.now();
    
    try {
      const result = await executeFn();
      const executionTime = Date.now() - startTime;
      
      console.log(`✅ ${phase} completed in ${executionTime}ms`);
      return result;
      
    } catch (error) {
      const executionTime = Date.now() - startTime;
      console.error(`❌ ${phase} failed after ${executionTime}ms:`, error);
      throw error;
    }
  }

  /**
   * Helper methods
   */
  private generateSessionId(phaseId: PhaseId): SessionId {
    const date = new Date().toISOString().slice(0, 10).replace(/-/g, '');
    return `phase-${phaseId}-${date}`;
  }

  private validateHumanApproval(approval: HumanApproval, expectedAction: string): void {
    if (approval.action !== expectedAction) {
      throw new Error(`Invalid approval action. Expected: ${expectedAction}, Received: ${approval.action}`);
    }
    
    if (!approval.timestamp) {
      throw new Error(`Approval timestamp is required`);
    }
  }

  private getLatestPhaseState(sessionId: SessionId, fromPhase: string): StateArtifact | null {
    return this.state.phaseHistory
      .filter(artifact => artifact.sessionId === sessionId && artifact.fromPhase === fromPhase)
      .pop() || null;
  }

  private determineNextAction(result: any): string {
    switch (result.status) {
      case 'approved':
      case 'completed':
      case 'passed':
        return 'Human approval required to proceed to next phase';
        
      case 'needs_review':
      case 'partial':
        return 'Review issues and provide approval to proceed';
        
      case 'rejected':
      case 'failed':
        return 'Address critical issues before proceeding';
        
      default:
        return 'Review result and determine next steps';
    }
  }

  private extractModifiedFiles(verificationResult: VerificationReport): string[] {
    // Extract file paths from acceptance criteria evidence
    const files = new Set<string>();
    
    verificationResult.acceptanceCriteria.forEach(ac => {
      if (ac.evidence) {
        const fileMatches = ac.evidence.match(/(\S+\.(py|js|ts|md))/g);
        if (fileMatches) {
          fileMatches.forEach(file => files.add(file));
        }
      }
    });
    
    return Array.from(files);
  }

  private handleError(error: Error, phase: string, context: any): void {
    const orchestratorError: OrchestratorError = {
      phase: phase as any,
      errorType: 'system_error',
      message: error.message,
      details: context,
      recoverable: false,
      suggestedAction: 'Review error details and retry workflow'
    };

    console.error(`🚨 Orchestrator Error in ${phase}:`, orchestratorError);
    
    // Store error in state for debugging
    if (this.state.currentSession) {
      this.state.pendingApprovals.push({
        phase: phase as any,
        action: 'error' as any,
        timestamp: new Date().toISOString(),
        comments: `Error: ${error.message}`
      });
    }
  }

  /**
   * Public methods for workflow management
   */
  getCurrentSession(): SessionContext | null {
    return this.state.currentSession;
  }

  getPhaseHistory(sessionId?: SessionId): StateArtifact[] {
    if (sessionId) {
      return this.state.phaseHistory.filter(artifact => artifact.sessionId === sessionId);
    }
    return this.state.phaseHistory;
  }

  getPendingApprovals(): HumanApproval[] {
    return this.state.pendingApprovals;
  }

  async saveState(filePath?: string): Promise<void> {
    const statePath = filePath || path.join(this.config.projectRoot, '.orchestrator-state.json');
    await fs.writeFile(statePath, JSON.stringify({
      currentSession: this.state.currentSession,
      phaseHistory: this.state.phaseHistory,
      pendingApprovals: this.state.pendingApprovals
    }, null, 2));
  }

  async loadState(filePath?: string): Promise<void> {
    const statePath = filePath || path.join(this.config.projectRoot, '.orchestrator-state.json');
    
    try {
      const stateData = await fs.readFile(statePath, 'utf8');
      const loadedState = JSON.parse(stateData);
      
      this.state = {
        currentSession: loadedState.currentSession,
        phaseHistory: loadedState.phaseHistory || [],
        pendingApprovals: loadedState.pendingApprovals || []
      };
    } catch (error) {
      console.warn('Could not load orchestrator state, starting fresh');
    }
  }
}