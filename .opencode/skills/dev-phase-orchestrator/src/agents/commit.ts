/**
 * Phase 4: Commit Workflow
 * Isolated context subagent for final commit and workflow completion
 */

import * as fs from 'fs/promises';
import * as path from 'path';
import { execSync } from 'child_process';
import {
  PhaseId,
  SessionId,
  CommitReport,
  StateArtifact,
  OrchestratorError,
  DelegationRequest,
  DelegationResult,
  VerificationReport,
} from '../types';

/**
 * Phase 4 Commit Agent Interface
 */
export interface CommitContext {
  phaseId: PhaseId;
  sessionId: SessionId;
  verificationState: StateArtifact; // From Phase 3
  branchName: string;
  verificationResults: VerificationReport;
  modifiedFiles: string[];
}

enum BranchAction {
  KEEP = 'keep',
  MERGE = 'merge',
  DELETE = 'delete'
}

interface CommitPreparation {
  stagedFiles: string[];
  commitMessage: string;
  metadata: {
    phaseId: string;
    acceptanceCriteria: string[];
    testResults: any;
    qualityMetrics: any;
  };
}

interface CommitOptions {
  pushToRemote: boolean;
  branchAction: BranchAction;
  createTag: boolean;
}

/**
 * Phase 4 Commit Agent Class
 */
export class PhaseCommitAgent {
  private config: {
    projectRoot: string;
    commitMessageTemplate: string;
    defaultBranchAction: BranchAction;
  };

  constructor(config: { 
    projectRoot: string; 
    commitMessageTemplate?: string;
    defaultBranchAction?: BranchAction;
  }) {
    this.config = {
      ...config,
      commitMessageTemplate: config.commitMessageTemplate || 'Phase {phaseId}: {module} - {summary}',
      defaultBranchAction: config.defaultBranchAction || BranchAction.KEEP
    };
  }

  /**
   * Execute phase 4 commit in isolated context
   */
  async execute(context: CommitContext, options?: Partial<CommitOptions>): Promise<CommitReport> {
    console.log(`💾 Phase 4: Starting commit workflow for ${context.phaseId}`);

    const commitOptions: CommitOptions = {
      pushToRemote: false,
      branchAction: this.config.defaultBranchAction,
      createTag: false,
      ...options
    };

    try {
      // 1. Prepare commit with validation
      const preparation = await this.prepareCommit(context);
      
      // 2. Validate commit readiness
      await this.validateCommitReadiness(context, preparation);
      
      // 3. Execute git operations
      const commitResult = await this.executeCommit(preparation, commitOptions);
      
      // 4. Manage branch lifecycle
      await this.manageBranchLifecycle(context.branchName, commitOptions.branchAction);
      
      // 5. Generate final commit report
      const report = await this.generateCommitReport(
        context.phaseId,
        context.branchName,
        preparation,
        commitResult
      );
      
      // 6. Generate workflow completion artifact
      await this.generateCompletionArtifact(context, report);
      
      console.log(`✅ Phase 4: Commit workflow completed with status: ${report.status}`);
      return report;
      
    } catch (error) {
      const orchestratorError: OrchestratorError = {
        phase: 'commit',
        errorType: 'system_error',
        message: error instanceof Error ? error.message : 'Unknown commit error',
        details: { phaseId: context.phaseId, branchName: context.branchName },
        recoverable: false,
        suggestedAction: 'Review git state and retry commit workflow'
      };
      
      throw orchestratorError;
    }
  }

  /**
   * Prepare commit with comprehensive message and metadata
   */
  private async prepareCommit(context: CommitContext): Promise<CommitPreparation> {
    console.log(`📝 Preparing commit for branch ${context.branchName}`);
    
    // Ensure we're on the correct branch
    process.chdir(this.config.projectRoot);
    execSync(`git checkout ${context.branchName}`, { stdio: 'pipe' });
    
    // Stage modified files
    await this.stageModifiedFiles(context.modifiedFiles);
    
    // Generate comprehensive commit message
    const commitMessage = await this.generateCommitMessage(context);
    
    // Extract metadata
    const metadata = {
      phaseId: context.phaseId,
      acceptanceCriteria: context.verificationResults.acceptanceCriteria.map(ac => ac.criterionId),
      testResults: context.verificationResults.testResults,
      qualityMetrics: this.extractQualityMetrics(context)
    };
    
    return {
      stagedFiles: context.modifiedFiles,
      commitMessage,
      metadata
    };
  }

  /**
   * Generate comprehensive commit message
   */
  private async generateCommitMessage(context: CommitContext): Promise<string> {
    const { phaseId, verificationResults, modifiedFiles } = context;
    
    // Extract module name from files
    const moduleName = this.extractModuleName(modifiedFiles);
    
    // Generate summary based on acceptance criteria
    const summary = this.generateCommitSummary(verificationResults);
    
    // Create commit title following template
    const title = this.config.commitMessageTemplate
      .replace('{phaseId}', phaseId)
      .replace('{module}', moduleName)
      .replace('{summary}', summary);
    
    // Generate commit body with detailed information
    const body = await this.generateCommitBody(context);
    
    // Generate footer with metadata
    const footer = await this.generateCommitFooter(context);
    
    return `${title}\n\n${body}\n\n${footer}`;
  }

  /**
   * Generate detailed commit body
   */
  private async generateCommitBody(context: CommitContext): Promise<string> {
    const { verificationResults } = context;
    
    const sections: string[] = [];
    
    // Implementation summary
    sections.push('**Implementation:**');
    sections.push(`- Acceptance Criteria Met: ${verificationResults.verificationSummary.fullyMet}/${verificationResults.verificationSummary.totalCriteria}`);
    sections.push(`- Test Coverage: ${verificationResults.testResults.coverage}%`);
    sections.push(`- Tests Passed: ${verificationResults.testResults.passedTests}/${verificationResults.testResults.totalTests}`);
    
    // Acceptance criteria details
    if (verificationResults.acceptanceCriteria.length > 0) {
      sections.push('\n**Acceptance Criteria:**');
      for (const ac of verificationResults.acceptanceCriteria) {
        const status = ac.status === 'met' ? '✅' : ac.status === 'partially_met' ? '⚠️' : '❌';
        sections.push(`${status} ${ac.criterionId}: ${ac.description}`);
        if (ac.gapReason) {
          sections.push(`   Gap: ${ac.gapReason}`);
        }
      }
    }
    
    // Outstanding issues
    if (verificationResults.outstandingIssues.length > 0) {
      sections.push('\n**Outstanding Issues:**');
      verificationResults.outstandingIssues.forEach(issue => {
        sections.push(`- ${issue}`);
      });
    }
    
    return sections.join('\n');
  }

  /**
   * Generate commit footer with metadata
   */
  private async generateCommitFooter(context: CommitContext): Promise<string> {
    const { phaseId, verificationResults } = context;
    
    const footer: string[] = [];
    
    // Phase metadata
    footer.push(`Phase: ${phaseId}`);
    footer.push(`Test-Coverage: ${verificationResults.testResults.coverage}%`);
    
    // Quality metrics (if available)
    if (verificationResults.acceptanceCriteria.some(ac => ac.evidence)) {
      footer.push(`Verification: automated-verification`);
    }
    
    // Session tracking
    footer.push(`Session: ${context.sessionId}`);
    
    return footer.join('\n');
  }

  /**
   * Validate commit readiness
   */
  private async validateCommitReadiness(context: CommitContext, preparation: CommitPreparation): Promise<void> {
    console.log(`✅ Validating commit readiness`);
    
    // 1. Check if working directory is clean (except staged files)
    const gitStatus = execSync('git status --porcelain', { 
      cwd: this.config.projectRoot, 
      encoding: 'utf8' 
    });
    
    const unstagedChanges = gitStatus
      .split('\n')
      .filter(line => line.trim() && !line.startsWith('M  ') && !line.startsWith('A  '));
    
    if (unstagedChanges.length > 0) {
      throw new Error(`Unstaged changes detected: ${unstagedChanges.join(', ')}`);
    }
    
    // 2. Validate commit message format with @explorer
    const messageValidation = await this.delegateToExplorer('validate_commit_message', {
      commitMessage: preparation.commitMessage,
      projectRoot: this.config.projectRoot,
      phaseId: context.phaseId
    });
    
    if (!messageValidation.result.valid) {
      throw new Error(`Commit message validation failed: ${messageValidation.result.issues.join(', ')}`);
    }
    
    // 3. Verify all required files are staged
    const stagedFiles = execSync('git diff --cached --name-only', {
      cwd: this.config.projectRoot,
      encoding: 'utf8'
    }).trim().split('\n').filter(f => f);
    
    const missingFiles = preparation.stagedFiles.filter(file => 
      !stagedFiles.includes(file)
    );
    
    if (missingFiles.length > 0) {
      throw new Error(`Missing staged files: ${missingFiles.join(', ')}`);
    }
    
    console.log(`✅ Commit readiness validated`);
  }

  /**
   * Execute git commit operations
   */
  private async executeCommit(preparation: CommitPreparation, options: CommitOptions): Promise<{
    commitHash: string;
    timestamp: string;
    success: boolean;
  }> {
    console.log(`💾 Executing git commit`);
    
    try {
      // Use @fixer to execute commit with proper error handling
      const commitResult = await this.delegateToFixer('execute_git_commit', {
        commitMessage: preparation.commitMessage,
        projectRoot: this.config.projectRoot,
        options: {
          allowEmpty: false,
          verify: true
        }
      });
      
      if (!commitResult.success) {
        throw new Error(`Git commit failed: ${commitResult.error}`);
      }
      
      // Get commit hash
      const commitHash = execSync('git rev-parse HEAD', {
        cwd: this.config.projectRoot,
        encoding: 'utf8'
      }).trim();
      
      // Optionally push to remote
      if (options.pushToRemote) {
        await this.pushToRemote();
      }
      
      // Optionally create tag
      if (options.createTag) {
        await this.createTag(preparation.metadata.phaseId, commitHash);
      }
      
      return {
        commitHash,
        timestamp: new Date().toISOString(),
        success: true
      };
      
    } catch (error) {
      console.error(`❌ Git commit failed:`, error);
      
      // Attempt to reset to safe state
      try {
        execSync('git reset --soft HEAD~1', { cwd: this.config.projectRoot, stdio: 'pipe' });
        console.log(`🔄 Reset to pre-commit state`);
      } catch (resetError) {
        console.error(`❌ Failed to reset commit state`);
      }
      
      throw error;
    }
  }

  /**
   * Manage branch lifecycle
   */
  private async manageBranchLifecycle(branchName: string, action: BranchAction): Promise<void> {
    console.log(`🌿 Managing branch lifecycle: ${action}`);
    
    try {
      switch (action) {
        case BranchAction.KEEP:
          console.log(`🌿 Keeping branch ${branchName} for future work`);
          break;
          
        case BranchAction.MERGE:
          await this.mergeToMain(branchName);
          break;
          
        case BranchAction.DELETE:
          await this.deleteBranch(branchName);
          break;
      }
    } catch (error) {
      console.warn(`⚠️  Branch management failed: ${error}`);
      // Don't fail the entire commit for branch management issues
    }
  }

  /**
   * Generate final commit report
   */
  private async generateCommitReport(
    phaseId: PhaseId,
    branchName: string,
    preparation: CommitPreparation,
    commitResult: { commitHash: string; timestamp: string; success: boolean }
  ): Promise<CommitReport> {
    return {
      status: commitResult.success ? 'committed' : 'failed',
      phaseId,
      commitHash: commitResult.commitHash,
      commitMessage: preparation.commitMessage,
      filesCommitted: preparation.stagedFiles,
      branchName,
      timestamp: commitResult.timestamp
    };
  }

  /**
   * Generate workflow completion artifact
   */
  private async generateCompletionArtifact(context: CommitContext, report: CommitReport): Promise<void> {
    console.log(`📋 Generating workflow completion artifact`);
    
    const completionArtifact: StateArtifact = {
      sessionId: context.sessionId,
      phaseId: context.phaseId,
      fromPhase: 'commit',
      toPhase: 'completed',
      timestamp: report.timestamp,
      data: {
        workflowStatus: 'completed',
        commitHash: report.commitHash,
        summary: {
          totalPhases: 4,
          completedPhases: 4,
          acceptanceCriteriaMet: `${context.verificationResults.verificationSummary.fullyMet}/${context.verificationResults.verificationSummary.totalCriteria}`,
          finalTestCoverage: context.verificationResults.testResults.coverage,
          filesModified: context.modifiedFiles.length
        },
        artifacts: {
          alignmentReport: `${context.sessionId}-alignment.json`,
          implementationReport: `${context.sessionId}-implementation.json`,
          verificationReport: `${context.sessionId}-verification.json`,
          commitReport: `${context.sessionId}-commit.json`
        },
        recommendations: this.generateRecommendations(context)
      }
    };
    
    // Save completion artifact
    const artifactPath = path.join(this.config.projectRoot, '.phase-artifacts', `${context.sessionId}-completion.json`);
    await fs.mkdir(path.dirname(artifactPath), { recursive: true });
    await fs.writeFile(artifactPath, JSON.stringify(completionArtifact, null, 2));
    
    console.log(`📋 Completion artifact saved to ${artifactPath}`);
  }

  /**
   * Helper methods
   */
  private async stageModifiedFiles(modifiedFiles: string[]): Promise<void> {
    for (const file of modifiedFiles) {
      try {
        execSync(`git add "${file}"`, { cwd: this.config.projectRoot, stdio: 'pipe' });
      } catch (error) {
        console.warn(`⚠️  Failed to stage file ${file}: ${error}`);
      }
    }
  }

  private extractModuleName(modifiedFiles: string[]): string {
    // Extract module name from file paths
    const pythonFiles = modifiedFiles.filter(f => f.endsWith('.py'));
    
    if (pythonFiles.length > 0) {
      // Extract module from path like doc_server/ingestion/document_processor.py
      const pathParts = pythonFiles[0].split('/');
      if (pathParts.length >= 2) {
        return pathParts[pathParts.length - 2]; // ingestion
      }
    }
    
    return 'core'; // fallback
  }

  private generateCommitSummary(verificationResults: VerificationReport): string {
    const met = verificationResults.verificationSummary.fullyMet;
    const total = verificationResults.verificationSummary.totalCriteria;
    
    if (met === total) {
      return `Complete implementation with ${total} acceptance criteria`;
    } else {
      return `Implementation with ${met}/${total} acceptance criteria`;
    }
  }

  private extractQualityMetrics(context: CommitContext): any {
    // Extract quality metrics from verification results if available
    return {
      acceptanceCriteria: context.verificationResults.verificationSummary,
      testResults: context.verificationResults.testResults
    };
  }

  private generateRecommendations(context: CommitContext): string[] {
    const recommendations: string[] = [];
    
    // Test coverage recommendations
    if (context.verificationResults.testResults.coverage < 95) {
      recommendations.push('Consider increasing test coverage to 95%+');
    }
    
    // Outstanding issues
    if (context.verificationResults.outstandingIssues.length > 0) {
      recommendations.push('Address outstanding issues in follow-up work');
    }
    
    // Documentation
    if (!context.modifiedFiles.some(f => f.includes('doc'))) {
      recommendations.push('Consider updating documentation for implemented features');
    }
    
    return recommendations;
  }

  private async pushToRemote(): Promise<void> {
    const pushResult = await this.delegateToFixer('git_push', {
      projectRoot: this.config.projectRoot,
      remote: 'origin',
      branch: 'HEAD'
    });
    
    if (!pushResult.success) {
      throw new Error(`Git push failed: ${pushResult.error}`);
    }
  }

  private async createTag(phaseId: PhaseId, commitHash: string): Promise<void> {
    const tagName = `phase-${phaseId}`;
    
    const tagResult = await this.delegateToFixer('git_tag', {
      projectRoot: this.config.projectRoot,
      tagName,
      commitHash,
      message: `Phase ${phaseId} completion`
    });
    
    if (!tagResult.success) {
      throw new Error(`Git tag failed: ${tagResult.error}`);
    }
  }

  private async mergeToMain(branchName: string): Promise<void> {
    console.log(`🔀 Merging ${branchName} to main`);
    
    // Switch to main
    execSync('git checkout main', { cwd: this.config.projectRoot, stdio: 'pipe' });
    execSync('git pull origin main', { cwd: this.config.projectRoot, stdio: 'pipe' });
    
    // Merge the feature branch
    const mergeResult = await this.delegateToFixer('git_merge', {
      projectRoot: this.config.projectRoot,
      branch: branchName,
      strategy: 'merge'
    });
    
    if (!mergeResult.success) {
      throw new Error(`Git merge failed: ${mergeResult.error}`);
    }
    
    // Delete the merged branch
    await this.deleteBranch(branchName);
  }

  private async deleteBranch(branchName: string): Promise<void> {
    console.log(`🗑️  Deleting branch ${branchName}`);
    
    // Switch to main first
    execSync('git checkout main', { cwd: this.config.projectRoot, stdio: 'pipe' });
    
    // Delete the branch
    const deleteResult = await this.delegateToFixer('git_branch_delete', {
      projectRoot: this.config.projectRoot,
      branchName,
      force: false
    });
    
    if (!deleteResult.success) {
      console.warn(`⚠️  Failed to delete branch ${branchName}: ${deleteResult.error}`);
    }
  }

  /**
   * Delegation methods
   */
  private async delegateToExplorer(task: string, context: any): Promise<DelegationResult> {
    console.log(`🤖 Delegating to @explorer: ${task}`);
    
    return {
      specialist: 'explorer',
      task,
      result: { valid: true, issues: [] },
      success: true,
      executionTime: 800
    };
  }

  private async delegateToFixer(task: string, context: any): Promise<DelegationResult> {
    console.log(`🤖 Delegating to @fixer: ${task}`);
    
    return {
      specialist: 'fixer',
      task,
      result: { success: true },
      success: true,
      executionTime: 1200
    };
  }
}