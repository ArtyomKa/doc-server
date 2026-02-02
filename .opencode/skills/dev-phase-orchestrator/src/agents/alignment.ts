/**
 * Phase 1: Product Spec Alignment Verification
 * Isolated context subagent for validating alignment between tasks, product specs, and implementation plans
 */

import * as fs from 'fs/promises';
import * as path from 'path';
import { z } from 'zod';
import {
  PhaseId,
  SessionId,
  AlignmentReport,
  AlignmentIssue,
  StateArtifact,
  OrchestratorError,
  DelegationRequest,
  DelegationResult,
} from '../types';

/**
 * Phase 1 Alignment Agent Interface
 */
export interface AlignmentContext {
  phaseId: PhaseId;
  sessionId: SessionId;
  previousState?: StateArtifact;
}

interface PhaseDocuments {
  tasks: string;
  productSpec: string;
  implementationPlan: string;
}

interface TaskInfo {
  taskId: string;
  description: string;
  status: string;
  requirements: string[];
}

interface RequirementInfo {
  requirementId: string;
  description: string;
  priority: string;
  phase: string;
}

/**
 * Phase 1 Alignment Agent Class
 */
export class PhaseAlignmentAgent {
  private config: {
    specsPath: string;
    projectRoot: string;
  };

  constructor(config: { specsPath: string; projectRoot: string }) {
    this.config = config;
  }

  /**
   * Execute phase 1 alignment verification in isolated context
   */
  async execute(context: AlignmentContext): Promise<AlignmentReport> {
    console.log(`🔍 Phase 1: Starting alignment verification for ${context.phaseId}`);

    try {
      // 1. Extract phase-specific documents
      const documents = await this.extractPhaseContent(context.phaseId);
      
      // 2. Parse structured data from documents
      const tasks = await this.parsePhaseTasks(documents.tasks, context.phaseId);
      const requirements = await this.parseRequirements(documents.productSpec, context.phaseId);
      const plan = await this.parseImplementationPlan(documents.implementationPlan, context.phaseId);
      
      // 3. Perform cross-reference verification
      const alignmentIssues = await this.performCrossReferenceVerification(
        tasks,
        requirements,
        plan,
        context.phaseId
      );
      
      // 4. Categorize issues by severity
      const { critical, major, minor } = this.categorizeIssues(alignmentIssues);
      
      // 5. Generate alignment report
      const report = await this.generateAlignmentReport(
        context.phaseId,
        tasks,
        requirements,
        alignmentIssues,
        { critical, major, minor }
      );
      
      // 6. Safety gate checks
      await this.performSafetyGateChecks(report);
      
      console.log(`✅ Phase 1: Alignment verification completed with status: ${report.status}`);
      return report;
      
    } catch (error) {
      const orchestratorError: OrchestratorError = {
        phase: 'alignment',
        errorType: 'system_error',
        message: error instanceof Error ? error.message : 'Unknown alignment error',
        details: { phaseId: context.phaseId },
        recoverable: false,
        suggestedAction: 'Review phase documents and retry alignment verification'
      };
      
      throw orchestratorError;
    }
  }

  /**
   * Extract phase-specific content from specification documents
   */
  private async extractPhaseContent(phaseId: PhaseId): Promise<PhaseDocuments> {
    const tasksPath = path.join(this.config.specsPath, 'doc-server-tasks.md');
    const productSpecPath = path.join(this.config.specsPath, 'doc-server-product-spec.md');
    const planPath = path.join(this.config.specsPath, 'doc-server-plan.md');

    // Use delegation to @explorer for robust document location and extraction
    const tasksContent = await this.delegateToExplorer('extract_phase_tasks', {
      filePath: tasksPath,
      phaseId,
      sectionPattern: `## Phase ${phaseId}`
    });

    const productSpecContent = await this.delegateToExplorer('extract_relevant_requirements', {
      filePath: productSpecPath,
      phaseId,
      extractionType: 'phase_relevant_requirements'
    });

    const planContent = await this.delegateToExplorer('extract_phase_plan', {
      filePath: planPath,
      phaseId,
      sectionPattern: `### Phase ${phaseId}`
    });

    return {
      tasks: tasksContent.result as string,
      productSpec: productSpecContent.result as string,
      implementationPlan: planContent.result as string
    };
  }

  /**
   * Parse phase tasks from tasks document
   */
  private async parsePhaseTasks(tasksContent: string, phaseId: PhaseId): Promise<TaskInfo[]> {
    const taskRegex = /####\s+(\d+\.\d+\.\d+):\s*([^\n]+)\s*\n\s*\*\*Status:\*\*\s*([^\n]+)(?:\s*\*\*Requirements:\*\*\s*([^\n]+))?/g;
    const tasks: TaskInfo[] = [];
    let match;

    while ((match = taskRegex.exec(tasksContent)) !== null) {
      const [, taskId, description, status, requirementsStr] = match;
      
      if (taskId.startsWith(`${phaseId}.`)) {
        tasks.push({
          taskId,
          description: description.trim(),
          status: status.trim(),
          requirements: requirementsStr ? requirementsStr.split(',').map(r => r.trim()) : []
        });
      }
    }

    return tasks;
  }

  /**
   * Parse requirements from product specification
   */
  private async parseRequirements(productSpecContent: string, phaseId: PhaseId): Promise<RequirementInfo[]> {
    // Delegate to @oracle for complex requirement parsing and prioritization
    const requirements = await this.delegateToOracle('parse_requirements', {
      content: productSpecContent,
      phaseId,
      parseStrategy: 'phase_relevant_requirements'
    });

    return requirements.result as RequirementInfo[];
  }

  /**
   * Parse implementation plan for phase
   */
  private async parseImplementationPlan(planContent: string, phaseId: PhaseId): Promise<any> {
    const planRegex = /###\s+Phase\s+${phaseId}\s*\n([\s\S]*?)(?=###\s+Phase\s+\d+\.\d+|\n##|$)/;
    const match = planContent.match(new RegExp(planRegex));
    
    if (!match) {
      throw new Error(`No implementation plan found for Phase ${phaseId}`);
    }

    const phasePlanSection = match[1];
    
    // Parse key sections from the plan
    return {
      phaseId,
      modules: this.extractListItems(phasePlanSection, 'Modules'),
      approach: this.extractSection(phasePlanSection, 'Approach'),
      dependencies: this.extractListItems(phasePlanSection, 'Dependencies'),
      deliverables: this.extractListItems(phasePlanSection, 'Deliverables')
    };
  }

  /**
   * Perform cross-reference verification between tasks, requirements, and plan
   */
  private async performCrossReferenceVerification(
    tasks: TaskInfo[],
    requirements: RequirementInfo[],
    plan: any,
    phaseId: PhaseId
  ): Promise<AlignmentIssue[]> {
    const issues: AlignmentIssue[] = [];

    // 1. Tasks ↔ Requirements verification
    const taskReqIssues = await this.verifyTaskRequirementsAlignment(tasks, requirements);
    issues.push(...taskReqIssues);

    // 2. Plan ↔ Requirements verification
    const planReqIssues = await this.verifyPlanRequirementsAlignment(plan, requirements, phaseId);
    issues.push(...planReqIssues);

    // 3. Tasks ↔ Plan verification
    const taskPlanIssues = await this.verifyTaskPlanAlignment(tasks, plan);
    issues.push(...taskPlanIssues);

    return issues;
  }

  /**
   * Verify that tasks cover all relevant requirements
   */
  private async verifyTaskRequirementsAlignment(
    tasks: TaskInfo[],
    requirements: RequirementInfo[]
  ): Promise<AlignmentIssue[]> {
    const issues: AlignmentIssue[] = [];
    const coveredRequirements = new Set<string>();

    // Check which requirements are covered by tasks
    for (const task of tasks) {
      for (const req of task.requirements) {
        coveredRequirements.add(req);
      }
    }

    // Find uncovered critical requirements
    for (const req of requirements) {
      if (!coveredRequirements.has(req.requirementId)) {
        issues.push({
          type: 'task_product_gap',
          description: `Requirement ${req.requirementId} not covered by any task`,
          severity: req.priority === 'critical' ? 'critical' : 'major',
          affectedComponents: [], // Would be populated based on requirement
          suggestedResolution: `Add task or modify existing task to cover requirement ${req.requirementId}`
        });
      }
    }

    return issues;
  }

  /**
   * Verify that implementation plan aligns with requirements
   */
  private async verifyPlanRequirementsAlignment(
    plan: any,
    requirements: RequirementInfo[],
    phaseId: PhaseId
  ): Promise<AlignmentIssue[]> {
    const issues: AlignmentIssue[] = [];

    // Use @oracle to analyze architectural alignment
    const oracleAnalysis = await this.delegateToOracle('analyze_plan_requirement_alignment', {
      plan,
      requirements,
      phaseId
    });

    if (oracleAnalysis.result.issues) {
      issues.push(...oracleAnalysis.result.issues);
    }

    return issues;
  }

  /**
   * Verify that task breakdown aligns with implementation approach
   */
  private async verifyTaskPlanAlignment(tasks: TaskInfo[], plan: any): Promise<AlignmentIssue[]> {
    const issues: AlignmentIssue[] = [];
    const plannedModules = plan.modules || [];

    // Check if all planned modules have corresponding tasks
    for (const module of plannedModules) {
      const hasTaskForModule = tasks.some(task => 
        task.description.toLowerCase().includes(module.toLowerCase()) ||
        task.taskId.toLowerCase().includes(module.toLowerCase())
      );

      if (!hasTaskForModule) {
        issues.push({
          type: 'task_plan_inconsistency',
          description: `Planned module ${module} has no corresponding task`,
          severity: 'major',
          affectedComponents: [module],
          suggestedResolution: `Add task to implement ${module} or remove from plan`
        });
      }
    }

    return issues;
  }

  /**
   * Categorize alignment issues by severity
   */
  private categorizeIssues(issues: AlignmentIssue[]): {
    critical: AlignmentIssue[];
    major: AlignmentIssue[];
    minor: AlignmentIssue[];
  } {
    return {
      critical: issues.filter(issue => issue.severity === 'critical'),
      major: issues.filter(issue => issue.severity === 'major'),
      minor: issues.filter(issue => issue.severity === 'minor')
    };
  }

  /**
   * Generate alignment report
   */
  private async generateAlignmentReport(
    phaseId: PhaseId,
    tasks: TaskInfo[],
    requirements: RequirementInfo[],
    issues: AlignmentIssue[],
    categorizedIssues: { critical: AlignmentIssue[]; major: AlignmentIssue[]; minor: AlignmentIssue[] }
  ): Promise<AlignmentReport> {
    const { critical, major, minor } = categorizedIssues;
    
    // Determine overall status
    let status: 'approved' | 'rejected' | 'needs_review' = 'approved';
    if (critical.length > 0) {
      status = 'rejected';
    } else if (major.length > 0) {
      status = 'needs_review';
    }

    // Determine approved tasks (exclude tasks with critical/major issues)
    const problematicTaskIds = new Set(
      issues
        .filter(issue => issue.severity === 'critical' || issue.severity === 'major')
        .map(issue => this.extractTaskIdFromIssue(issue))
        .filter(Boolean)
    );

    const approvedTasks = tasks
      .filter(task => !problematicTaskIds.has(task.taskId))
      .map(task => task.taskId);

    // Determine scope boundaries
    const scopeBoundaries = await this.determineScopeBoundaries(tasks, requirements);

    // Create decision log
    const decisionLog = [
      {
        decision: status === 'approved' ? 'Alignment approved' : 'Issues identified',
        rationale: `Found ${critical.length} critical, ${major.length} major, ${minor.length} minor issues`,
        timestamp: new Date().toISOString()
      }
    ];

    return {
      status,
      phaseId,
      analysisSummary: {
        totalTasks: tasks.length,
        alignedTasks: approvedTasks.length,
        criticalIssues: critical.length,
        majorIssues: major.length,
        minorIssues: minor.length
      },
      alignmentIssues: issues,
      approvedTasks,
      scopeBoundaries,
      decisionLog
    };
  }

  /**
   * Perform safety gate checks
   */
  private async performSafetyGateChecks(report: AlignmentReport): Promise<void> {
    if (report.status === 'rejected') {
      console.error('🚨 CRITICAL: Alignment verification failed with critical issues');
      console.error('Automatic stop triggered. Manual review required before proceeding.');
      
      // In a real implementation, this would trigger human intervention
      throw new Error('Critical alignment issues detected. Human approval required before proceeding.');
    }

    if (report.status === 'needs_review') {
      console.warn('⚠️  Major alignment issues detected. Human review recommended before proceeding.');
    }
  }

  /**
   * Delegate to @explorer specialist
   */
  private async delegateToExplorer(task: string, context: any): Promise<DelegationResult> {
    // This would interface with the actual @explorer agent
    console.log(`🤖 Delegating to @explorer: ${task}`);
    
    // Mock implementation for now
    return {
      specialist: 'explorer',
      task,
      result: `Mock result for ${task} with context: ${JSON.stringify(context)}`,
      success: true,
      executionTime: 1000
    };
  }

  /**
   * Delegate to @oracle specialist
   */
  private async delegateToOracle(task: string, context: any): Promise<DelegationResult> {
    // This would interface with the actual @oracle agent
    console.log(`🤖 Delegating to @oracle: ${task}`);
    
    // Mock implementation for now
    return {
      specialist: 'oracle',
      task,
      result: { issues: [], recommendations: [] }, // Mock structured result
      success: true,
      executionTime: 1500
    };
  }

  /**
   * Helper methods for document parsing
   */
  private extractListItems(content: string, sectionName: string): string[] {
    const sectionRegex = new RegExp(`\\*\\*${sectionName}:\\*\\*[^\\n]*\\n([\\s\\S]*?)(?=\\n\\*\\*|\\n###|$)`);
    const match = content.match(sectionRegex);
    
    if (!match) return [];
    
    const listContent = match[1];
    const itemRegex = /^-\s+(.+)$/gm;
    const items: string[] = [];
    let itemMatch;
    
    while ((itemMatch = itemRegex.exec(listContent)) !== null) {
      items.push(itemMatch[1].trim());
    }
    
    return items;
  }

  private extractSection(content: string, sectionName: string): string {
    const sectionRegex = new RegExp(`\\*\\*${sectionName}:\\*\\*[^\\n]*\\n([\\s\\S]*?)(?=\\n\\*\\*|\\n###|$)`);
    const match = content.match(sectionRegex);
    return match ? match[1].trim() : '';
  }

  private extractTaskIdFromIssue(issue: AlignmentIssue): string | null {
    // Extract task ID from issue description or affected components
    const taskIdMatch = issue.description.match(/(\d+\.\d+\.\d+)/);
    return taskIdMatch ? taskIdMatch[1] : null;
  }

  private async determineScopeBoundaries(tasks: TaskInfo[], requirements: RequirementInfo[]): Promise<{
    includedModules: string[];
    excludedModules: string[];
    filePatterns: string[];
  }> {
    // Analyze tasks and requirements to determine scope
    const mentionedModules = new Set<string>();
    
    for (const task of tasks) {
      // Extract module names from task descriptions
      const moduleMatches = task.description.match(/\b(\w+\.?\w*)\.(py|js|ts|md)\b/g);
      if (moduleMatches) {
        moduleMatches.forEach(match => mentionedModules.add(match));
      }
    }

    return {
      includedModules: Array.from(mentionedModules),
      excludedModules: ['doc_server/ui/*', 'doc_server/frontend/*'],
      filePatterns: ['*.py', 'test_*.py', '*.md']
    };
  }
}