/**
 * Phase 2: Implementation with Branch Isolation
 * Isolated context subagent for implementing approved tasks in dedicated branch
 */

import * as fs from 'fs/promises';
import * as path from 'path';
import { execSync } from 'child_process';
import {
  PhaseId,
  SessionId,
  ImplementationReport,
  ImplementationTask,
  StateArtifact,
  OrchestratorError,
  DelegationRequest,
  DelegationResult,
  ScopeBoundaries,
} from '../types';

/**
 * Phase 2 Implementation Agent Interface
 */
export interface ImplementationContext {
  phaseId: PhaseId;
  sessionId: SessionId;
  alignmentState: StateArtifact; // From Phase 1
  approvedTasks: string[];
  scopeBoundaries: ScopeBoundaries;
}

interface QualityMetrics {
  lintScore: number;
  typeCheckScore: number;
  maintainabilityScore: number;
  testCoverage: number;
}

/**
 * Phase 2 Implementation Agent Class
 */
export class PhaseImplementationAgent {
  private config: {
    projectRoot: string;
    specsPath: string;
    minTestCoverage: number;
  };

  constructor(config: { projectRoot: string; specsPath: string; minTestCoverage?: number }) {
    this.config = {
      ...config,
      minTestCoverage: config.minTestCoverage || 90.0
    };
  }

  /**
   * Execute phase 2 implementation in isolated context
   */
  async execute(context: ImplementationContext): Promise<ImplementationReport> {
    console.log(`🔧 Phase 2: Starting implementation for ${context.phaseId}`);

    try {
      // 1. Create isolated branch
      const branchName = await this.createIsolatedBranch(context.phaseId);
      
      // 2. Decompose approved tasks into implementation tasks
      const implementationTasks = await this.decomposeTasks(context.approvedTasks);
      
      // 3. Execute implementation (parallel when possible)
      const { completedTasks, failedTasks } = await this.executeImplementation(
        implementationTasks,
        context.scopeBoundaries
      );
      
      // 4. Run comprehensive testing
      const testResults = await this.runTestingSuite();
      
      // 5. Perform quality checks
      await this.performQualityChecks();
      
      // 6. Generate implementation report
      const report = await this.generateImplementationReport(
        context.phaseId,
        branchName,
        completedTasks,
        failedTasks,
        testResults
      );
      
      // 7. Safety gate checks
      await this.performSafetyGateChecks(report);
      
      console.log(`✅ Phase 2: Implementation completed with status: ${report.status}`);
      return report;
      
    } catch (error) {
      const orchestratorError: OrchestratorError = {
        phase: 'implementation',
        errorType: 'system_error',
        message: error instanceof Error ? error.message : 'Unknown implementation error',
        details: { phaseId: context.phaseId },
        recoverable: false,
        suggestedAction: 'Review implementation tasks and retry'
      };
      
      throw orchestratorError;
    }
  }

  /**
   * Create isolated development branch
   */
  private async createIsolatedBranch(phaseId: PhaseId): Promise<string> {
    const branchName = `phase-${phaseId}`;
    
    console.log(`🌿 Creating isolated branch: ${branchName}`);
    
    // Change to project root
    process.chdir(this.config.projectRoot);
    
    try {
      // Ensure we're on main and up to date
      execSync('git checkout main', { stdio: 'pipe' });
      execSync('git pull origin main', { stdio: 'pipe' });
      
      // Create and checkout new branch
      execSync(`git checkout -b ${branchName}`, { stdio: 'pipe' });
      
      console.log(`✅ Created branch: ${branchName}`);
      return branchName;
      
    } catch (error) {
      throw new Error(`Failed to create branch ${branchName}: ${error}`);
    }
  }

  /**
   * Decompose approved tasks into detailed implementation tasks
   */
  private async decomposeTasks(approvedTasks: string[]): Promise<ImplementationTask[]> {
    console.log(`📋 Decomposing ${approvedTasks.length} approved tasks`);
    
    // Use @explorer to analyze existing codebase patterns
    const existingPatterns = await this.delegateToExplorer('analyze_codebase_patterns', {
      projectRoot: this.config.projectRoot,
      focusAreas: ['structure', 'testing', 'naming', 'imports']
    });
    
    // Use @fixer to parallel task decomposition
    const decomposition = await this.delegateToFixer('decompose_tasks', {
      approvedTasks,
      existingPatterns: existingPatterns.result,
      projectRoot: this.config.projectRoot
    });
    
    return decomposition.result as ImplementationTask[];
  }

  /**
   * Execute implementation tasks (parallel when possible)
   */
  private async executeImplementation(
    tasks: ImplementationTask[],
    scopeBoundaries: ScopeBoundaries
  ): Promise<{
    completedTasks: ImplementationTask[];
    failedTasks: ImplementationTask[];
  }> {
    console.log(`🚀 Executing ${tasks.length} implementation tasks`);
    
    // Group tasks by dependencies for parallel execution
    const taskGroups = this.groupTasksByDependencies(tasks);
    const completedTasks: ImplementationTask[] = [];
    const failedTasks: ImplementationTask[] = [];
    
    for (const group of taskGroups) {
      if (group.length === 1) {
        // Single task - execute directly
        const task = group[0];
        try {
          await this.executeSingleTask(task, scopeBoundaries);
          completedTasks.push(task);
        } catch (error) {
          console.error(`❌ Task ${task.taskId} failed:`, error);
          failedTasks.push(task);
        }
      } else {
        // Multiple independent tasks - execute in parallel
        const parallelResults = await Promise.allSettled(
          group.map(task => this.executeSingleTask(task, scopeBoundaries).then(() => task))
        );
        
        parallelResults.forEach((result, index) => {
          const task = group[index];
          if (result.status === 'fulfilled') {
            completedTasks.push(result.value);
          } else {
            console.error(`❌ Task ${task.taskId} failed:`, result.reason);
            failedTasks.push(task);
          }
        });
      }
    }
    
    return { completedTasks, failedTasks };
  }

  /**
   * Execute a single implementation task
   */
  private async executeSingleTask(
    task: ImplementationTask,
    scopeBoundaries: ScopeBoundaries
  ): Promise<void> {
    console.log(`🔨 Executing task: ${task.taskId}`);
    
    // Validate task is within scope boundaries
    this.validateTaskScope(task, scopeBoundaries);
    
    // Use @librarian for API documentation if needed
    if (task.dependencies && task.dependencies.length > 0) {
      await this.delegateToLibrarian('get_api_docs', {
        dependencies: task.dependencies,
        taskContext: task.description
      });
    }
    
    // Use @fixer for actual implementation
    const implementation = await this.delegateToFixer('implement_task', {
      task,
      scopeBoundaries,
      projectRoot: this.config.projectRoot
    });
    
    if (!implementation.success) {
      throw new Error(`Implementation failed: ${implementation.error}`);
    }
    
    // Run task-specific tests
    await this.runTaskTests(task);
  }

  /**
   * Run comprehensive testing suite
   */
  private async runTestingSuite(): Promise<{
    totalTests: number;
    passedTests: number;
    failedTests: number;
    coverage: number;
  }> {
    console.log(`🧪 Running testing suite`);
    
    try {
      // Run pytest with coverage
      const testOutput = execSync(
        'python -m pytest tests/ --cov=doc_server --cov-report=json --cov-report=term-missing',
        { 
          cwd: this.config.projectRoot,
          encoding: 'utf8',
          stdio: 'pipe'
        }
      );
      
      // Parse coverage from coverage.json
      const coverageData = JSON.parse(
        await fs.readFile(
          path.join(this.config.projectRoot, 'coverage.json'),
          'utf8'
        )
      );
      
      const totalCoverage = coverageData.totals.percent_covered;
      
      // Parse test results from pytest output
      const testResults = this.parsePytestOutput(testOutput);
      
      console.log(`📊 Test Results: ${testResults.passed}/${testResults.total} passed, ${testResults.coverage}% coverage`);
      
      return {
        totalTests: testResults.total,
        passedTests: testResults.passed,
        failedTests: testResults.failed,
        coverage: totalCoverage
      };
      
    } catch (error) {
      // If tests fail, parse the output for partial results
      const testResults = this.parsePytestOutput(error.stdout || '');
      
      return {
        totalTests: testResults.total,
        passedTests: testResults.passed,
        failedTests: testResults.failed,
        coverage: 0
      };
    }
  }

  /**
   * Perform quality checks
   */
  private async performQualityChecks(): Promise<QualityMetrics> {
    console.log(`✨ Performing quality checks`);
    
    // 1. Linting with ruff/black
    const lintResults = await this.runLinting();
    
    // 2. Type checking with mypy
    const typeCheckResults = await this.runTypeChecking();
    
    // 3. Maintainability analysis
    const maintainabilityResults = await this.analyzeMaintainability();
    
    return {
      lintScore: lintResults.score,
      typeCheckScore: typeCheckResults.score,
      maintainabilityScore: maintainabilityResults.score,
      testCoverage: 0 // Will be updated from test results
    };
  }

  /**
   * Generate implementation report
   */
  private async generateImplementationReport(
    phaseId: PhaseId,
    branchName: string,
    completedTasks: ImplementationTask[],
    failedTasks: ImplementationTask[],
    testResults: {
      totalTests: number;
      passedTests: number;
      failedTests: number;
      coverage: number;
    }
  ): Promise<ImplementationReport> {
    const status = failedTasks.length === 0 && testResults.coverage >= this.config.minTestCoverage 
      ? 'completed' 
      : failedTasks.length > 0 
      ? 'failed' 
      : 'partial';
    
    // Generate implementation notes
    const implementationNotes = await this.generateImplementationNotes(completedTasks);
    
    return {
      status,
      phaseId,
      branchName,
      completedTasks,
      failedTasks,
      testResults,
      implementationNotes
    };
  }

  /**
   * Perform safety gate checks
   */
  private async performSafetyGateChecks(report: ImplementationReport): Promise<void> {
    if (report.status === 'failed') {
      console.error('🚨 CRITICAL: Implementation failed with blocked tasks');
      console.error('Automatic stop triggered. Manual review required before proceeding.');
      
      throw new Error('Implementation failed with blocked tasks. Human approval required before proceeding.');
    }
    
    if (report.testResults.coverage < this.config.minTestCoverage) {
      console.warn(`⚠️  Test coverage ${report.testResults.coverage}% below minimum ${this.config.minTestCoverage}%`);
      
      if (report.testResults.coverage < this.config.minTestCoverage * 0.8) {
        throw new Error('Test coverage critically low. Human approval required before proceeding.');
      }
    }
  }

  /**
   * Helper methods
   */
  private groupTasksByDependencies(tasks: ImplementationTask[]): ImplementationTask[][] {
    // Simple dependency grouping - can be enhanced with proper topological sort
    const groups: ImplementationTask[][] = [];
    const processed = new Set<string>();
    
    while (processed.size < tasks.length) {
      const currentGroup: ImplementationTask[] = [];
      
      for (const task of tasks) {
        if (processed.has(task.taskId)) continue;
        
        const hasUnprocessedDependencies = (task.dependencies || []).some(
          dep => !processed.has(dep)
        );
        
        if (!hasUnprocessedDependencies) {
          currentGroup.push(task);
          processed.add(task.taskId);
        }
      }
      
      if (currentGroup.length === 0) {
        // Circular dependency or missing dependency
        const remaining = tasks.filter(t => !processed.has(t.taskId));
        groups.push(remaining);
        break;
      }
      
      groups.push(currentGroup);
    }
    
    return groups;
  }

  private validateTaskScope(task: ImplementationTask, scopeBoundaries: ScopeBoundaries): void {
    // Check if task files are within included modules
    for (const file of task.targetFiles) {
      const isIncluded = scopeBoundaries.includedModules.some(module => 
        file.startsWith(module)
      );
      
      const isExcluded = scopeBoundaries.excludedModules.some(module => 
        file.startsWith(module)
      );
      
      if (isExcluded || !isIncluded) {
        throw new Error(`Task ${task.taskId} attempts to modify out-of-scope file: ${file}`);
      }
    }
  }

  private async runTaskTests(task: ImplementationTask): Promise<void> {
    for (const testFile of task.testFiles) {
      try {
        execSync(`python -m pytest ${testFile} -v`, {
          cwd: this.config.projectRoot,
          stdio: 'pipe'
        });
      } catch (error) {
        throw new Error(`Tests failed for ${testFile}: ${error}`);
      }
    }
  }

  private parsePytestOutput(output: string): {
    total: number;
    passed: number;
    failed: number;
  } {
    // Parse pytest output for test counts
    const totalMatch = output.match(/(\d+)\s+tests?/);
    const passedMatch = output.match(/(\d+)\s+passed/);
    const failedMatch = output.match(/(\d+)\s+failed/);
    
    return {
      total: parseInt(totalMatch?.[1] || '0'),
      passed: parseInt(passedMatch?.[1] || '0'),
      failed: parseInt(failedMatch?.[1] || '0')
    };
  }

  private async runLinting(): Promise<{ score: number; errors: string[] }> {
    try {
      execSync('ruff check .', { cwd: this.config.projectRoot, stdio: 'pipe' });
      return { score: 10.0, errors: [] };
    } catch (error) {
      const errors = (error.stdout || '').toString().split('\n').filter(line => line.trim());
      return { score: Math.max(0, 10 - errors.length), errors };
    }
  }

  private async runTypeChecking(): Promise<{ score: number; errors: string[] }> {
    try {
      execSync('mypy doc_server/', { cwd: this.config.projectRoot, stdio: 'pipe' });
      return { score: 10.0, errors: [] };
    } catch (error) {
      const errors = (error.stdout || '').toString().split('\n').filter(line => line.trim());
      return { score: Math.max(0, 10 - errors.length), errors };
    }
  }

  private async analyzeMaintainability(): Promise<{ score: number; issues: string[] }> {
    // Simple maintainability analysis based on file sizes and complexity
    const issues: string[] = [];
    let score = 10.0;
    
    try {
      const files = await this.readdirRecursive(this.config.projectRoot);
      
      for (const file of files) {
        if (file.endsWith('.py')) {
          const content = await fs.readFile(file, 'utf8');
          const lines = content.split('\n').length;
          
          if (lines > 500) {
            issues.push(`Large file: ${file} (${lines} lines)`);
            score -= 1;
          }
        }
      }
      
    } catch (error) {
      issues.push(`Failed to analyze maintainability: ${error}`);
      score = 5.0;
    }
    
    return { score: Math.max(0, score), issues };
  }

  private async readdirRecursive(dir: string): Promise<string[]> {
    const files: string[] = [];
    
    try {
      const entries = await fs.readdir(dir, { withFileTypes: true });
      
      for (const entry of entries) {
        const fullPath = path.join(dir, entry.name);
        
        if (entry.isDirectory() && !entry.name.startsWith('.') && entry.name !== 'node_modules') {
          files.push(...await this.readdirRecursive(fullPath));
        } else if (entry.isFile()) {
          files.push(fullPath);
        }
      }
    } catch (error) {
      // Ignore permission errors
    }
    
    return files;
  }

  private async generateImplementationNotes(completedTasks: ImplementationTask[]): Promise<string[]> {
    const notes: string[] = [];
    
    for (const task of completedTasks) {
      if (task.description.includes('optimization') || task.description.includes('optimize')) {
        notes.push(`Optimized performance for ${task.taskId}`);
      }
      
      if (task.description.includes('error') || task.description.includes('exception')) {
        notes.push(`Enhanced error handling in ${task.taskId}`);
      }
    }
    
    if (completedTasks.length > 0) {
      notes.push(`Successfully implemented ${completedTasks.length} tasks in isolated environment`);
    }
    
    return notes;
  }

  /**
   * Delegation methods
   */
  private async delegateToExplorer(task: string, context: any): Promise<DelegationResult> {
    console.log(`🤖 Delegating to @explorer: ${task}`);
    
    return {
      specialist: 'explorer',
      task,
      result: `Mock result for ${task} with context: ${JSON.stringify(context)}`,
      success: true,
      executionTime: 1200
    };
  }

  private async delegateToFixer(task: string, context: any): Promise<DelegationResult> {
    console.log(`🤖 Delegating to @fixer: ${task}`);
    
    return {
      specialist: 'fixer',
      task,
      result: { success: true, implementedFiles: [] },
      success: true,
      executionTime: 2000
    };
  }

  private async delegateToLibrarian(task: string, context: any): Promise<DelegationResult> {
    console.log(`🤖 Delegating to @librarian: ${task}`);
    
    return {
      specialist: 'librarian',
      task,
      result: { apiDocs: {} },
      success: true,
      executionTime: 800
    };
  }
}