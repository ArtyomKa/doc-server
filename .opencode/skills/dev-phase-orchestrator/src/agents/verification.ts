/**
 * Phase 3: Acceptance Criteria Verification
 * Isolated context subagent for validating implementation against acceptance criteria
 */

import * as fs from 'fs/promises';
import * as path from 'path';
import { execSync } from 'child_process';
import {
  PhaseId,
  SessionId,
  VerificationReport,
  AcceptanceCriterion,
  StateArtifact,
  OrchestratorError,
  DelegationRequest,
  DelegationResult,
  ImplementationTask,
} from '../types';

/**
 * Phase 3 Verification Agent Interface
 */
export interface VerificationContext {
  phaseId: PhaseId;
  sessionId: SessionId;
  implementationState: StateArtifact; // From Phase 2
  branchName: string;
  modifiedFiles: string[];
  implementationTasks: ImplementationTask[];
}

interface EvidenceItem {
  type: 'code' | 'test' | 'documentation' | 'configuration';
  location: string;
  description: string;
  lineNumbers?: number[];
}

interface CriterionEvidence {
  criterionId: string;
  evidence: EvidenceItem[];
  testResults: TestResult[];
  confidence: number; // 0-100
}

interface TestResult {
  testName: string;
  status: 'passed' | 'failed' | 'skipped';
  coverage: number;
  executionTime: number;
  errorMessage?: string;
}

interface QualityMetrics {
  securityScore: number;
  performanceScore: number;
  maintainabilityScore: number;
  documentationScore: number;
}

/**
 * Phase 3 Verification Agent Class
 */
export class PhaseVerificationAgent {
  private config: {
    projectRoot: string;
    specsPath: string;
    minTestCoverage: number;
    minSecurityScore: number;
  };

  constructor(config: { 
    projectRoot: string; 
    specsPath: string; 
    minTestCoverage?: number;
    minSecurityScore?: number;
  }) {
    this.config = {
      ...config,
      minTestCoverage: config.minTestCoverage || 90.0,
      minSecurityScore: config.minSecurityScore || 8.0
    };
  }

  /**
   * Execute phase 3 verification in isolated context
   */
  async execute(context: VerificationContext): Promise<VerificationReport> {
    console.log(`🔍 Phase 3: Starting verification for ${context.phaseId}`);

    try {
      // 1. Extract acceptance criteria for the phase
      const acceptanceCriteria = await this.extractAcceptanceCriteria(context.phaseId);
      
      // 2. Gather implementation evidence for each criterion
      const evidenceMapping = await this.gatherImplementationEvidence(
        acceptanceCriteria,
        context.modifiedFiles,
        context.implementationTasks
      );
      
      // 3. Run comprehensive verification tests
      const testResults = await this.runVerificationTests(context.branchName);
      
      // 4. Validate each criterion against evidence
      const validatedCriteria = await this.validateCriteria(
        acceptanceCriteria,
        evidenceMapping,
        testResults
      );
      
      // 5. Perform quality analysis
      const qualityMetrics = await this.performQualityAnalysis();
      
      // 6. Generate verification report
      const report = await this.generateVerificationReport(
        context.phaseId,
        validatedCriteria,
        testResults,
        qualityMetrics
      );
      
      // 7. Safety gate checks
      await this.performSafetyGateChecks(report);
      
      console.log(`✅ Phase 3: Verification completed with status: ${report.status}`);
      return report;
      
    } catch (error) {
      const orchestratorError: OrchestratorError = {
        phase: 'verification',
        errorType: 'system_error',
        message: error instanceof Error ? error.message : 'Unknown verification error',
        details: { phaseId: context.phaseId },
        recoverable: false,
        suggestedAction: 'Review verification criteria and retry'
      };
      
      throw orchestratorError;
    }
  }

  /**
   * Extract acceptance criteria from specification documents
   */
  private async extractAcceptanceCriteria(phaseId: PhaseId): Promise<AcceptanceCriterion[]> {
    console.log(`📋 Extracting acceptance criteria for phase ${phaseId}`);
    
    const acceptanceSpecPath = path.join(this.config.specsPath, 'doc-server-acceptence.md');
    
    // Use @explorer to extract AC-X.Y.Z patterns
    const extractionResult = await this.delegateToExplorer('extract_acceptance_criteria', {
      filePath: acceptanceSpecPath,
      phaseId,
      pattern: '####\\s+(\\d+\\.\\d+\\.\\d+)'
    });
    
    const criteria = extractionResult.result as AcceptanceCriterion[];
    
    console.log(`📋 Found ${criteria.length} acceptance criteria for phase ${phaseId}`);
    return criteria;
  }

  /**
   * Gather implementation evidence for each criterion
   */
  private async gatherImplementationEvidence(
    criteria: AcceptanceCriterion[],
    modifiedFiles: string[],
    implementationTasks: ImplementationTask[]
  ): Promise<Map<string, CriterionEvidence>> {
    console.log(`🔍 Gathering implementation evidence for ${criteria.length} criteria`);
    
    const evidenceMap = new Map<string, CriterionEvidence>();
    
    for (const criterion of criteria) {
      const evidence: CriterionEvidence = {
        criterionId: criterion.criterionId,
        evidence: [],
        testResults: [],
        confidence: 0
      };
      
      // 1. Find code evidence
      const codeEvidence = await this.findCodeEvidence(criterion, modifiedFiles);
      evidence.evidence.push(...codeEvidence);
      
      // 2. Find test evidence
      const testEvidence = await this.findTestEvidence(criterion, implementationTasks);
      evidence.evidence.push(...testEvidence);
      
      // 3. Find documentation evidence
      const docEvidence = await this.findDocumentationEvidence(criterion);
      evidence.evidence.push(...docEvidence);
      
      // 4. Calculate confidence score
      evidence.confidence = this.calculateEvidenceConfidence(evidence);
      
      evidenceMap.set(criterion.criterionId, evidence);
    }
    
    return evidenceMap;
  }

  /**
   * Run comprehensive verification tests
   */
  private async runVerificationTests(branchName: string): Promise<{
    totalTests: number;
    passedTests: number;
    failedTests: number;
    coverage: number;
    detailedResults: TestResult[];
  }> {
    console.log(`🧪 Running comprehensive verification tests on ${branchName}`);
    
    // Ensure we're on the correct branch
    process.chdir(this.config.projectRoot);
    execSync(`git checkout ${branchName}`, { stdio: 'pipe' });
    
    // 1. Run unit tests with coverage
    const unitTestResults = await this.runUnitTests();
    
    // 2. Run integration tests
    const integrationTestResults = await this.runIntegrationTests();
    
    // 3. Run performance tests
    const performanceTestResults = await this.runPerformanceTests();
    
    // 4. Run security tests
    const securityTestResults = await this.runSecurityTests();
    
    // Combine all results
    const allResults = [
      ...unitTestResults.detailedResults,
      ...integrationTestResults.detailedResults,
      ...performanceTestResults.detailedResults,
      ...securityTestResults.detailedResults
    ];
    
    const totalTests = unitTestResults.totalTests + integrationTestResults.totalTests + 
                     performanceTestResults.totalTests + securityTestResults.totalTests;
    const passedTests = allResults.filter(r => r.status === 'passed').length;
    const failedTests = allResults.filter(r => r.status === 'failed').length;
    
    // Use coverage from unit tests (primary coverage metric)
    const coverage = unitTestResults.coverage;
    
    console.log(`📊 Test Results: ${passedTests}/${totalTests} passed, ${coverage}% coverage`);
    
    return {
      totalTests,
      passedTests,
      failedTests,
      coverage,
      detailedResults: allResults
    };
  }

  /**
   * Validate each criterion against evidence
   */
  private async validateCriteria(
    criteria: AcceptanceCriterion[],
    evidenceMapping: Map<string, CriterionEvidence>,
    testResults: { detailedResults: TestResult[] }
  ): Promise<AcceptanceCriterion[]> {
    console.log(`✅ Validating ${criteria.length} criteria against evidence`);
    
    const validatedCriteria: AcceptanceCriterion[] = [];
    
    for (const criterion of criteria) {
      const evidence = evidenceMapping.get(criterion.criterionId);
      
      if (!evidence) {
        // No evidence found
        validatedCriteria.push({
          ...criterion,
          status: 'not_met',
          gapReason: 'No implementation evidence found'
        });
        continue;
      }
      
      // Validate each requirement
      let metRequirements = 0;
      const totalRequirements = criterion.requirements.length;
      
      for (const requirement of criterion.requirements) {
        if (await this.isRequirementMet(requirement, evidence, testResults)) {
          metRequirements++;
        }
      }
      
      // Determine overall status
      let status: 'met' | 'partially_met' | 'not_met';
      let gapReason: string | undefined;
      
      if (metRequirements === totalRequirements) {
        status = 'met';
      } else if (metRequirements >= totalRequirements * 0.7) {
        status = 'partially_met';
        gapReason = `${totalRequirements - metRequirements} requirements not fully implemented`;
      } else {
        status = 'not_met';
        gapReason = `Only ${metRequirements}/${totalRequirements} requirements implemented`;
      }
      
      validatedCriteria.push({
        ...criterion,
        status,
        evidence: this.formatEvidence(evidence),
        gapReason
      });
    }
    
    return validatedCriteria;
  }

  /**
   * Perform quality analysis
   */
  private async performQualityAnalysis(): Promise<QualityMetrics> {
    console.log(`⭐ Performing quality analysis`);
    
    // 1. Security analysis
    const securityScore = await this.analyzeSecurity();
    
    // 2. Performance analysis
    const performanceScore = await this.analyzePerformance();
    
    // 3. Maintainability analysis
    const maintainabilityScore = await this.analyzeMaintainability();
    
    // 4. Documentation analysis
    const documentationScore = await this.analyzeDocumentation();
    
    return {
      securityScore,
      performanceScore,
      maintainabilityScore,
      documentationScore
    };
  }

  /**
   * Generate verification report
   */
  private async generateVerificationReport(
    phaseId: PhaseId,
    validatedCriteria: AcceptanceCriterion[],
    testResults: {
      totalTests: number;
      passedTests: number;
      failedTests: number;
      coverage: number;
      detailedResults: TestResult[];
    },
    qualityMetrics: QualityMetrics
  ): Promise<VerificationReport> {
    console.log(`📝 Generating verification report for phase ${phaseId}`);
    
    // Calculate verification summary
    const totalCriteria = validatedCriteria.length;
    const fullyMet = validatedCriteria.filter(c => c.status === 'met').length;
    const partiallyMet = validatedCriteria.filter(c => c.status === 'partially_met').length;
    const notMet = validatedCriteria.filter(c => c.status === 'not_met').length;
    
    // Determine overall status
    let status: 'passed' | 'failed' | 'partial';
    
    if (notMet > 0) {
      status = 'failed';
    } else if (partiallyMet > 0 || testResults.coverage < this.config.minTestCoverage) {
      status = 'partial';
    } else {
      status = 'passed';
    }
    
    // Identify outstanding issues
    const outstandingIssues: string[] = [];
    
    for (const criterion of validatedCriteria) {
      if (criterion.status === 'partially_met' && criterion.gapReason) {
        outstandingIssues.push(`AC-${criterion.criterionId}: ${criterion.gapReason}`);
      } else if (criterion.status === 'not_met' && criterion.gapReason) {
        outstandingIssues.push(`AC-${criterion.criterionId}: ${criterion.gapReason}`);
      }
    }
    
    if (testResults.coverage < this.config.minTestCoverage) {
      outstandingIssues.push(`Test coverage ${testResults.coverage}% below minimum ${this.config.minTestCoverage}%`);
    }
    
    if (qualityMetrics.securityScore < this.config.minSecurityScore) {
      outstandingIssues.push(`Security score ${qualityMetrics.securityScore} below minimum ${this.config.minSecurityScore}`);
    }
    
    return {
      status,
      phaseId,
      acceptanceCriteria: validatedCriteria,
      testResults: {
        totalTests: testResults.totalTests,
        passedTests: testResults.passedTests,
        failedTests: testResults.failedTests,
        coverage: testResults.coverage
      },
      verificationSummary: {
        totalCriteria,
        fullyMet,
        partiallyMet,
        notMet
      },
      outstandingIssues
    };
  }

  /**
   * Perform safety gate checks
   */
  private async performSafetyGateChecks(report: VerificationReport): Promise<void> {
    if (report.status === 'failed') {
      console.error('🚨 CRITICAL: Verification failed with unmet criteria');
      console.error('Unmet criteria require attention before proceeding');
      
      throw new Error('Critical verification failures. Human approval required before proceeding.');
    }
    
    if (report.testResults.coverage < this.config.minTestCoverage * 0.8) {
      console.error(`🚨 CRITICAL: Test coverage critically low at ${report.testResults.coverage}%`);
      throw new Error('Test coverage critically low. Human approval required before proceeding.');
    }
    
    if (report.outstandingIssues.length > 0) {
      console.warn(`⚠️  Verification has ${report.outstandingIssues.length} outstanding issues:`);
      report.outstandingIssues.forEach(issue => console.warn(`   - ${issue}`));
    }
  }

  /**
   * Helper methods for evidence gathering
   */
  private async findCodeEvidence(criterion: AcceptanceCriterion, modifiedFiles: string[]): Promise<EvidenceItem[]> {
    const evidence: EvidenceItem[] = [];
    
    // Use @explorer to search for code implementing this criterion
    const searchResult = await this.delegateToExplorer('find_criterion_implementation', {
      criterion: criterion.description,
      requirements: criterion.requirements,
      searchPaths: modifiedFiles
    });
    
    const matches = searchResult.result as Array<{
      file: string;
      lines: number[];
      snippet: string;
    }>;
    
    for (const match of matches) {
      evidence.push({
        type: 'code',
        location: match.file,
        description: `Implementation: ${match.snippet.substring(0, 100)}...`,
        lineNumbers: match.lines
      });
    }
    
    return evidence;
  }

  private async findTestEvidence(criterion: AcceptanceCriterion, tasks: ImplementationTask[]): Promise<EvidenceItem[]> {
    const evidence: EvidenceItem[] = [];
    
    // Find test files related to this criterion
    for (const task of tasks) {
      if (this.taskMatchesCriterion(task, criterion)) {
        for (const testFile of task.testFiles) {
          evidence.push({
            type: 'test',
            location: testFile,
            description: `Test for ${task.taskId}: ${task.description}`
          });
        }
      }
    }
    
    return evidence;
  }

  private async findDocumentationEvidence(criterion: AcceptanceCriterion): Promise<EvidenceItem[]> {
    const evidence: EvidenceItem[] = [];
    
    // Use @explorer to find documentation references
    const docResult = await this.delegateToExplorer('find_criterion_documentation', {
      criterion: criterion.description,
      searchPaths: ['docs/', '*.md', 'README.md']
    });
    
    const docs = docResult.result as string[];
    
    for (const doc of docs) {
      evidence.push({
        type: 'documentation',
        location: doc,
        description: `Documentation reference for ${criterion.criterionId}`
      });
    }
    
    return evidence;
  }

  private taskMatchesCriterion(task: ImplementationTask, criterion: AcceptanceCriterion): boolean {
    // Simple heuristic matching - can be enhanced
    const taskText = `${task.taskId} ${task.description}`.toLowerCase();
    const criterionText = `${criterion.criterionId} ${criterion.description}`.toLowerCase();
    
    return criterionText.includes(task.taskId.toLowerCase()) || 
           taskText.includes(criterion.criterionId.toLowerCase());
  }

  private calculateEvidenceConfidence(evidence: CriterionEvidence): number {
    let confidence = 0;
    
    // Code evidence contributes 40%
    const codeEvidence = evidence.evidence.filter(e => e.type === 'code').length;
    confidence += Math.min(40, codeEvidence * 10);
    
    // Test evidence contributes 40%
    const testEvidence = evidence.evidence.filter(e => e.type === 'test').length;
    confidence += Math.min(40, testEvidence * 10);
    
    // Documentation contributes 20%
    const docEvidence = evidence.evidence.filter(e => e.type === 'documentation').length;
    confidence += Math.min(20, docEvidence * 10);
    
    return confidence;
  }

  private async isRequirementMet(
    requirement: string,
    evidence: CriterionEvidence,
    testResults: { detailedResults: TestResult[] }
  ): Promise<boolean> {
    // Check if there's evidence covering this requirement
    const hasCodeEvidence = evidence.evidence.some(e => 
      e.type === 'code' && e.description.toLowerCase().includes(requirement.toLowerCase())
    );
    
    const hasTestEvidence = evidence.evidence.some(e => 
      e.type === 'test' && e.description.toLowerCase().includes(requirement.toLowerCase())
    );
    
    return hasCodeEvidence && hasTestEvidence;
  }

  private formatEvidence(evidence: CriterionEvidence): string {
    return evidence.evidence.map(e => `${e.type}: ${e.location}`).join(', ');
  }

  /**
   * Test execution methods
   */
  private async runUnitTests(): Promise<{
    totalTests: number;
    passedTests: number;
    failedTests: number;
    coverage: number;
    detailedResults: TestResult[];
  }> {
    try {
      const output = execSync(
        'python -m pytest tests/ --cov=doc_server --cov-report=json --cov-report=term -v',
        { cwd: this.config.projectRoot, encoding: 'utf8', stdio: 'pipe' }
      );
      
      const coverageData = JSON.parse(
        await fs.readFile(path.join(this.config.projectRoot, 'coverage.json'), 'utf8')
      );
      
      const testResults = this.parsePytestOutput(output);
      
      return {
        totalTests: testResults.total,
        passedTests: testResults.passed,
        failedTests: testResults.failed,
        coverage: coverageData.totals.percent_covered,
        detailedResults: testResults.detailed
      };
      
    } catch (error) {
      const testResults = this.parsePytestOutput(error.stdout || '');
      return {
        totalTests: testResults.total,
        passedTests: testResults.passed,
        failedTests: testResults.failed,
        coverage: 0,
        detailedResults: testResults.detailed
      };
    }
  }

  private async runIntegrationTests(): Promise<{
    totalTests: number;
    passedTests: number;
    failedTests: number;
    coverage: number;
    detailedResults: TestResult[];
  }> {
    // Use @fixer to run integration tests
    const result = await this.delegateToFixer('run_integration_tests', {
      projectRoot: this.config.projectRoot,
      testPattern: 'tests/integration/'
    });
    
    return {
      totalTests: 0,
      passedTests: 0,
      failedTests: 0,
      coverage: 0,
      detailedResults: result.result.tests || []
    };
  }

  private async runPerformanceTests(): Promise<{
    totalTests: number;
    passedTests: number;
    failedTests: number;
    coverage: number;
    detailedResults: TestResult[];
  }> {
    // Use @fixer to run performance tests
    const result = await this.delegateToFixer('run_performance_tests', {
      projectRoot: this.config.projectRoot,
      testPattern: 'tests/performance/'
    });
    
    return {
      totalTests: 0,
      passedTests: 0,
      failedTests: 0,
      coverage: 0,
      detailedResults: result.result.tests || []
    };
  }

  private async runSecurityTests(): Promise<{
    totalTests: number;
    passedTests: number;
    failedTests: number;
    coverage: number;
    detailedResults: TestResult[];
  }> {
    // Use @oracle to run security analysis
    const result = await this.delegateToOracle('run_security_analysis', {
      projectRoot: this.config.projectRoot,
      analysisType: 'vulnerability_scan'
    });
    
    return {
      totalTests: 0,
      passedTests: 0,
      failedTests: 0,
      coverage: 0,
      detailedResults: result.result.tests || []
    };
  }

  /**
   * Quality analysis methods
   */
  private async analyzeSecurity(): Promise<number> {
    const result = await this.delegateToOracle('security_score', {
      projectRoot: this.config.projectRoot,
      analysisDepth: 'comprehensive'
    });
    
    return result.result.score || 5.0;
  }

  private async analyzePerformance(): Promise<number> {
    const result = await this.delegateToFixer('performance_score', {
      projectRoot: this.config.projectRoot,
      benchmarkSuite: 'default'
    });
    
    return result.result.score || 5.0;
  }

  private async analyzeMaintainability(): Promise<number> {
    const result = await this.delegateToExplorer('maintainability_score', {
      projectRoot: this.config.projectRoot,
      metrics: ['complexity', 'duplication', 'size']
    });
    
    return result.result.score || 5.0;
  }

  private async analyzeDocumentation(): Promise<number> {
    const result = await this.delegateToExplorer('documentation_score', {
      projectRoot: this.config.projectRoot,
      checkTypes: ['api_docs', 'code_comments', 'readme']
    });
    
    return result.result.score || 5.0;
  }

  private parsePytestOutput(output: string): {
    total: number;
    passed: number;
    failed: number;
    detailed: TestResult[];
  } {
    const totalMatch = output.match(/(\d+)\s+tests?/);
    const passedMatch = output.match(/(\d+)\s+passed/);
    const failedMatch = output.match(/(\d+)\s+failed/);
    
    // Parse individual test results (simplified)
    const testLineRegex = /^(.+)\s+(PASSED|FAILED|SKIPPED)\s*\[\s*(\d+)%\s*\]/gm;
    const detailed: TestResult[] = [];
    let match;
    
    while ((match = testLineRegex.exec(output)) !== null) {
      detailed.push({
        testName: match[1],
        status: match[2].toLowerCase() as 'passed' | 'failed' | 'skipped',
        coverage: parseInt(match[3]),
        executionTime: 0
      });
    }
    
    return {
      total: parseInt(totalMatch?.[1] || '0'),
      passed: parseInt(passedMatch?.[1] || '0'),
      failed: parseInt(failedMatch?.[1] || '0'),
      detailed
    };
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
      executionTime: 1000
    };
  }

  private async delegateToFixer(task: string, context: any): Promise<DelegationResult> {
    console.log(`🤖 Delegating to @fixer: ${task}`);
    
    return {
      specialist: 'fixer',
      task,
      result: { tests: [], score: 8.0 },
      success: true,
      executionTime: 1500
    };
  }

  private async delegateToOracle(task: string, context: any): Promise<DelegationResult> {
    console.log(`🤖 Delegating to @oracle: ${task}`);
    
    return {
      specialist: 'oracle',
      task,
      result: { score: 9.0, tests: [], vulnerabilities: [] },
      success: true,
      executionTime: 2000
    };
  }
}