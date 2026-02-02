/**
 * Specialist Delegation Service
 * Manages delegation to specialist agents (@explorer, @librarian, @oracle, @designer, @fixer)
 * with proper triggers, context management, and result handling
 */

import {
  SpecialistType,
  DelegationRequest,
  DelegationResult,
  OrchestratorError,
} from '../types';

/**
 * Delegation trigger patterns for each phase
 */
export const DELEGATION_TRIGGERS = {
  // Phase 1: Alignment
  alignment: {
    explorer: [
      'find_missing_patterns',
      'extract_phase_tasks', 
      'extract_relevant_requirements',
      'extract_phase_plan',
      'scope_boundaries',
      'analyze_codebase_patterns'
    ],
    oracle: [
      'architectural_decisions',
      'requirement_clarification',
      'analyze_plan_requirement_alignment'
    ],
    fixer: [],
    librarian: [],
    designer: []
  },

  // Phase 2: Implementation
  implementation: {
    explorer: [
      'analyze_existing_patterns',
      'analyze_codebase_structure',
      'find_similar_implementations',
      'validate_scope_boundaries'
    ],
    fixer: [
      'decompose_tasks',
      'implement_task',
      'run_targeted_tests',
      'parallel_implementation',
      'code_quality_analysis',
      'test_strategy_planning'
    ],
    librarian: [
      'get_api_docs',
      'library_usage_patterns',
      'best_practices_lookup'
    ],
    oracle: [
      'architectural_decisions',
      'complex_design_decisions',
      'performance_optimization'
    ],
    designer: []
  },

  // Phase 3: Verification
  verification: {
    explorer: [
      'extract_acceptance_criteria',
      'find_criterion_implementation',
      'find_criterion_documentation',
      'analyze_maintainability',
      'analyze_documentation'
    ],
    fixer: [
      'run_integration_tests',
      'run_performance_tests',
      'performance_analysis',
      'security_testing'
    ],
    librarian: [
      'verify_api_compliance',
      'library_version_checks'
    ],
    oracle: [
      'run_security_analysis',
      'security_assessment',
      'complex_validation'
    ],
    designer: []
  },

  // Phase 4: Commit
  commit: {
    explorer: [
      'validate_commit_message',
      'analyze_git_history',
      'branch_analysis'
    ],
    fixer: [
      'execute_git_commit',
      'git_push',
      'git_tag',
      'git_merge',
      'git_branch_delete',
      'conflict_resolution',
      'cleanup_operations'
    ],
    librarian: [],
    oracle: [],
    designer: []
  }
};

/**
 * Specialist Agent Interface
 */
interface SpecialistAgent {
  type: SpecialistType;
  capabilities: string[];
  execute(request: DelegationRequest): Promise<DelegationResult>;
}

/**
 * Specialist Delegation Manager
 */
export class SpecialistDelegationManager {
  private agents: Map<SpecialistType, SpecialistAgent>;
  private maxConcurrentDelegations: number;
  private activeDelegations: Map<string, Promise<DelegationResult>>;

  constructor(maxConcurrentDelegations: number = 3) {
    this.maxConcurrentDelegations = maxConcurrentDelegations;
    this.activeDelegations = new Map();
    this.agents = new Map();
    
    this.initializeAgents();
  }

  /**
   * Initialize specialist agents
   */
  private initializeAgents(): void {
    // Mock implementations for now - in real scenario these would connect to actual agents
    this.agents.set('explorer', new MockExplorerAgent());
    this.agents.set('librarian', new MockLibrarianAgent());
    this.agents.set('oracle', new MockOracleAgent());
    this.agents.set('designer', new MockDesignerAgent());
    this.agents.set('fixer', new MockFixerAgent());
  }

  /**
   * Delegate task to specialist agent
   */
  async delegate(request: DelegationRequest): Promise<DelegationResult> {
    // Check concurrency limit
    if (this.activeDelegations.size >= this.maxConcurrentDelegations) {
      throw new Error(`Maximum concurrent delegations (${this.maxConcurrentDelegations}) reached`);
    }

    // Get agent for the specialist type
    const agent = this.agents.get(request.specialist);
    if (!agent) {
      throw new Error(`Unknown specialist type: ${request.specialist}`);
    }

    // Validate agent capability
    if (!agent.capabilities.includes(request.task.split('_')[0])) {
      throw new Error(`Agent ${request.specialist} does not support task: ${request.task}`);
    }

    // Generate delegation ID
    const delegationId = `${request.specialist}-${request.task}-${Date.now()}`;

    // Execute delegation
    const delegationPromise = this.executeDelegation(agent, request, delegationId);
    
    // Track active delegation
    this.activeDelegations.set(delegationId, delegationPromise);

    try {
      const result = await delegationPromise;
      return result;
    } finally {
      // Clean up
      this.activeDelegations.delete(delegationId);
    }
  }

  /**
   * Execute delegation with error handling
   */
  private async executeDelegation(
    agent: SpecialistAgent,
    request: DelegationRequest,
    delegationId: string
  ): Promise<DelegationResult> {
    const startTime = Date.now();
    
    try {
      console.log(`🤖 Delegating to @${request.specialist}: ${request.task} (${delegationId})`);
      
      const result = await agent.execute(request);
      
      const executionTime = Date.now() - startTime;
      
      console.log(`✅ Delegation completed: @${request.specialist}:${request.task} in ${executionTime}ms`);
      
      return {
        ...result,
        executionTime
      };
      
    } catch (error) {
      const executionTime = Date.now() - startTime;
      
      console.error(`❌ Delegation failed: @${request.specialist}:${request.task} after ${executionTime}ms:`, error);
      
      return {
        specialist: request.specialist,
        task: request.task,
        result: null,
        success: false,
        error: error instanceof Error ? error.message : 'Unknown error',
        executionTime
      };
    }
  }

  /**
   * Parallel delegation for multiple independent tasks
   */
  async delegateParallel(requests: DelegationRequest[]): Promise<DelegationResult[]> {
    if (requests.length > this.maxConcurrentDelegations) {
      throw new Error(`Too many parallel requests. Max: ${this.maxConcurrentDelegations}`);
    }

    console.log(`🚀 Executing ${requests.length} parallel delegations`);

    const promises = requests.map(request => this.delegate(request));
    const results = await Promise.allSettled(promises);

    return results.map((result, index) => {
      if (result.status === 'fulfilled') {
        return result.value;
      } else {
        return {
          specialist: requests[index].specialist,
          task: requests[index].task,
          result: null,
          success: false,
          error: result.reason instanceof Error ? result.reason.message : 'Unknown error',
          executionTime: 0
        };
      }
    });
  }

  /**
   * Get delegation triggers for a phase
   */
  getTriggersForPhase(phase: keyof typeof DELEGATION_TRIGGERS): {
    [key in SpecialistType]?: string[];
  } {
    return DELEGATION_TRIGGERS[phase] || {};
  }

  /**
   * Check if a task can be delegated to a specialist
   */
  canDelegate(phase: keyof typeof DELEGATION_TRIGGERS, specialist: SpecialistType, task: string): boolean {
    const triggers = this.getTriggersForPhase(phase);
    const specialistTasks = triggers[specialist] || [];
    
    return specialistTasks.includes(task) || 
           specialistTasks.some(pattern => task.startsWith(pattern));
  }

  /**
   * Get active delegations status
   */
  getActiveDelegations(): Array<{
    id: string;
    specialist: SpecialistType;
    task: string;
    duration: number;
  }> {
    const status = [];
    
    for (const [id, promise] of this.activeDelegations) {
      // Note: In a real implementation, we'd track start times separately
      status.push({
        id,
        specialist: id.split('-')[0] as SpecialistType,
        task: id.split('-').slice(1, -1).join('-'),
        duration: 0 // Would be calculated from start time
      });
    }
    
    return status;
  }
}

/**
 * Mock Specialist Agent Implementations
 * In a real scenario, these would connect to actual specialist agents
 */

class MockExplorerAgent implements SpecialistAgent {
  type: SpecialistType = 'explorer';
  capabilities: string[] = ['find', 'extract', 'analyze', 'search', 'validate'];

  async execute(request: DelegationRequest): Promise<DelegationResult> {
    // Simulate exploration work
    await this.simulateWork(1000, 2000);
    
    const mockResults: { [key: string]: any } = {
      extract_phase_tasks: {
        tasks: [
          { taskId: '2.4.1', description: 'Implement document processor core', status: 'pending' },
          { taskId: '2.4.2', description: 'Add file format support', status: 'pending' }
        ]
      },
      find_missing_patterns: {
        patterns: ['PDF parsing', 'DOCX extraction', 'Error handling'],
        recommendations: ['Use pdfminer.six', 'Use python-docx', 'Add try-catch blocks']
      },
      analyze_codebase_patterns: {
        patterns: ['OOP structure', 'Test naming conventions', 'Import organization'],
        recommendations: ['Follow existing class structure', 'Use test_ prefix', 'Group imports by type']
      }
    };

    return {
      specialist: this.type,
      task: request.task,
      result: mockResults[request.task] || { success: true, message: 'Exploration completed' },
      success: true,
      executionTime: 0
    };
  }

  private async simulateWork(minMs: number, maxMs: number): Promise<void> {
    const delay = Math.random() * (maxMs - minMs) + minMs;
    await new Promise(resolve => setTimeout(resolve, delay));
  }
}

class MockLibrarianAgent implements SpecialistAgent {
  type: SpecialistType = 'librarian';
  capabilities: string[] = ['lookup', 'get', 'find', 'verify', 'check'];

  async execute(request: DelegationRequest): Promise<DelegationResult> {
    await this.simulateWork(800, 1500);
    
    const mockResults: { [key: string]: any } = {
      get_api_docs: {
        'pdfminer.six': 'Library for PDF text extraction with metadata support',
        'python-docx': 'Library for DOCX file manipulation and content extraction'
      },
      verify_api_compliance: {
        compliant: true,
        issues: [],
        recommendations: ['Use latest stable versions']
      }
    };

    return {
      specialist: this.type,
      task: request.task,
      result: mockResults[request.task] || { docs: 'API documentation retrieved' },
      success: true,
      executionTime: 0
    };
  }

  private async simulateWork(minMs: number, maxMs: number): Promise<void> {
    const delay = Math.random() * (maxMs - minMs) + minMs;
    await new Promise(resolve => setTimeout(resolve, delay));
  }
}

class MockOracleAgent implements SpecialistAgent {
  type: SpecialistType = 'oracle';
  capabilities: string[] = ['analyze', 'decide', 'architectural', 'complex', 'strategic'];

  async execute(request: DelegationRequest): Promise<DelegationResult> {
    await this.simulateWork(1500, 3000);
    
    const mockResults: { [key: string]: any } = {
      architectural_decisions: {
        decision: 'Use strategy pattern for file processors',
        reasoning: 'Allows easy extension for new file formats',
        alternatives: ['Factory pattern', 'Adapter pattern'],
        recommendation: 'Strategy pattern best fits extensibility requirements'
      },
      analyze_plan_requirement_alignment: {
        aligned: true,
        gaps: [],
        recommendations: ['Consider adding error handling specifications']
      },
      run_security_analysis: {
        score: 9.2,
        vulnerabilities: [],
        recommendations: ['Add input validation', 'Implement rate limiting']
      }
    };

    return {
      specialist: this.type,
      task: request.task,
      result: mockResults[request.task] || { analysis: 'Strategic analysis completed' },
      success: true,
      executionTime: 0
    };
  }

  private async simulateWork(minMs: number, maxMs: number): Promise<void> {
    const delay = Math.random() * (maxMs - minMs) + minMs;
    await new Promise(resolve => setTimeout(resolve, delay));
  }
}

class MockDesignerAgent implements SpecialistAgent {
  type: SpecialistType = 'designer';
  capabilities: string[] = ['design', 'ui', 'ux', 'visual', 'interface'];

  async execute(request: DelegationRequest): Promise<DelegationResult> {
    await this.simulateWork(1200, 2500);
    
    return {
      specialist: this.type,
      task: request.task,
      result: { design: 'UI/UX design recommendations provided' },
      success: true,
      executionTime: 0
    };
  }

  private async simulateWork(minMs: number, maxMs: number): Promise<void> {
    const delay = Math.random() * (maxMs - minMs) + minMs;
    await new Promise(resolve => setTimeout(resolve, delay));
  }
}

class MockFixerAgent implements SpecialistAgent {
  type: SpecialistType = 'fixer';
  capabilities: string[] = ['implement', 'execute', 'run', 'fix', 'test', 'build'];

  async execute(request: DelegationRequest): Promise<DelegationResult> {
    await this.simulateWork(2000, 4000);
    
    const mockResults: { [key: string]: any } = {
      implement_task: {
        success: true,
        implementedFiles: ['document_processor.py'],
        testFiles: ['test_document_processor.py']
      },
      run_integration_tests: {
        tests: [
          { name: 'test_pdf_processing', status: 'passed', coverage: 95 },
          { name: 'test_docx_processing', status: 'passed', coverage: 92 }
        ],
        totalTests: 2,
        passedTests: 2
      },
      execute_git_commit: {
        success: true,
        commitHash: 'abc123def456',
        filesCommitted: 3
      }
    };

    return {
      specialist: this.type,
      task: request.task,
      result: mockResults[request.task] || { success: true, message: 'Task completed' },
      success: true,
      executionTime: 0
    };
  }

  private async simulateWork(minMs: number, maxMs: number): Promise<void> {
    const delay = Math.random() * (maxMs - minMs) + minMs;
    await new Promise(resolve => setTimeout(resolve, delay));
  }
}

/**
 * Create global delegation manager instance
 */
export const delegationManager = new SpecialistDelegationManager();

/**
 * Convenience functions for delegation
 */
export async function delegateToExplorer(task: string, context?: any): Promise<DelegationResult> {
  return delegationManager.delegate({
    specialist: 'explorer',
    task,
    context
  });
}

export async function delegateToLibrarian(task: string, context?: any): Promise<DelegationResult> {
  return delegationManager.delegate({
    specialist: 'librarian',
    task,
    context
  });
}

export async function delegateToOracle(task: string, context?: any): Promise<DelegationResult> {
  return delegationManager.delegate({
    specialist: 'oracle',
    task,
    context
  });
}

export async function delegateToFixer(task: string, context?: any): Promise<DelegationResult> {
  return delegationManager.delegate({
    specialist: 'fixer',
    task,
    context
  });
}

export async function delegateToDesigner(task: string, context?: any): Promise<DelegationResult> {
  return delegationManager.delegate({
    specialist: 'designer',
    task,
    context
  });
}