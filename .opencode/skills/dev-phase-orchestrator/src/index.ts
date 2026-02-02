/**
 * Main entry point for the dev-phase-orchestrator skill
 */

import { DevPhaseOrchestrator } from './orchestrator';
import { OrchestratorConfig, HumanApproval } from './types';
import * as path from 'path';

/**
 * Initialize the orchestrator with configuration
 */
export function initializeOrchestrator(config?: Partial<OrchestratorConfig>): DevPhaseOrchestrator {
  const defaultConfig: OrchestratorConfig = {
    projectRoot: process.cwd(),
    specsPath: path.join(process.cwd(), 'specs'),
    maxConcurrentDelegations: 3,
    humanApprovalRequired: true,
    stateRetentionHours: 24,
    enableMetrics: true
  };

  const finalConfig = { ...defaultConfig, ...config };
  return new DevPhaseOrchestrator(finalConfig);
}

/**
 * Main skill function - handles user input and orchestrates workflow
 */
export async function handleUserRequest(input: string, config?: Partial<OrchestratorConfig>): Promise<{
  response: string;
  nextAction?: string;
  sessionId?: string;
}> {
  const orchestrator = initializeOrchestrator(config);

  // Parse user input to determine intent
  const phaseMatch = input.match(/start\s+(?:implementation\s+of\s+)?phase\s+(\d+\.\d+)/i);
  
  if (phaseMatch) {
    const phaseId = phaseMatch[1];
    
    try {
      const result = await orchestrator.startPhase(phaseId, input);
      
      return {
        response: `🚀 Started Phase ${phaseId} development workflow. Alignment verification completed.`,
        nextAction: result.nextAction,
        sessionId: result.sessionId
      };
    } catch (error) {
      return {
        response: `❌ Failed to start Phase ${phaseId}: ${error instanceof Error ? error.message : 'Unknown error'}`
      };
    }
  }

  // Handle approval requests
  const approvalMatch = input.match(/proceed\s+to\s+(implementation|verification|commit)/i);
  if (approvalMatch) {
    const action = approvalMatch[1].toLowerCase();
    const sessionId = extractSessionId(input);
    
    if (!sessionId) {
      return {
        response: '❌ Session ID required to proceed. Please include the session ID in your request.'
      };
    }

    const approval: HumanApproval = {
      phase: 'approval' as any,
      action: `proceed_to_${action}` as any,
      timestamp: new Date().toISOString(),
      comments: input
    };

    try {
      switch (action) {
        case 'implementation': {
          const result = await orchestrator.proceedToImplementation(sessionId, approval);
          return {
            response: `🔧 Implementation phase completed for Phase ${result.result.phaseId}.`,
            nextAction: result.nextAction,
            sessionId
          };
        }
        
        case 'verification': {
          const result = await orchestrator.proceedToVerification(sessionId, approval);
          return {
            response: `🔍 Verification phase completed for Phase ${result.result.phaseId}.`,
            nextAction: result.nextAction,
            sessionId
          };
        }
        
        case 'commit': {
          const result = await orchestrator.proceedToCommit(sessionId, approval);
          return {
            response: `💾 Commit phase completed. Workflow complete! Commit hash: ${result.result.commitHash}`,
            sessionId
          };
        }
      }
    } catch (error) {
      return {
        response: `❌ Failed to proceed to ${action}: ${error instanceof Error ? error.message : 'Unknown error'}`
      };
    }
  }

  // Handle status requests
  if (input.toLowerCase().includes('status') || input.toLowerCase().includes('progress')) {
    const sessionId = extractSessionId(input);
    const history = orchestrator.getPhaseHistory(sessionId);
    
    if (history.length === 0) {
      return {
        response: 'No active sessions found. Start a new phase with "I\'m going to start implementation of phase X.Y"'
      };
    }

    const latestSession = sessionId ? history.filter(h => h.sessionId === sessionId) : history;
    const latest = latestSession[latestSession.length - 1];
    
    return {
      response: `📊 Workflow Status:\n` +
               `Session: ${latest.sessionId}\n` +
               `Current Phase: ${latest.toPhase}\n` +
               `Last Updated: ${latest.timestamp}\n` +
               `Total Phases Completed: ${latestSession.length}`
    };
  }

  // Default response
  return {
    response: `👋 I'm the Dev Phase Orchestrator. I can help you with:\n\n` +
             `• Start a new phase: "I'm going to start implementation of phase X.Y"\n` +
             `• Proceed with approval: "proceed to implementation/verification/commit"\n` +
             `• Check status: "status" or "progress"\n\n` +
             `Each phase requires human approval before proceeding to the next step.`
  };
}

/**
 * Helper function to extract session ID from user input
 */
function extractSessionId(input: string): string | null {
  const sessionMatch = input.match(/phase-\d+\.\d+-\d{8}/);
  return sessionMatch ? sessionMatch[0] : null;
}

/**
 * Export main classes and functions
 */
export {
  DevPhaseOrchestrator,
  initializeOrchestrator
};

// Export all types for external use
export * from './types';
export * from './agents/alignment';
export * from './agents/implementation';
export * from './agents/verification';
export * from './agents/commit';