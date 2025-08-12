"""Base agent interface for specialized agentic components."""

from abc import ABC, abstractmethod
from typing import Dict, Any, Optional
import logging
import time

from .execution_plan import ExecutionStep, StepResult, StepStatus, StepType
from .context import AgentContext

logger = logging.getLogger(__name__)


class BaseAgent(ABC):
    """Abstract base class for specialized agents in the agentic system."""
    
    def __init__(self, name: str):
        """Initialize the base agent.
        
        Args:
            name: Unique name for this agent
        """
        self.name = name
        self.agent_logger = logging.getLogger(f'agent.{name}')
    
    @abstractmethod
    def can_handle(self, step: ExecutionStep) -> bool:
        """Check if this agent can handle the given execution step.
        
        Args:
            step: ExecutionStep to evaluate
            
        Returns:
            True if this agent can handle the step, False otherwise
        """
        pass
    
    @abstractmethod
    def execute_step(self, step: ExecutionStep, context: AgentContext) -> StepResult:
        """Execute a specific step within the given context.
        
        Args:
            step: ExecutionStep to execute
            context: AgentContext for cross-step communication
            
        Returns:
            StepResult containing the execution results
        """
        pass
    
    def get_capabilities(self) -> list[str]:
        """Get list of capabilities this agent provides.
        
        Returns:
            List of capability strings
        """
        return []
    
    def validate_parameters(self, step: ExecutionStep) -> bool:
        """Validate that the step parameters are valid for this agent.
        
        Args:
            step: ExecutionStep to validate
            
        Returns:
            True if parameters are valid, False otherwise
        """
        return True
    
    def _create_success_result(self, step: ExecutionStep, data: Dict[str, Any], 
                              confidence: float = 1.0, execution_time: float = 0.0) -> StepResult:
        """Create a successful step result.
        
        Args:
            step: The executed step
            data: Result data
            confidence: Confidence score (0.0 to 1.0)
            execution_time: Time taken to execute
            
        Returns:
            StepResult indicating success
        """
        return StepResult(
            step_id=step.id,
            step_type=step.step_type,
            status=StepStatus.COMPLETED,
            data=data,
            confidence=confidence,
            execution_time=execution_time,
            agent_used=self.name
        )
    
    def _create_failure_result(self, step: ExecutionStep, error: str, 
                              execution_time: float = 0.0) -> StepResult:
        """Create a failed step result.
        
        Args:
            step: The executed step
            error: Error message
            execution_time: Time taken before failure
            
        Returns:
            StepResult indicating failure
        """
        return StepResult(
            step_id=step.id,
            step_type=step.step_type,
            status=StepStatus.FAILED,
            error=error,
            execution_time=execution_time,
            agent_used=self.name
        )
    
    def _log_step_start(self, step: ExecutionStep) -> None:
        """Log the start of step execution."""
        self.agent_logger.info(f"Starting execution of step {step.step_type.value} (ID: {step.id})")
    
    def _log_step_complete(self, step: ExecutionStep, result: StepResult) -> None:
        """Log the completion of step execution."""
        status = "SUCCESS" if result.is_successful() else "FAILED"
        self.agent_logger.info(f"Completed step {step.step_type.value} - {status} "
                              f"(confidence: {result.confidence:.2f}, time: {result.execution_time:.2f}s)")
    
    def _safe_execute_step(self, step: ExecutionStep, context: AgentContext) -> StepResult:
        """Safely execute a step with error handling and timing.
        
        Args:
            step: ExecutionStep to execute
            context: AgentContext for communication
            
        Returns:
            StepResult with execution results or error information
        """
        start_time = time.time()
        self._log_step_start(step)
        
        try:
            # Validate parameters first
            if not self.validate_parameters(step):
                return self._create_failure_result(
                    step, "Invalid parameters for step execution", 
                    time.time() - start_time
                )
            
            # Execute the step
            result = self.execute_step(step, context)
            
            # Ensure execution time is set
            if result.execution_time == 0.0:
                result.execution_time = time.time() - start_time
            
            # Log the result
            self._log_step_complete(step, result)
            
            # Add to context history
            if result.is_successful():
                context.add_to_history(f"{self.name} completed {step.step_type.value}")
                context.set_confidence(step.id, result.confidence)
            else:
                context.add_to_history(f"{self.name} failed {step.step_type.value}: {result.error}")
            
            return result
            
        except Exception as e:
            execution_time = time.time() - start_time
            error_msg = f"Unexpected error in {self.name}: {str(e)}"
            self.agent_logger.error(error_msg)
            
            result = self._create_failure_result(step, error_msg, execution_time)
            self._log_step_complete(step, result)
            context.add_to_history(f"{self.name} error in {step.step_type.value}: {str(e)}")
            
            return result