"""Execution plan for multi-step agentic workflows."""

from dataclasses import dataclass, field
from typing import List, Dict, Any, Optional, Union
from enum import Enum
import uuid


class StepType(Enum):
    """Types of execution steps available in the agentic system."""
    DISCOVER_DOCUMENT = "discover_document"
    EXTRACT_CONTENT = "extract_content"
    ANALYZE_DECISIONS = "analyze_decisions"
    COMPARE_ACROSS_DOCS = "compare_across_docs"
    FIND_RELATIONSHIPS = "find_relationships"
    SYNTHESIZE_FINDINGS = "synthesize_findings"
    RETURN_METADATA = "return_metadata"
    RETURN_FILE_PATH = "return_file_path"


class StepStatus(Enum):
    """Status of an execution step."""
    PENDING = "pending"
    IN_PROGRESS = "in_progress"
    COMPLETED = "completed"
    FAILED = "failed"
    SKIPPED = "skipped"


@dataclass
class StepResult:
    """Standardized result format for agent communication."""
    step_id: str
    step_type: StepType
    status: StepStatus
    data: Dict[str, Any] = field(default_factory=dict)
    error: Optional[str] = None
    confidence: float = 1.0
    execution_time: float = 0.0
    agent_used: Optional[str] = None
    
    def is_successful(self) -> bool:
        """Check if step completed successfully."""
        return self.status == StepStatus.COMPLETED and self.error is None
    
    def get_value(self, key: str, default: Any = None) -> Any:
        """Safely get a value from the result data."""
        return self.data.get(key, default)


@dataclass
class ExecutionStep:
    """Represents a single step in an execution plan."""
    id: str = field(default_factory=lambda: str(uuid.uuid4()))
    step_type: StepType = StepType.DISCOVER_DOCUMENT
    agent_name: str = "discovery"
    parameters: Dict[str, Any] = field(default_factory=dict)
    dependencies: List[str] = field(default_factory=list)
    status: StepStatus = StepStatus.PENDING
    result: Optional[StepResult] = None
    max_retries: int = 2
    retry_count: int = 0
    
    def can_execute(self, completed_steps: List[str]) -> bool:
        """Check if this step can be executed based on dependencies."""
        return all(dep_id in completed_steps for dep_id in self.dependencies)
    
    def is_ready(self) -> bool:
        """Check if step is ready to execute."""
        return self.status == StepStatus.PENDING
    
    def is_complete(self) -> bool:
        """Check if step is completed (successfully or failed)."""
        return self.status in [StepStatus.COMPLETED, StepStatus.FAILED, StepStatus.SKIPPED]
    
    def can_retry(self) -> bool:
        """Check if step can be retried."""
        return self.status == StepStatus.FAILED and self.retry_count < self.max_retries


@dataclass
class ExecutionPlan:
    """Structured representation of multi-step workflows."""
    id: str = field(default_factory=lambda: str(uuid.uuid4()))
    query: str = ""
    intent: str = ""
    steps: List[ExecutionStep] = field(default_factory=list)
    context: Dict[str, Any] = field(default_factory=dict)
    status: str = "created"
    created_at: float = field(default_factory=lambda: __import__('time').time())
    
    def add_step(self, step_type: StepType, agent_name: str, 
                 parameters: Dict[str, Any], dependencies: List[str] = None) -> str:
        """Add a new step to the execution plan."""
        step = ExecutionStep(
            step_type=step_type,
            agent_name=agent_name,
            parameters=parameters,
            dependencies=dependencies or []
        )
        self.steps.append(step)
        return step.id
    
    def get_next_steps(self) -> List[ExecutionStep]:
        """Get the next steps that can be executed."""
        completed_step_ids = [step.id for step in self.steps if step.status == StepStatus.COMPLETED]
        
        return [
            step for step in self.steps 
            if step.is_ready() and step.can_execute(completed_step_ids)
        ]
    
    def get_step(self, step_id: str) -> Optional[ExecutionStep]:
        """Get a specific step by ID."""
        for step in self.steps:
            if step.id == step_id:
                return step
        return None
    
    def update_step_status(self, step_id: str, status: StepStatus, 
                          result: Optional[StepResult] = None) -> bool:
        """Update the status of a specific step."""
        step = self.get_step(step_id)
        if step:
            step.status = status
            if result:
                step.result = result
            return True
        return False
    
    def is_complete(self) -> bool:
        """Check if all steps are complete."""
        return all(step.is_complete() for step in self.steps)
    
    def has_failed_steps(self) -> bool:
        """Check if any steps have failed."""
        return any(step.status == StepStatus.FAILED for step in self.steps)
    
    def get_results(self) -> List[StepResult]:
        """Get all completed step results."""
        return [step.result for step in self.steps 
                if step.result and step.result.is_successful()]
    
    def add_context(self, key: str, value: Any) -> None:
        """Add context information that can be shared across steps."""
        self.context[key] = value
    
    def get_context(self, key: str, default: Any = None) -> Any:
        """Get context information."""
        return self.context.get(key, default)
    
    def get_execution_summary(self) -> Dict[str, Any]:
        """Get a summary of the execution plan."""
        completed = len([s for s in self.steps if s.status == StepStatus.COMPLETED])
        failed = len([s for s in self.steps if s.status == StepStatus.FAILED])
        pending = len([s for s in self.steps if s.status == StepStatus.PENDING])
        
        return {
            "plan_id": self.id,
            "query": self.query,
            "intent": self.intent,
            "total_steps": len(self.steps),
            "completed_steps": completed,
            "failed_steps": failed,
            "pending_steps": pending,
            "status": self.status,
            "is_complete": self.is_complete(),
            "has_failures": self.has_failed_steps(),
            "execution_time": __import__('time').time() - self.created_at
        }