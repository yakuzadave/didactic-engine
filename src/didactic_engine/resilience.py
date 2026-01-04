"""
Resilience utilities for robust pipeline execution.

This module provides tools for making the pipeline more resilient:
- Retry decorators with exponential backoff
- Circuit breaker for external tool failures
- Health checks for dependencies
- Resource cleanup utilities

These patterns help the pipeline gracefully handle transient failures,
network issues, and resource exhaustion.

Example:
    >>> from didactic_engine.resilience import retry_with_backoff, CircuitBreaker
    >>>
    >>> @retry_with_backoff(max_retries=3, base_delay=1.0)
    ... def flaky_operation():
    ...     # Some operation that might fail
    ...     pass
    >>>
    >>> breaker = CircuitBreaker(failure_threshold=5, recovery_timeout=60.0)
    >>> with breaker:
    ...     run_external_tool()
"""

from __future__ import annotations

import functools
import logging
import os
import shutil
import time
from contextlib import contextmanager
from dataclasses import dataclass, field
from enum import Enum
from pathlib import Path
from typing import (
    Any,
    Callable,
    Dict,
    Generic,
    List,
    Optional,
    Sequence,
    Tuple,
    Type,
    TypeVar,
    Union,
)

logger = logging.getLogger(__name__)
logger.addHandler(logging.NullHandler())

T = TypeVar("T")
F = TypeVar("F", bound=Callable[..., Any])


class RetryError(Exception):
    """Raised when all retry attempts have been exhausted."""

    def __init__(
        self,
        message: str,
        attempts: int,
        last_exception: Optional[Exception] = None,
    ):
        super().__init__(message)
        self.attempts = attempts
        self.last_exception = last_exception


class CircuitState(Enum):
    """Circuit breaker states."""
    CLOSED = "closed"  # Normal operation
    OPEN = "open"  # Failing, rejecting calls
    HALF_OPEN = "half_open"  # Testing if service recovered


@dataclass
class RetryConfig:
    """Configuration for retry behavior.
    
    Attributes:
        max_retries: Maximum number of retry attempts (0 = no retries).
        base_delay: Initial delay between retries in seconds.
        max_delay: Maximum delay cap in seconds.
        exponential_base: Base for exponential backoff (e.g., 2.0 = double each time).
        jitter: Random jitter factor (0.0-1.0) to add to delays.
        retryable_exceptions: Tuple of exception types that trigger retries.
            Defaults to all exceptions.
    """
    max_retries: int = 3
    base_delay: float = 1.0
    max_delay: float = 60.0
    exponential_base: float = 2.0
    jitter: float = 0.1
    retryable_exceptions: Tuple[Type[Exception], ...] = (Exception,)


def compute_delay(
    attempt: int,
    base_delay: float,
    max_delay: float,
    exponential_base: float,
    jitter: float,
) -> float:
    """Compute delay for a retry attempt with exponential backoff and jitter.
    
    Args:
        attempt: Current attempt number (0-indexed).
        base_delay: Initial delay in seconds.
        max_delay: Maximum delay cap in seconds.
        exponential_base: Base for exponential growth.
        jitter: Random jitter factor (0.0-1.0).
    
    Returns:
        Delay in seconds.
    """
    import random
    
    # Exponential backoff
    delay = base_delay * (exponential_base ** attempt)
    
    # Cap at max_delay
    delay = min(delay, max_delay)
    
    # Add jitter
    if jitter > 0:
        jitter_amount = delay * jitter * random.random()
        delay += jitter_amount
    
    return delay


def retry_with_backoff(
    max_retries: int = 3,
    base_delay: float = 1.0,
    max_delay: float = 60.0,
    exponential_base: float = 2.0,
    jitter: float = 0.1,
    retryable_exceptions: Tuple[Type[Exception], ...] = (Exception,),
    on_retry: Optional[Callable[[int, Exception], None]] = None,
) -> Callable[[F], F]:
    """Decorator for retry with exponential backoff.
    
    Retries a function on failure with exponentially increasing delays.
    Useful for transient failures like network issues or temporary
    resource unavailability.
    
    Args:
        max_retries: Maximum retry attempts (0 = no retries).
        base_delay: Initial delay between retries in seconds.
        max_delay: Maximum delay cap in seconds.
        exponential_base: Multiplier for each retry (2.0 = double).
        jitter: Random factor (0-1) added to delays to prevent thundering herd.
        retryable_exceptions: Exception types that trigger retries.
        on_retry: Optional callback(attempt, exception) called before each retry.
    
    Returns:
        Decorated function with retry logic.
    
    Example:
        >>> @retry_with_backoff(max_retries=3, base_delay=0.5)
        ... def call_external_api():
        ...     response = requests.get("https://api.example.com")
        ...     return response.json()
        
        >>> # With specific exceptions
        >>> @retry_with_backoff(
        ...     max_retries=5,
        ...     retryable_exceptions=(IOError, TimeoutError)
        ... )
        ... def read_file(path):
        ...     return Path(path).read_text()
    """
    def decorator(func: F) -> F:
        @functools.wraps(func)
        def wrapper(*args: Any, **kwargs: Any) -> Any:
            last_exception: Optional[Exception] = None
            
            for attempt in range(max_retries + 1):
                try:
                    return func(*args, **kwargs)
                except retryable_exceptions as e:
                    last_exception = e
                    
                    if attempt < max_retries:
                        delay = compute_delay(
                            attempt, base_delay, max_delay, exponential_base, jitter
                        )
                        
                        logger.warning(
                            "Attempt %d/%d failed for %s: %s. Retrying in %.2fs...",
                            attempt + 1,
                            max_retries + 1,
                            func.__name__,
                            str(e),
                            delay,
                        )
                        
                        if on_retry:
                            on_retry(attempt, e)
                        
                        time.sleep(delay)
                    else:
                        logger.error(
                            "All %d attempts failed for %s. Last error: %s",
                            max_retries + 1,
                            func.__name__,
                            str(e),
                        )
                        raise RetryError(
                            f"All {max_retries + 1} attempts failed for {func.__name__}",
                            attempts=max_retries + 1,
                            last_exception=e,
                        ) from e
            
            # Should not reach here, but satisfy type checker
            raise RetryError(
                f"Retry logic error in {func.__name__}",
                attempts=max_retries + 1,
                last_exception=last_exception,
            )
        
        return wrapper  # type: ignore
    
    return decorator


@dataclass
class CircuitBreaker:
    """Circuit breaker pattern implementation for external dependencies.
    
    Prevents cascading failures by stopping calls to a failing service
    and allowing it time to recover.
    
    States:
        - CLOSED: Normal operation, calls pass through
        - OPEN: Service failing, calls rejected immediately  
        - HALF_OPEN: Testing if service recovered
    
    Attributes:
        failure_threshold: Number of failures before opening circuit.
        recovery_timeout: Seconds to wait before testing recovery.
        success_threshold: Successes needed in HALF_OPEN to close circuit.
        
    Example:
        >>> breaker = CircuitBreaker(failure_threshold=5, recovery_timeout=30.0)
        >>> 
        >>> def call_service():
        ...     with breaker:
        ...         return external_api_call()
        >>>
        >>> # Or use as decorator
        >>> @breaker.protect
        ... def call_service():
        ...     return external_api_call()
    """
    failure_threshold: int = 5
    recovery_timeout: float = 60.0
    success_threshold: int = 2
    
    # Internal state
    _state: CircuitState = field(default=CircuitState.CLOSED, init=False)
    _failure_count: int = field(default=0, init=False)
    _success_count: int = field(default=0, init=False)
    _last_failure_time: float = field(default=0.0, init=False)
    _name: str = field(default="circuit", init=False)
    
    @property
    def state(self) -> CircuitState:
        """Current circuit state."""
        return self._state
    
    @property
    def is_closed(self) -> bool:
        """Whether circuit is closed (normal operation)."""
        return self._state == CircuitState.CLOSED
    
    @property
    def is_open(self) -> bool:
        """Whether circuit is open (rejecting calls)."""
        return self._state == CircuitState.OPEN
    
    def __enter__(self) -> "CircuitBreaker":
        """Context manager entry - check if call should proceed."""
        self._check_state()
        return self
    
    def __exit__(
        self,
        exc_type: Optional[Type[BaseException]],
        exc_val: Optional[BaseException],
        exc_tb: Any,
    ) -> bool:
        """Context manager exit - record success or failure."""
        if exc_type is None:
            self._record_success()
        else:
            self._record_failure()
        return False  # Don't suppress exceptions
    
    def _check_state(self) -> None:
        """Check circuit state and raise if open."""
        if self._state == CircuitState.OPEN:
            # Check if recovery timeout has elapsed
            if time.time() - self._last_failure_time >= self.recovery_timeout:
                logger.info(
                    "Circuit '%s' entering HALF_OPEN state after timeout",
                    self._name,
                )
                self._state = CircuitState.HALF_OPEN
                self._success_count = 0
            else:
                remaining = self.recovery_timeout - (time.time() - self._last_failure_time)
                raise RuntimeError(
                    f"Circuit '{self._name}' is OPEN. "
                    f"Service unavailable. Retry in {remaining:.1f}s."
                )
    
    def _record_success(self) -> None:
        """Record a successful call."""
        if self._state == CircuitState.HALF_OPEN:
            self._success_count += 1
            if self._success_count >= self.success_threshold:
                logger.info(
                    "Circuit '%s' closing after %d successful calls",
                    self._name,
                    self._success_count,
                )
                self._state = CircuitState.CLOSED
                self._failure_count = 0
                self._success_count = 0
        elif self._state == CircuitState.CLOSED:
            # Reset failure count on success
            self._failure_count = 0
    
    def _record_failure(self) -> None:
        """Record a failed call."""
        self._failure_count += 1
        self._last_failure_time = time.time()
        
        if self._state == CircuitState.HALF_OPEN:
            logger.warning(
                "Circuit '%s' reopening after failure in HALF_OPEN state",
                self._name,
            )
            self._state = CircuitState.OPEN
            self._success_count = 0
        elif self._state == CircuitState.CLOSED:
            if self._failure_count >= self.failure_threshold:
                logger.warning(
                    "Circuit '%s' opening after %d failures",
                    self._name,
                    self._failure_count,
                )
                self._state = CircuitState.OPEN
    
    def reset(self) -> None:
        """Manually reset circuit to closed state."""
        self._state = CircuitState.CLOSED
        self._failure_count = 0
        self._success_count = 0
        self._last_failure_time = 0.0
        logger.info("Circuit '%s' manually reset", self._name)
    
    def protect(self, func: F) -> F:
        """Decorator to protect a function with this circuit breaker.
        
        Example:
            >>> breaker = CircuitBreaker()
            >>> 
            >>> @breaker.protect
            ... def call_api():
            ...     return requests.get(url)
        """
        @functools.wraps(func)
        def wrapper(*args: Any, **kwargs: Any) -> Any:
            with self:
                return func(*args, **kwargs)
        return wrapper  # type: ignore


@dataclass
class HealthCheck:
    """Health check results for a dependency.
    
    Attributes:
        name: Name of the dependency.
        available: Whether the dependency is available.
        version: Optional version string.
        details: Additional details or error message.
        check_time: Time when check was performed.
    """
    name: str
    available: bool
    version: Optional[str] = None
    details: str = ""
    check_time: float = field(default_factory=time.time)


def check_demucs_health() -> HealthCheck:
    """Check if Demucs is available and get version info."""
    import subprocess
    
    name = "demucs"
    
    if shutil.which("demucs") is None:
        return HealthCheck(
            name=name,
            available=False,
            details="demucs command not found on PATH",
        )
    
    try:
        result = subprocess.run(
            ["demucs", "--help"],
            capture_output=True,
            text=True,
            timeout=10,
        )
        if result.returncode == 0:
            # Try to extract version from help text
            version = None
            for line in result.stdout.split("\n"):
                if "version" in line.lower():
                    version = line.strip()
                    break
            return HealthCheck(
                name=name,
                available=True,
                version=version,
                details="CLI available",
            )
        else:
            return HealthCheck(
                name=name,
                available=False,
                details=f"demucs --help failed: {result.stderr}",
            )
    except subprocess.TimeoutExpired:
        return HealthCheck(
            name=name,
            available=False,
            details="demucs --help timed out",
        )
    except Exception as e:
        return HealthCheck(
            name=name,
            available=False,
            details=str(e),
        )


def check_basic_pitch_health() -> HealthCheck:
    """Check if Basic Pitch is available and get version info."""
    import subprocess
    
    name = "basic-pitch"
    
    if shutil.which("basic-pitch") is None:
        return HealthCheck(
            name=name,
            available=False,
            details="basic-pitch command not found on PATH",
        )
    
    try:
        result = subprocess.run(
            ["basic-pitch", "--help"],
            capture_output=True,
            text=True,
            timeout=10,
        )
        if result.returncode == 0:
            return HealthCheck(
                name=name,
                available=True,
                details="CLI available",
            )
        else:
            return HealthCheck(
                name=name,
                available=False,
                details=f"basic-pitch --help failed: {result.stderr}",
            )
    except subprocess.TimeoutExpired:
        return HealthCheck(
            name=name,
            available=False,
            details="basic-pitch --help timed out",
        )
    except Exception as e:
        return HealthCheck(
            name=name,
            available=False,
            details=str(e),
        )


def check_essentia_health() -> HealthCheck:
    """Check if Essentia is available."""
    name = "essentia"
    
    try:
        import essentia
        version = getattr(essentia, "__version__", "unknown")
        return HealthCheck(
            name=name,
            available=True,
            version=version,
            details="Python module available",
        )
    except ImportError as e:
        return HealthCheck(
            name=name,
            available=False,
            details=f"Import failed: {e}",
        )


def check_ffmpeg_health() -> HealthCheck:
    """Check if FFmpeg is available (required by pydub)."""
    import subprocess
    
    name = "ffmpeg"
    
    if shutil.which("ffmpeg") is None:
        return HealthCheck(
            name=name,
            available=False,
            details="ffmpeg command not found on PATH",
        )
    
    try:
        result = subprocess.run(
            ["ffmpeg", "-version"],
            capture_output=True,
            text=True,
            timeout=5,
        )
        if result.returncode == 0:
            # Extract version from first line
            version = result.stdout.split("\n")[0] if result.stdout else None
            return HealthCheck(
                name=name,
                available=True,
                version=version,
                details="CLI available",
            )
        else:
            return HealthCheck(
                name=name,
                available=False,
                details=f"ffmpeg -version failed",
            )
    except Exception as e:
        return HealthCheck(
            name=name,
            available=False,
            details=str(e),
        )


def run_all_health_checks() -> Dict[str, HealthCheck]:
    """Run all health checks and return results.
    
    Returns:
        Dictionary mapping dependency names to their health check results.
    
    Example:
        >>> results = run_all_health_checks()
        >>> for name, check in results.items():
        ...     status = "✓" if check.available else "✗"
        ...     print(f"{status} {name}: {check.details}")
    """
    return {
        "demucs": check_demucs_health(),
        "basic-pitch": check_basic_pitch_health(),
        "essentia": check_essentia_health(),
        "ffmpeg": check_ffmpeg_health(),
    }


def print_health_report() -> None:
    """Print a formatted health check report to stdout."""
    print("=" * 60)
    print("Didactic Engine - Dependency Health Check")
    print("=" * 60)
    
    results = run_all_health_checks()
    
    for name, check in results.items():
        status = "✓" if check.available else "✗"
        print(f"\n{status} {name.upper()}")
        print(f"  Available: {check.available}")
        if check.version:
            print(f"  Version: {check.version}")
        print(f"  Details: {check.details}")
    
    available_count = sum(1 for c in results.values() if c.available)
    total_count = len(results)
    
    print("\n" + "=" * 60)
    print(f"Summary: {available_count}/{total_count} dependencies available")
    print("=" * 60)


@contextmanager
def resource_cleanup(
    paths: Optional[Sequence[Union[str, Path]]] = None,
    cleanup_on_error: bool = True,
    cleanup_on_success: bool = False,
):
    """Context manager for automatic resource cleanup.
    
    Ensures temporary files and directories are cleaned up after
    pipeline operations, even if errors occur.
    
    Args:
        paths: Paths to clean up (files or directories).
        cleanup_on_error: Whether to clean up if an exception occurs.
        cleanup_on_success: Whether to clean up on successful completion.
    
    Example:
        >>> with resource_cleanup([temp_dir], cleanup_on_error=True):
        ...     process_audio(input_file, temp_dir)
        ...     # temp_dir cleaned up if exception raised
    """
    paths_to_clean: List[Path] = []
    if paths:
        paths_to_clean = [Path(p) for p in paths]
    
    error_occurred = False
    
    try:
        yield
    except Exception:
        error_occurred = True
        raise
    finally:
        should_cleanup = (error_occurred and cleanup_on_error) or (
            not error_occurred and cleanup_on_success
        )
        
        if should_cleanup:
            for path in paths_to_clean:
                try:
                    if path.exists():
                        if path.is_dir():
                            shutil.rmtree(path)
                            logger.debug("Cleaned up directory: %s", path)
                        else:
                            path.unlink()
                            logger.debug("Cleaned up file: %s", path)
                except Exception as e:
                    logger.warning("Failed to clean up %s: %s", path, e)


@dataclass
class ProcessingCheckpoint:
    """Checkpoint for pipeline state to enable resume functionality.
    
    Stores the state of a pipeline run so it can be resumed after
    interruption.
    
    Attributes:
        song_id: Song being processed.
        completed_steps: List of completed step names.
        last_step: Name of the last completed step.
        intermediate_results: Cached results from completed steps.
        checkpoint_time: When checkpoint was created.
    """
    song_id: str
    completed_steps: List[str] = field(default_factory=list)
    last_step: str = ""
    intermediate_results: Dict[str, Any] = field(default_factory=dict)
    checkpoint_time: float = field(default_factory=time.time)
    
    def mark_complete(self, step_name: str, result: Any = None) -> None:
        """Mark a step as completed.
        
        Args:
            step_name: Name of the completed step.
            result: Optional result to cache.
        """
        if step_name not in self.completed_steps:
            self.completed_steps.append(step_name)
        self.last_step = step_name
        if result is not None:
            self.intermediate_results[step_name] = result
        self.checkpoint_time = time.time()
    
    def is_complete(self, step_name: str) -> bool:
        """Check if a step has been completed."""
        return step_name in self.completed_steps
    
    def get_result(self, step_name: str) -> Optional[Any]:
        """Get cached result for a step."""
        return self.intermediate_results.get(step_name)
    
    def save(self, path: Union[str, Path]) -> None:
        """Save checkpoint to JSON file.
        
        Args:
            path: Path to save checkpoint file.
        """
        import json
        
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        
        data = {
            "song_id": self.song_id,
            "completed_steps": self.completed_steps,
            "last_step": self.last_step,
            "checkpoint_time": self.checkpoint_time,
            # Note: intermediate_results may not be JSON-serializable
            # in all cases, so we skip it for now
        }
        
        with open(path, "w") as f:
            json.dump(data, f, indent=2)
        
        logger.info("Saved checkpoint to %s", path)
    
    @classmethod
    def load(cls, path: Union[str, Path]) -> "ProcessingCheckpoint":
        """Load checkpoint from JSON file.
        
        Args:
            path: Path to checkpoint file.
            
        Returns:
            Loaded checkpoint.
        """
        import json
        
        with open(path, "r") as f:
            data = json.load(f)
        
        return cls(
            song_id=data["song_id"],
            completed_steps=data.get("completed_steps", []),
            last_step=data.get("last_step", ""),
            checkpoint_time=data.get("checkpoint_time", time.time()),
        )


def with_checkpoint(
    checkpoint_dir: Union[str, Path],
    step_name: str,
) -> Callable[[F], F]:
    """Decorator to add checkpoint support to a pipeline step.
    
    Skips execution if step is already complete according to checkpoint.
    Marks step complete after successful execution.
    
    Args:
        checkpoint_dir: Directory containing checkpoint files.
        step_name: Name of this step for checkpoint tracking.
    
    Returns:
        Decorated function.
    
    Example:
        >>> @with_checkpoint("output/checkpoints", "step_1_ingest")
        ... def ingest_audio(song_id, input_path):
        ...     return load_audio(input_path)
    """
    def decorator(func: F) -> F:
        @functools.wraps(func)
        def wrapper(*args: Any, **kwargs: Any) -> Any:
            # Try to get song_id from args/kwargs
            song_id = kwargs.get("song_id")
            if song_id is None and args:
                # Assume first arg might be config or song_id
                first_arg = args[0]
                if hasattr(first_arg, "song_id"):
                    song_id = first_arg.song_id
                elif isinstance(first_arg, str):
                    song_id = first_arg
            
            if song_id is None:
                # No song_id found, just run without checkpoint
                return func(*args, **kwargs)
            
            checkpoint_path = Path(checkpoint_dir) / f"{song_id}_checkpoint.json"
            
            # Try to load existing checkpoint
            checkpoint: Optional[ProcessingCheckpoint] = None
            if checkpoint_path.exists():
                try:
                    checkpoint = ProcessingCheckpoint.load(checkpoint_path)
                except Exception as e:
                    logger.warning("Failed to load checkpoint: %s", e)
            
            if checkpoint is None:
                checkpoint = ProcessingCheckpoint(song_id=song_id)
            
            # Check if step already complete
            if checkpoint.is_complete(step_name):
                cached = checkpoint.get_result(step_name)
                if cached is not None:
                    logger.info(
                        "Skipping %s (already complete, using cached result)",
                        step_name,
                    )
                    return cached
                logger.info(
                    "Step %s marked complete but no cached result",
                    step_name,
                )
            
            # Execute step
            result = func(*args, **kwargs)
            
            # Mark complete and save
            checkpoint.mark_complete(step_name, result)
            try:
                checkpoint.save(checkpoint_path)
            except Exception as e:
                logger.warning("Failed to save checkpoint: %s", e)
            
            return result
        
        return wrapper  # type: ignore
    
    return decorator


# Pre-configured circuit breakers for common external tools
demucs_circuit = CircuitBreaker(
    failure_threshold=3,
    recovery_timeout=120.0,  # 2 minutes
    success_threshold=1,
)
demucs_circuit._name = "demucs"

basic_pitch_circuit = CircuitBreaker(
    failure_threshold=3,
    recovery_timeout=120.0,
    success_threshold=1,
)
basic_pitch_circuit._name = "basic_pitch"


__all__ = [
    "RetryError",
    "RetryConfig",
    "retry_with_backoff",
    "compute_delay",
    "CircuitState",
    "CircuitBreaker",
    "HealthCheck",
    "check_demucs_health",
    "check_basic_pitch_health",
    "check_essentia_health",
    "check_ffmpeg_health",
    "run_all_health_checks",
    "print_health_report",
    "resource_cleanup",
    "ProcessingCheckpoint",
    "with_checkpoint",
    "demucs_circuit",
    "basic_pitch_circuit",
]
