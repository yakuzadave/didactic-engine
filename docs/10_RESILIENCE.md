# Resilience Patterns for Didactic Engine

This document describes the resilience features added to make the audio processing pipeline more robust and production-ready.

## Overview

The `didactic_engine.resilience` module provides several patterns for building resilient applications:

1. **Retry with Exponential Backoff** - Automatically retry failed operations
2. **Circuit Breaker** - Prevent cascading failures from external dependencies
3. **Health Checks** - Monitor availability of optional dependencies
4. **Resource Cleanup** - Ensure temporary files are cleaned up
5. **Processing Checkpoints** - Enable resume capability for long pipelines

## Quick Start

```python
from didactic_engine import (
    retry_with_backoff,
    CircuitBreaker,
    run_all_health_checks,
    print_health_report,
    resource_cleanup,
    ProcessingCheckpoint,
)

# Check dependencies before running
print_health_report()

# Use retry for flaky operations
@retry_with_backoff(max_retries=3, base_delay=1.0)
def process_audio(path):
    return load_and_analyze(path)

# Use circuit breaker for external services
breaker = CircuitBreaker(failure_threshold=5)

@breaker.protect
def call_external_api():
    return api_client.transcribe(audio)
```

## Retry with Exponential Backoff

The `retry_with_backoff` decorator automatically retries failed function calls with configurable backoff:

```python
from didactic_engine.resilience import retry_with_backoff, RetryConfig

# Basic usage
@retry_with_backoff(max_retries=3)
def flaky_operation():
    # May fail sometimes
    pass

# Advanced configuration
@retry_with_backoff(
    max_retries=5,
    base_delay=0.5,          # Start with 0.5s delay
    max_delay=30.0,          # Cap at 30s
    exponential_base=2.0,    # Double each retry
    jitter=0.1,              # Add 10% random jitter
    retryable_exceptions=(IOError, TimeoutError),  # Only retry these
    on_retry=lambda attempt, exc: log.warning(f"Retry {attempt}: {exc}"),
)
def network_call():
    return requests.get(url)
```

### Parameters

| Parameter | Default | Description |
|-----------|---------|-------------|
| `max_retries` | 3 | Maximum retry attempts (0 = no retries) |
| `base_delay` | 1.0 | Initial delay in seconds |
| `max_delay` | 60.0 | Maximum delay cap |
| `exponential_base` | 2.0 | Multiplier for each retry |
| `jitter` | 0.1 | Random factor (0-1) to prevent thundering herd |
| `retryable_exceptions` | `(Exception,)` | Exception types that trigger retries |
| `on_retry` | None | Callback before each retry |

### Delay Calculation

```
delay = min(base_delay * (exponential_base ** attempt), max_delay) + jitter
```

For default settings with 3 retries:
- Attempt 1: ~1.0s
- Attempt 2: ~2.0s  
- Attempt 3: ~4.0s

## Circuit Breaker

The circuit breaker prevents cascading failures by stopping calls to a failing service:

```python
from didactic_engine.resilience import CircuitBreaker, CircuitState

# Create circuit breaker
breaker = CircuitBreaker(
    failure_threshold=5,    # Open after 5 failures
    recovery_timeout=60.0,  # Wait 60s before testing
    success_threshold=2,    # Need 2 successes to close
)

# Use as context manager
try:
    with breaker:
        result = external_service.call()
except RuntimeError as e:
    if "OPEN" in str(e):
        print("Service unavailable, using fallback")
        result = fallback_value

# Or use as decorator
@breaker.protect
def call_service():
    return external_service.call()
```

### Circuit States

```
CLOSED ──(failures >= threshold)──► OPEN
   ▲                                   │
   │                                   │ (timeout elapsed)
   │                                   ▼
   └────(successes >= threshold)── HALF_OPEN
                                       │
                                       │ (failure)
                                       ▼
                                     OPEN
```

- **CLOSED**: Normal operation, calls pass through
- **OPEN**: Failing, calls rejected immediately
- **HALF_OPEN**: Testing if service recovered

### Pre-configured Breakers

The module provides pre-configured circuit breakers for external tools:

```python
from didactic_engine.resilience import demucs_circuit, basic_pitch_circuit

@demucs_circuit.protect
def separate_stems(audio_path):
    separator = StemSeparator()
    return separator.separate(audio_path, output_dir)
```

## Health Checks

Check availability of optional dependencies:

```python
from didactic_engine.resilience import (
    run_all_health_checks,
    print_health_report,
    check_demucs_health,
    check_basic_pitch_health,
    check_essentia_health,
    check_ffmpeg_health,
)

# Check all dependencies
results = run_all_health_checks()
for name, check in results.items():
    print(f"{name}: {'✓' if check.available else '✗'} - {check.details}")

# Print formatted report
print_health_report()

# Output:
# ============================================================
# Didactic Engine - Dependency Health Check
# ============================================================
#
# ✓ DEMUCS
#   Available: True
#   Details: CLI available
#
# ✗ BASIC-PITCH
#   Available: False
#   Details: basic-pitch command not found on PATH
# ...
```

### HealthCheck Structure

```python
@dataclass
class HealthCheck:
    name: str              # Dependency name
    available: bool        # Whether available
    version: str | None    # Optional version string
    details: str           # Additional details/error message
    check_time: float      # When check was performed
```

## Resource Cleanup

Ensure temporary files are cleaned up even on errors:

```python
from didactic_engine.resilience import resource_cleanup
from pathlib import Path

temp_dir = Path("/tmp/processing")

# Clean up on error only
with resource_cleanup([temp_dir], cleanup_on_error=True):
    process_audio(input_file, temp_dir)
    # If exception raised, temp_dir is deleted

# Clean up always
with resource_cleanup(
    [temp_file1, temp_file2],
    cleanup_on_error=True,
    cleanup_on_success=True,
):
    # Cleaned up regardless of success/failure
    pass
```

## Processing Checkpoints

Enable resume capability for long-running pipelines:

```python
from didactic_engine.resilience import ProcessingCheckpoint

# Create checkpoint
checkpoint = ProcessingCheckpoint(song_id="my_song")

# Mark steps complete
checkpoint.mark_complete("ingest", result=audio_data)
checkpoint.mark_complete("analyze", result=analysis)

# Save to disk
checkpoint.save("checkpoints/my_song.json")

# Later, resume from checkpoint
checkpoint = ProcessingCheckpoint.load("checkpoints/my_song.json")

if checkpoint.is_complete("ingest"):
    audio_data = checkpoint.get_result("ingest")
else:
    audio_data = ingest_audio(path)
    checkpoint.mark_complete("ingest", result=audio_data)
```

### Checkpoint Structure

```python
@dataclass
class ProcessingCheckpoint:
    song_id: str                           # Song being processed
    completed_steps: list[str]             # Completed step names
    last_step: str                         # Name of last completed step
    intermediate_results: dict[str, Any]   # Cached results
    checkpoint_time: float                 # When checkpoint created
```

## Batch Processing with Resilience

The enhanced batch processor uses these patterns:

```python
from didactic_engine.batch import BatchProcessor, BatchConfig
from didactic_engine.resilience import RetryConfig

config = BatchConfig(
    max_workers=4,
    retry_config=RetryConfig(max_retries=3, base_delay=1.0),
    progress_enabled=True,
    skip_existing=True,      # Skip already processed files
    cleanup_on_error=True,   # Clean up partial outputs
)

processor = BatchProcessor(config)
results = processor.process(
    input_files=[Path("song1.wav"), Path("song2.wav")],
    output_dir=Path("output"),
    analysis_sr=22050,
)

results.print_summary()
# ============================================================
# Batch Processing Summary
# ============================================================
# Total files:    10
# Successful:     9 (90.0%)
# Failed:         1 (10.0%)
# Skipped:        0
# Total time:     125.3s
# Avg per file:   12.5s
```

## Best Practices

### 1. Use Retries for Transient Failures

```python
# Good: Retry network operations
@retry_with_backoff(max_retries=3, retryable_exceptions=(IOError, TimeoutError))
def download_model():
    return requests.get(model_url)

# Bad: Don't retry on validation errors
@retry_with_backoff()  # Don't do this
def validate_audio(path):
    if not path.exists():
        raise FileNotFoundError()  # Won't succeed on retry
```

### 2. Configure Circuit Breakers Per Service

```python
# Good: Separate breakers for different services
demucs_breaker = CircuitBreaker(failure_threshold=3, recovery_timeout=120)
api_breaker = CircuitBreaker(failure_threshold=5, recovery_timeout=30)

# Bad: One breaker for everything
global_breaker = CircuitBreaker()  # Services affect each other
```

### 3. Check Health Before Batch Jobs

```python
from didactic_engine.resilience import run_all_health_checks

def validate_environment():
    checks = run_all_health_checks()
    
    # Require FFmpeg
    if not checks["ffmpeg"].available:
        raise RuntimeError("FFmpeg required for preprocessing")
    
    # Warn about optional deps
    if not checks["demucs"].available:
        logger.warning("Demucs not available, will skip stem separation")
```

### 4. Use Checkpoints for Long Pipelines

```python
def process_large_dataset(files, checkpoint_dir):
    for file in files:
        checkpoint_path = checkpoint_dir / f"{file.stem}.json"
        
        if checkpoint_path.exists():
            checkpoint = ProcessingCheckpoint.load(checkpoint_path)
        else:
            checkpoint = ProcessingCheckpoint(song_id=file.stem)
        
        try:
            if not checkpoint.is_complete("process"):
                result = process_file(file)
                checkpoint.mark_complete("process", result)
                checkpoint.save(checkpoint_path)
        except KeyboardInterrupt:
            checkpoint.save(checkpoint_path)  # Save progress
            raise
```

## Integration with Pipeline

The resilience features integrate with the main pipeline:

```python
from didactic_engine import AudioPipeline, PipelineConfig
from didactic_engine.resilience import retry_with_backoff, resource_cleanup

@retry_with_backoff(max_retries=2, base_delay=5.0)
def process_with_retry(input_path, output_dir):
    temp_dir = output_dir / "temp"
    
    with resource_cleanup([temp_dir], cleanup_on_error=True):
        cfg = PipelineConfig(
            song_id=input_path.stem,
            input_wav=input_path,
            out_dir=output_dir,
        )
        
        pipeline = AudioPipeline(cfg)
        return pipeline.run()
```

## See Also

- [Batch Processing Examples](../examples/batch_processing_example.py)
- [Pipeline Architecture](01_ARCHITECTURE.md)
- [Debugging Guide](03_DEBUGGING.md)
