"""
Tests for the resilience module.

Tests retry decorators, circuit breakers, health checks, and
resource cleanup utilities.
"""

import time
import pytest
from pathlib import Path
import tempfile
import shutil

from didactic_engine.resilience import (
    RetryError,
    RetryConfig,
    retry_with_backoff,
    compute_delay,
    CircuitState,
    CircuitBreaker,
    HealthCheck,
    check_demucs_health,
    check_basic_pitch_health,
    check_essentia_health,
    check_ffmpeg_health,
    run_all_health_checks,
    resource_cleanup,
    ProcessingCheckpoint,
)


class TestComputeDelay:
    """Tests for delay computation with backoff."""

    def test_base_delay(self):
        """First attempt should use base delay."""
        delay = compute_delay(0, base_delay=1.0, max_delay=60.0, exponential_base=2.0, jitter=0.0)
        assert delay == 1.0

    def test_exponential_growth(self):
        """Delay should grow exponentially."""
        delay1 = compute_delay(0, 1.0, 60.0, 2.0, 0.0)
        delay2 = compute_delay(1, 1.0, 60.0, 2.0, 0.0)
        delay3 = compute_delay(2, 1.0, 60.0, 2.0, 0.0)
        
        assert delay1 == 1.0
        assert delay2 == 2.0
        assert delay3 == 4.0

    def test_max_delay_cap(self):
        """Delay should be capped at max_delay."""
        delay = compute_delay(10, 1.0, 5.0, 2.0, 0.0)
        assert delay == 5.0

    def test_jitter_adds_variability(self):
        """Jitter should add randomness to delay."""
        delays = [compute_delay(1, 1.0, 60.0, 2.0, 0.5) for _ in range(10)]
        # With jitter, delays should vary
        assert len(set(delays)) > 1


class TestRetryDecorator:
    """Tests for retry_with_backoff decorator."""

    def test_success_no_retry(self):
        """Successful function should not retry."""
        call_count = 0
        
        @retry_with_backoff(max_retries=3, base_delay=0.01)
        def successful_func():
            nonlocal call_count
            call_count += 1
            return "success"
        
        result = successful_func()
        assert result == "success"
        assert call_count == 1

    def test_retry_then_success(self):
        """Function should retry on failure then succeed."""
        call_count = 0
        
        @retry_with_backoff(max_retries=3, base_delay=0.01)
        def flaky_func():
            nonlocal call_count
            call_count += 1
            if call_count < 3:
                raise ValueError("Transient error")
            return "success"
        
        result = flaky_func()
        assert result == "success"
        assert call_count == 3

    def test_exhausted_retries(self):
        """Should raise RetryError when all retries exhausted."""
        @retry_with_backoff(max_retries=2, base_delay=0.01)
        def always_fails():
            raise ValueError("Always fails")
        
        with pytest.raises(RetryError) as exc_info:
            always_fails()
        
        assert exc_info.value.attempts == 3
        assert exc_info.value.last_exception is not None

    def test_specific_exceptions(self):
        """Should only retry on specified exceptions."""
        call_count = 0
        
        @retry_with_backoff(
            max_retries=3, 
            base_delay=0.01,
            retryable_exceptions=(ValueError,)
        )
        def specific_error():
            nonlocal call_count
            call_count += 1
            raise TypeError("Not retryable")
        
        with pytest.raises(TypeError):
            specific_error()
        
        # Should not retry on TypeError
        assert call_count == 1

    def test_on_retry_callback(self):
        """Callback should be called on each retry."""
        retry_attempts = []
        
        def on_retry(attempt, exc):
            retry_attempts.append((attempt, str(exc)))
        
        call_count = 0
        
        @retry_with_backoff(max_retries=2, base_delay=0.01, on_retry=on_retry)
        def flaky_func():
            nonlocal call_count
            call_count += 1
            if call_count < 3:
                raise ValueError(f"Attempt {call_count}")
            return "success"
        
        result = flaky_func()
        assert result == "success"
        assert len(retry_attempts) == 2


class TestCircuitBreaker:
    """Tests for CircuitBreaker pattern."""

    def test_initial_state_closed(self):
        """Circuit should start in closed state."""
        breaker = CircuitBreaker()
        assert breaker.state == CircuitState.CLOSED
        assert breaker.is_closed

    def test_opens_after_failures(self):
        """Circuit should open after failure threshold."""
        breaker = CircuitBreaker(failure_threshold=3)
        
        for _ in range(3):
            try:
                with breaker:
                    raise ValueError("failure")
            except ValueError:
                pass
        
        assert breaker.state == CircuitState.OPEN
        assert breaker.is_open

    def test_rejects_when_open(self):
        """Open circuit should reject calls."""
        breaker = CircuitBreaker(failure_threshold=1, recovery_timeout=60.0)
        
        # Trigger open state
        try:
            with breaker:
                raise ValueError("failure")
        except ValueError:
            pass
        
        assert breaker.is_open
        
        # Should reject immediately
        with pytest.raises(RuntimeError) as exc_info:
            with breaker:
                pass
        
        assert "OPEN" in str(exc_info.value)

    def test_half_open_after_timeout(self):
        """Circuit should enter half-open after recovery timeout."""
        breaker = CircuitBreaker(failure_threshold=1, recovery_timeout=0.01, success_threshold=1)
        
        # Trigger open state
        try:
            with breaker:
                raise ValueError("failure")
        except ValueError:
            pass
        
        assert breaker.is_open
        
        # Wait for recovery timeout
        time.sleep(0.02)
        
        # Next call should work (half-open allows one through)
        with breaker:
            pass
        
        # Should close after success (success_threshold=1)
        assert breaker.is_closed

    def test_success_resets_failures(self):
        """Success should reset failure count."""
        breaker = CircuitBreaker(failure_threshold=3)
        
        # Some failures
        for _ in range(2):
            try:
                with breaker:
                    raise ValueError("failure")
            except ValueError:
                pass
        
        # Success resets count
        with breaker:
            pass
        
        # Should still be closed (failures reset)
        assert breaker.is_closed

    def test_manual_reset(self):
        """Manual reset should close circuit."""
        breaker = CircuitBreaker(failure_threshold=1)
        
        try:
            with breaker:
                raise ValueError("failure")
        except ValueError:
            pass
        
        assert breaker.is_open
        
        breaker.reset()
        
        assert breaker.is_closed

    def test_protect_decorator(self):
        """protect() should work as decorator."""
        breaker = CircuitBreaker(failure_threshold=2)
        
        call_count = 0
        
        @breaker.protect
        def protected_func():
            nonlocal call_count
            call_count += 1
            return "success"
        
        result = protected_func()
        assert result == "success"
        assert call_count == 1


class TestHealthChecks:
    """Tests for health check utilities."""

    def test_health_check_structure(self):
        """Health check should have required fields."""
        check = HealthCheck(
            name="test",
            available=True,
            version="1.0.0",
            details="Test check",
        )
        
        assert check.name == "test"
        assert check.available is True
        assert check.version == "1.0.0"
        assert check.details == "Test check"
        assert check.check_time > 0

    def test_ffmpeg_health_check(self):
        """FFmpeg health check should run without error."""
        result = check_ffmpeg_health()
        assert isinstance(result, HealthCheck)
        assert result.name == "ffmpeg"

    def test_run_all_health_checks(self):
        """Should run all health checks and return dict."""
        results = run_all_health_checks()
        
        assert isinstance(results, dict)
        assert "demucs" in results
        assert "basic-pitch" in results
        assert "essentia" in results
        assert "ffmpeg" in results
        
        for name, check in results.items():
            assert isinstance(check, HealthCheck)
            assert check.name == name


class TestResourceCleanup:
    """Tests for resource cleanup context manager."""

    def test_cleanup_on_error(self):
        """Should clean up on error when configured."""
        with tempfile.TemporaryDirectory() as tmpdir:
            test_file = Path(tmpdir) / "test.txt"
            test_file.write_text("test")
            
            assert test_file.exists()
            
            try:
                with resource_cleanup([test_file], cleanup_on_error=True):
                    raise ValueError("test error")
            except ValueError:
                pass
            
            assert not test_file.exists()

    def test_no_cleanup_on_success(self):
        """Should not clean up on success by default."""
        with tempfile.TemporaryDirectory() as tmpdir:
            test_file = Path(tmpdir) / "test.txt"
            test_file.write_text("test")
            
            with resource_cleanup([test_file], cleanup_on_error=True, cleanup_on_success=False):
                pass
            
            assert test_file.exists()

    def test_cleanup_directory(self):
        """Should clean up directories recursively."""
        with tempfile.TemporaryDirectory() as tmpdir:
            test_dir = Path(tmpdir) / "subdir"
            test_dir.mkdir()
            (test_dir / "file.txt").write_text("test")
            
            try:
                with resource_cleanup([test_dir], cleanup_on_error=True):
                    raise ValueError("test error")
            except ValueError:
                pass
            
            assert not test_dir.exists()


class TestProcessingCheckpoint:
    """Tests for checkpoint functionality."""

    def test_mark_complete(self):
        """Should mark steps as complete."""
        checkpoint = ProcessingCheckpoint(song_id="test_song")
        
        checkpoint.mark_complete("step1", result={"data": "value"})
        
        assert checkpoint.is_complete("step1")
        assert not checkpoint.is_complete("step2")
        assert checkpoint.get_result("step1") == {"data": "value"}

    def test_save_and_load(self):
        """Should save and load checkpoint."""
        with tempfile.TemporaryDirectory() as tmpdir:
            checkpoint_path = Path(tmpdir) / "checkpoint.json"
            
            checkpoint = ProcessingCheckpoint(song_id="test_song")
            checkpoint.mark_complete("step1")
            checkpoint.mark_complete("step2")
            checkpoint.save(checkpoint_path)
            
            loaded = ProcessingCheckpoint.load(checkpoint_path)
            
            assert loaded.song_id == "test_song"
            assert loaded.is_complete("step1")
            assert loaded.is_complete("step2")
            assert loaded.last_step == "step2"

    def test_completed_steps_order(self):
        """Should preserve step completion order."""
        checkpoint = ProcessingCheckpoint(song_id="test_song")
        
        checkpoint.mark_complete("step1")
        checkpoint.mark_complete("step2")
        checkpoint.mark_complete("step3")
        
        assert checkpoint.completed_steps == ["step1", "step2", "step3"]
        assert checkpoint.last_step == "step3"


class TestRetryConfig:
    """Tests for RetryConfig dataclass."""

    def test_defaults(self):
        """Should have sensible defaults."""
        config = RetryConfig()
        
        assert config.max_retries == 3
        assert config.base_delay == 1.0
        assert config.max_delay == 60.0
        assert config.exponential_base == 2.0
        assert config.jitter == 0.1
        assert config.retryable_exceptions == (Exception,)

    def test_custom_config(self):
        """Should accept custom values."""
        config = RetryConfig(
            max_retries=5,
            base_delay=0.5,
            max_delay=30.0,
            retryable_exceptions=(ValueError, IOError),
        )
        
        assert config.max_retries == 5
        assert config.base_delay == 0.5
        assert config.max_delay == 30.0
        assert config.retryable_exceptions == (ValueError, IOError)
