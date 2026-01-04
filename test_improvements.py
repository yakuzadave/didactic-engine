"""Quick validation tests for implemented improvements."""

from pathlib import Path
from src.didactic_engine.config import PipelineConfig

def test_config_validation():
    """Test that config validation works."""
    print("Testing config validation...")

    # Test 1: Negative sample rate should fail
    try:
        cfg = PipelineConfig(
            song_id='test',
            input_wav=Path('test.wav'),
            out_dir=Path('output'),
            analysis_sr=-1
        )
        print("  FAIL: Negative sample rate should have been rejected")
        return False
    except ValueError as e:
        if "analysis_sr must be positive" in str(e):
            print("  PASS: Negative sample rate rejected")
        else:
            print(f"  FAIL: Wrong error message: {e}")
            return False

    # Test 2: Zero time signature should fail
    try:
        cfg = PipelineConfig(
            song_id='test',
            input_wav=Path('test.wav'),
            out_dir=Path('output'),
            time_signature_num=0
        )
        print("  FAIL: Zero time signature should have been rejected")
        return False
    except ValueError as e:
        if "time_signature_num must be positive" in str(e):
            print("  PASS: Zero time signature rejected")
        else:
            print(f"  FAIL: Wrong error message: {e}")
            return False

    # Test 3: Invalid backend should fail
    try:
        cfg = PipelineConfig(
            song_id='test',
            input_wav=Path('test.wav'),
            out_dir=Path('output'),
            basic_pitch_backend='invalid'
        )
        print("  FAIL: Invalid backend should have been rejected")
        return False
    except ValueError as e:
        if "basic_pitch_backend must be one of" in str(e):
            print("  PASS: Invalid backend rejected")
        else:
            print(f"  FAIL: Wrong error message: {e}")
            return False

    # Test 4: Valid config should succeed
    try:
        cfg = PipelineConfig(
            song_id='test',
            input_wav=Path('test.wav'),
            out_dir=Path('output')
        )
        print("  PASS: Valid config accepted")
    except Exception as e:
        print(f"  FAIL: Valid config rejected: {e}")
        return False

    # Test 5: Timeout defaults
    cfg = PipelineConfig(
        song_id='test',
        input_wav=Path('test.wav'),
        out_dir=Path('output')
    )
    if cfg.demucs_timeout_s == 3600.0:
        print("  PASS: Demucs timeout default is 3600.0")
    else:
        print(f"  FAIL: Demucs timeout default is {cfg.demucs_timeout_s}")
        return False

    if cfg.basic_pitch_timeout_s == 1800.0:
        print("  PASS: Basic Pitch timeout default is 1800.0")
    else:
        print(f"  FAIL: Basic Pitch timeout default is {cfg.basic_pitch_timeout_s}")
        return False

    print("\nAll config validation tests PASSED!")
    return True

def test_resilience_imports():
    """Test that resilience features import correctly."""
    print("\nTesting resilience imports...")

    try:
        from src.didactic_engine.resilience import (
            retry_with_backoff,
            demucs_circuit,
            basic_pitch_circuit,
        )
        print("  PASS: Resilience imports successful")
        print(f"    - demucs_circuit state: {demucs_circuit.state.value}")
        print(f"    - basic_pitch_circuit state: {basic_pitch_circuit.state.value}")
        return True
    except ImportError as e:
        print(f"  FAIL: Resilience import failed: {e}")
        return False

def test_pipeline_imports():
    """Test that pipeline imports correctly with new features."""
    print("\nTesting pipeline imports...")

    try:
        from src.didactic_engine.pipeline import AudioPipeline
        print("  PASS: Pipeline imports successfully")

        # Check that _load_stem_with_retry exists
        if hasattr(AudioPipeline, '_load_stem_with_retry'):
            print("  PASS: Retry-wrapped stem loader exists")
        else:
            print("  FAIL: Retry-wrapped stem loader not found")
            return False

        return True
    except ImportError as e:
        print(f"  FAIL: Pipeline import failed: {e}")
        return False

if __name__ == '__main__':
    print("=" * 60)
    print("Testing Implemented Improvements")
    print("=" * 60)

    results = []
    results.append(test_config_validation())
    results.append(test_resilience_imports())
    results.append(test_pipeline_imports())

    print("\n" + "=" * 60)
    if all(results):
        print("ALL TESTS PASSED!")
        print("=" * 60)
    else:
        print("SOME TESTS FAILED")
        print("=" * 60)
        exit(1)
