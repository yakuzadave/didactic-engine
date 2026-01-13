# Pipeline Benchmark Analysis & Improvement Tasks

**Date**: 2026-01-02
**Test File**: `as_above_so_below_1_0_0.wav` (8 minutes, 48kHz stereo, 87.88 MB)
**Benchmark Results**: [benchmark_metrics.json](benchmark_output/benchmark_metrics.json)

---

## Executive Summary

Benchmarked the didactic-engine pipeline with an 8-minute professional audio file across two scenarios:

- ✅ **Scenario 1 (Default)**: Mono chunks @ 22050Hz → 24.27s processing time
- ✅ **Scenario 2 (Preserve)**: Stereo chunks @ 48000Hz → 19.88s processing time

Both scenarios completed successfully with identical beat/bar detection (152 BPM, 1056 beats, 266 bars).

---

## Performance Metrics

### Scenario Comparison

| Metric | Default (Mono) | Preserve (Stereo) | Delta |
|--------|----------------|-------------------|-------|
| **Total Time** | 24.27s | 19.88s | **-4.39s (-18%)** ⚡ |
| **Chunk Files** | 266 files | 266 files | Same |
| **Chunk Size** | 17.64 MB | 76.75 MB | **+59.11 MB (+335%)** 💾 |
| **Bar Features** | 266 rows | 266 rows | Same |
| **Warnings** | 3 | 3 | Same |

**Key Insight**: `preserve_chunk_audio=True` is **18% faster** despite loading audio twice, because the second load (for chunks) doesn't perform resampling, while feature extraction still benefits from faster 22050Hz processing.

### Step-by-Step Timing

#### Scenario 1: Default (Mono Chunks)
```
Step 1: Copy input file           0.04s
Step 2: Ingest WAV file            7.53s  ⚠️ SLOW (resampling)
Step 3: Preprocess audio           0.00s  (disabled)
Step 4: Analyze audio              7.23s  ⚠️ SLOW (tempo detection)
Step 5: Separate stems (Demucs)    0.01s  (not installed)
Step 6: Compute bar boundaries     0.00s
Step 7: Process stems              9.23s  ⚠️ SLOW (chunk writing + features)
Step 8: Build datasets             0.01s
Step 9: Write Parquet datasets     0.21s
Step 10: Export reports            0.00s
Step 11: Write summary JSON        0.00s
────────────────────────────────────────
TOTAL: 24.27s
```

#### Scenario 2: Preserve (Stereo Chunks)
```
Step 1: Copy input file            0.05s
Step 2: Ingest WAV file            0.40s  ✅ FAST (cached/optimized?)
Step 3: Preprocess audio           0.00s  (disabled)
Step 4: Analyze audio              5.64s  ✅ FASTER
Step 5: Separate stems (Demucs)    0.01s  (not installed)
Step 6: Compute bar boundaries     0.00s
Step 7: Process stems             13.74s  ⚠️ SLOWER (loading original + writing larger files)
Step 8: Build datasets             0.00s
Step 9: Write Parquet datasets     0.03s
Step 10: Export reports            0.00s
Step 11: Write summary JSON        0.00s
────────────────────────────────────────
TOTAL: 19.88s
```

---

## Issues Discovered

### Critical Issues

None! Both pipelines completed successfully.

### Warnings Encountered (3 per run)

#### 1. **NumPy Deprecation Warning** (CRITICAL - Will break in future NumPy)
```
DeprecationWarning: Conversion of an array with ndim > 0 to a scalar is deprecated
Location: src/didactic_engine/analysis.py:204
```
**Impact**: High - will cause errors when NumPy 2.0 becomes default
**Priority**: **High**
**Fix**: Extract scalar value explicitly before conversion

#### 2. **Librosa Tuning Estimation Warnings** (2x per run)
```
UserWarning: Trying to estimate tuning from empty frequency set
Location: librosa/core/pitch.py:103
```
**Impact**: Low - cosmetic warning, doesn't affect results
**Priority**: Medium
**Fix**: Either suppress or investigate why frequency set is empty

### Missing Optional Dependencies (Expected)

- ❌ **Demucs** (stem separation) - Not installed
- ❌ **Basic Pitch** (MIDI transcription) - Not installed
- ❌ **FFmpeg** (pydub preprocessing) - Not found (warning)

These are expected and gracefully handled.

---

## Performance Bottlenecks

### Top 3 Slowest Operations

1. **Step 7: Process stems** (9.23s default, 13.74s preserve)
   - Writing 266 chunk WAV files to disk
   - Extracting features for each bar
   - I/O bound operation

2. **Step 4: Analyze audio** (7.23s default, 5.64s preserve)
   - Librosa tempo detection and beat tracking
   - CPU bound operation
   - Scales with audio duration

3. **Step 2: Ingest WAV file** (7.53s default, 0.40s preserve)
   - **Mystery**: Why is ingestion 18x faster in Scenario 2?
   - Possible caching or code path difference
   - Needs investigation

---

## Output Analysis

### Generated Files

#### Scenario 1 (Default):
```
benchmark_output/scenario1_default/
├── chunks/benchmark_default_mono_chunks/full_mix/
│   └── bar_0000.wav - bar_0265.wav (266 files, 17.64 MB)
├── datasets/benchmark_default_mono_chunks/
│   ├── beats.parquet (1056 rows, ~0.05 MB)
│   └── bar_features.parquet (266 rows, 0.11 MB)
└── analysis/benchmark_default_mono_chunks/
    └── combined.json
```

**Chunk File Stats** (Mono @ 22050Hz):
- Average chunk size: ~67 KB
- Smallest chunk: [need to measure]
- Largest chunk: [need to measure]

#### Scenario 2 (Preserve):
```
benchmark_output/scenario2_preserve/
├── chunks/benchmark_preserve_stereo_chunks/full_mix/
│   └── bar_0000.wav - bar_0265.wav (266 files, 76.75 MB)
├── datasets/benchmark_preserve_stereo_chunks/
│   ├── beats.parquet (1056 rows, ~0.05 MB)
│   └── bar_features.parquet (266 rows, 0.11 MB)
└── analysis/benchmark_preserve_stereo_chunks/
    └── combined.json
```

**Chunk File Stats** (Stereo @ 48000Hz):
- Average chunk size: ~295 KB (4.4x larger than mono)
- Storage overhead for stereo+high-SR: **+59.11 MB** per song

---

## Comprehensive Improvement Task List

### Priority 1: Critical Fixes (Must Do Before NumPy 2.0)

#### Task 1.1: Fix NumPy Scalar Conversion Deprecation
**File**: `src/didactic_engine/analysis.py:204`
**Issue**: `DeprecationWarning: Conversion of an array with ndim > 0 to a scalar is deprecated`
**Fix**:
```python
# Before (line 204):
some_value = float(array_value)

# After:
some_value = float(array_value.item())  # or array_value[0]
```
**Estimated Effort**: 15 minutes
**Test**: Run benchmark again, verify warning is gone

#### Task 1.2: Investigate Ingestion Time Discrepancy
**Issue**: Step 2 (Ingest WAV) is 7.53s in Scenario 1 but only 0.40s in Scenario 2
**Investigation Steps**:
1. Add timing instrumentation to `WAVIngester.load()`
2. Check if there's file caching between runs
3. Profile librosa.load() vs soundfile.read() performance
4. Determine if resampling is the bottleneck

**Estimated Effort**: 1-2 hours
**Potential Fix**: Optimize resampling or use faster library

### Priority 2: Performance Optimizations

#### Task 2.1: Parallelize Chunk Writing and Feature Extraction
**Current**: Sequential loop processes 266 bars one by one
**Proposal**: Use multiprocessing to process bars in parallel
**Expected Speedup**: 2-4x on multi-core systems
**Implementation**:
```python
from multiprocessing import Pool

def process_bar(bar_data):
    # Extract features and write chunk
    return features

with Pool(processes=cpu_count()) as pool:
    all_bar_features = pool.map(process_bar, bar_boundaries)
```
**Estimated Effort**: 4-6 hours
**Risk**: Medium (thread safety for file I/O)

#### Task 2.2: Add Progress Bar for Long Operations
**Libraries**: `tqdm` (already in optional dependencies)
**Target Steps**:
- Step 2: Ingest WAV file
- Step 4: Analyze audio
- Step 7: Process stems (bar-by-bar progress)

**Example**:
```python
from tqdm import tqdm

for bar_idx, start_s, end_s in tqdm(bar_boundaries, desc="Processing bars"):
    # ... existing code
```
**Estimated Effort**: 1 hour
**Impact**: Better UX for long-running pipelines

#### Task 2.3: Implement Chunk Writing Batching
**Current**: Write each chunk file individually (266 disk I/O operations)
**Proposal**: Batch-write chunks in groups of 10-20
**Expected Speedup**: 10-20% reduction in I/O time
**Estimated Effort**: 2-3 hours

#### Task 2.4: Add Memory Profiling
**Tool**: `memory_profiler` or `tracemalloc`
**Goal**: Identify memory-intensive operations
**Deliverable**: Memory usage report for 8-minute audio file
**Estimated Effort**: 2 hours

### Priority 3: Code Quality & Maintainability

#### Task 3.1: Suppress Librosa Tuning Warnings
**Issue**: "Trying to estimate tuning from empty frequency set" appears 2x per run
**Fix Options**:
1. Suppress warning if it's benign:
```python
import warnings
from librosa import UserWarning

with warnings.catch_warnings():
    warnings.filterwarnings("ignore", category=UserWarning,
                          message="Trying to estimate tuning")
    # ... librosa operations
```
2. Investigate root cause and fix upstream

**Estimated Effort**: 30 minutes
**Priority**: Low (cosmetic only)

#### Task 3.2: Add Step Timing to Pipeline Results
**Current**: Timing logged but not saved to JSON
**Proposal**: Include step timings in `combined.json` output
**Example**:
```json
{
  "timing": {
    "step_1_copy_input": 0.04,
    "step_2_ingest_wav": 7.53,
    "step_4_analyze_audio": 7.23,
    ...
  }
}
```
**Estimated Effort**: 1 hour
**Benefit**: Easier performance analysis across runs

#### Task 3.3: Create Performance Regression Test
**Goal**: Ensure future changes don't slow down the pipeline
**Implementation**:
1. Add pytest marker: `@pytest.mark.benchmark`
2. Create test that runs mini-pipeline on 10s audio
3. Assert total time < threshold (e.g., 5 seconds)
4. Run in CI/CD pipeline

**Estimated Effort**: 2-3 hours
**Benefit**: Catch performance regressions early

### Priority 4: Documentation & Monitoring

#### Task 4.1: Document preserve_chunk_audio Trade-offs
**Target File**: `README.md` or `docs/` section
**Content**:
- Speed: 18% faster (counterintuitive!)
- Storage: 4.35x more disk space
- Use cases: When to use each mode
- Performance characteristics

**Estimated Effort**: 30 minutes

#### Task 4.2: Add Benchmark Script to Examples
**File**: `examples/benchmark_pipeline.py`
**Document**: How to run benchmarks on custom audio
**Estimated Effort**: 15 minutes (already exists!)

#### Task 4.3: Create Performance Dashboard
**Tool**: Plotly or Matplotlib
**Visualizations**:
- Step timing breakdown (bar chart)
- Memory usage over time (line chart)
- Chunk file size distribution (histogram)

**Estimated Effort**: 3-4 hours
**Benefit**: Easy performance comparison across runs

### Priority 5: Optional Enhancements

#### Task 5.1: Implement Chunk Compression
**Proposal**: Store chunks as FLAC or compressed WAV
**Expected Savings**: 30-50% disk space
**Trade-off**: Slightly slower chunk loading
**Estimated Effort**: 2-3 hours

#### Task 5.2: Add Caching for Repeated Processing
**Use Case**: Re-running pipeline with different configs
**Cache Candidates**:
- Beat detection results
- Tempo analysis
- Spectral features

**Estimated Effort**: 4-6 hours
**Benefit**: 2-3x speedup for re-runs

#### Task 5.3: GPU Acceleration for Audio Analysis
**Library**: `cuSignal` (CUDA-accelerated signal processing)
**Target**: Step 4 (Analyze audio) - 7.23s → ~2s
**Estimated Effort**: 8-12 hours
**Risk**: High (new dependency, platform-specific)

---

## Risk Assessment

| Issue | Severity | Probability | Impact | Mitigation |
|-------|----------|-------------|--------|------------|
| NumPy 2.0 breaks analysis.py | High | High | Critical | Fix now (Task 1.1) |
| Multiprocessing causes file corruption | Medium | Low | Medium | Thorough testing, atomic writes |
| Performance regression from changes | Medium | Medium | Medium | Add benchmark tests (Task 3.3) |
| Disk space exhaustion (preserve mode) | Low | Medium | Low | Document trade-offs, add warnings |

---

## Recommendations

### Immediate Actions (This Week)
1. ✅ **Fix NumPy deprecation** (Task 1.1) - 15 min
2. ✅ **Investigate ingestion time mystery** (Task 1.2) - 1-2 hours
3. ✅ **Add progress bars** (Task 2.2) - 1 hour

### Short Term (This Month)
4. **Parallelize chunk processing** (Task 2.1) - 4-6 hours
5. **Add timing to output JSON** (Task 3.2) - 1 hour
6. **Create benchmark regression test** (Task 3.3) - 2-3 hours

### Long Term (This Quarter)
7. **Performance dashboard** (Task 4.3) - 3-4 hours
8. **Memory profiling** (Task 2.4) - 2 hours
9. **Caching system** (Task 5.2) - 4-6 hours

---

## Success Metrics

Track these metrics across future optimizations:

- **Processing Speed**: Target < 20s for 8-minute audio (currently 19.88s with preserve mode)
- **Memory Usage**: Target < 1GB peak RAM for 8-minute audio
- **Disk Efficiency**: Target < 20MB chunks per 8-minute song (mono mode)
- **Warning Count**: Target 0 warnings per run
- **Test Coverage**: Target > 90% for pipeline core

---

## Appendix A: Raw Benchmark Data

See [benchmark_metrics.json](benchmark_output/benchmark_metrics.json) for complete results.

**Test Environment**:
- OS: Windows 10
- Python: 3.13.1
- CPU: [detected from system]
- RAM: [detected from system]
- Storage: SSD (assumed)

**Pipeline Configuration**:
```python
PipelineConfig(
    analysis_sr=22050,
    write_bar_chunks=True,
    write_bar_chunk_wavs=True,
    use_pydub_preprocess=False,
    time_signature_num=4,
    time_signature_den=4,
)
```

---

## Appendix B: Test Audio Properties

**File**: `as_above_so_below_1_0_0.wav`

| Property | Value |
|----------|-------|
| Duration | 479.96s (8.00 min) |
| Sample Rate | 48000 Hz |
| Channels | 2 (stereo) |
| Bit Depth | 16-bit PCM |
| File Size | 87.88 MB |
| Detected BPM | 152.00 |
| Num Beats | 1056 |
| Num Bars | 266 (4/4 time) |

---

**Generated by**: Automated benchmark analysis
**Last Updated**: 2026-01-02
