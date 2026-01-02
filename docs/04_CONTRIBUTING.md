# Contributing Guide

Welcome! This guide explains how to contribute to didactic-engine.

---

## Getting Started

### Development Setup

```bash
# Clone the repository
git clone https://github.com/yakuzadave/didactic-engine.git
cd didactic-engine

# Create virtual environment (recommended)
python -m venv venv
source venv/bin/activate  # Linux/macOS
# or: venv\Scripts\activate  # Windows

# Install in editable mode with dev dependencies
pip install -e .

# Verify installation
pytest tests/ -v
```

### Running Tests

```bash
# Run all tests
pytest tests/ -v

# Run with coverage
pytest tests/ --cov=didactic_engine --cov-report=html

# Run specific test file
pytest tests/test_pipeline.py -v

# Run specific test class
pytest tests/test_pipeline.py::TestWAVIngester -v
```

---

## Code Style

### Python Style

- Follow PEP 8
- Maximum line length: 100 characters
- Use type hints for all public functions
- Use Google-style docstrings (see [STYLE_GUIDE.md](../STYLE_GUIDE.md))

### Code Formatting

```bash
# Format with black (if installed)
black src/ tests/

# Lint with ruff (if installed)
ruff check src/ tests/
```

### Docstrings

All public functions/classes must have docstrings. See [STYLE_GUIDE.md](../STYLE_GUIDE.md) for format.

---

## Making Changes

### Branch Naming

- `feature/description` - New features
- `fix/description` - Bug fixes
- `docs/description` - Documentation changes
- `refactor/description` - Code refactoring

### Commit Messages

Use conventional commits:

```
type(scope): brief description

Longer description if needed.

Fixes #123
```

Types:
- `feat`: New feature
- `fix`: Bug fix
- `docs`: Documentation
- `refactor`: Code refactoring
- `test`: Adding/updating tests
- `chore`: Maintenance tasks

### Pull Request Process

1. Create a branch from `main`
2. Make your changes
3. Add/update tests as needed
4. Update documentation if needed
5. Run tests locally
6. Push and create PR
7. Address review feedback

---

## Adding Features

### Adding a New Feature Extractor

1. **Add to features.py:**
```python
def extract_new_feature(self, audio: np.ndarray, sr: int) -> Dict[str, Any]:
    """
    Extract new feature from audio.
    
    Args:
        audio: Input audio array (1D mono).
        sr: Sample rate.
    
    Returns:
        Dictionary containing feature values.
    """
    # Implementation
    return {"new_feature": value}
```

2. **Update pipeline.py** to call the new extractor

3. **Add to Parquet output** if needed

4. **Write tests:**
```python
def test_extract_new_feature(self):
    extractor = FeatureExtractor()
    audio = np.random.randn(22050).astype(np.float32)
    result = extractor.extract_new_feature(audio, 22050)
    assert "new_feature" in result
```

5. **Update documentation**

### Adding a CLI Option

1. **Add argument in cli.py:**
```python
parser.add_argument(
    "--new-option",
    type=int,
    default=10,
    help="Description of the new option",
)
```

2. **Add to PipelineConfig** if it affects pipeline behavior:
```python
@dataclass(frozen=True)
class PipelineConfig:
    # ... existing fields ...
    new_option: int = 10
```

3. **Wire up in cli.py:**
```python
cfg = PipelineConfig(
    # ... existing args ...
    new_option=args.new_option,
)
```

4. **Update README.md** with usage example

### Adding a New Export Format

1. **Create export module:** `src/didactic_engine/export_newformat.py`

2. **Implement export function:**
```python
def export_newformat(data: Dict, output_path: str) -> bool:
    """
    Export data to new format.
    
    Args:
        data: Data to export.
        output_path: Output file path.
    
    Returns:
        True if successful, False otherwise.
    """
    try:
        # Implementation
        return True
    except Exception as e:
        print(f"Export failed: {e}")
        return False
```

3. **Add to __init__.py exports**

4. **Call from pipeline.py**

5. **Write tests**

---

## Testing Guidelines

### What to Test

- All public functions and methods
- Edge cases (empty input, invalid input)
- Error handling (exceptions raised correctly)
- Integration between components

### Test Structure

```python
class TestComponentName:
    """Tests for ComponentName."""
    
    def test_normal_case(self):
        """Test normal operation."""
        # Arrange
        component = Component()
        input_data = create_test_data()
        
        # Act
        result = component.process(input_data)
        
        # Assert
        assert result is not None
        assert "expected_key" in result
    
    def test_edge_case_empty_input(self):
        """Test with empty input."""
        component = Component()
        result = component.process([])
        assert result == expected_empty_result
    
    def test_error_case_invalid_input(self):
        """Test error handling for invalid input."""
        component = Component()
        with pytest.raises(ValueError):
            component.process(None)
```

### Mocking External Dependencies

```python
from unittest.mock import patch, MagicMock

def test_demucs_not_available():
    """Test graceful handling when Demucs unavailable."""
    with patch("shutil.which", return_value=None):
        separator = StemSeparator()
        assert not separator._check_demucs_available()
```

---

## Documentation Guidelines

### Updating Docs

When adding features, update:

1. Relevant docstrings
2. README.md if user-facing
3. docs/*.md if architecture changes
4. DOC_INVENTORY.md to track status

### Writing Good Docstrings

```python
def process_audio(
    audio: np.ndarray,
    sample_rate: int,
    normalize: bool = True,
) -> Dict[str, Any]:
    """
    Process audio through the analysis pipeline.
    
    Applies normalization (optional) and extracts features using
    librosa. The audio is assumed to be mono; stereo inputs are
    automatically converted.
    
    Args:
        audio: Input audio as 1D numpy array. Must be float32.
        sample_rate: Sample rate in Hz. Common values: 22050, 44100.
        normalize: If True, normalize audio to [-1, 1] range.
            Defaults to True.
    
    Returns:
        Dictionary containing:
            - "tempo": Estimated tempo in BPM
            - "beats": List of beat times in seconds
            - "features": Nested dict of audio features
    
    Raises:
        ValueError: If audio is empty or contains NaN/Inf.
        TypeError: If audio is not a numpy array.
    
    Example:
        >>> audio = np.random.randn(22050).astype(np.float32)
        >>> result = process_audio(audio, 22050)
        >>> print(f"Tempo: {result['tempo']:.1f} BPM")
        Tempo: 120.0 BPM
    
    See Also:
        AudioAnalyzer: For more advanced analysis options.
    """
```

---

## Review Checklist

Before submitting a PR, verify:

- [ ] Tests pass locally
- [ ] New code has tests
- [ ] Public APIs have docstrings
- [ ] Type hints on public functions
- [ ] README updated if user-facing
- [ ] No unnecessary print statements
- [ ] No hardcoded paths or credentials

---

## Getting Help

- **Questions:** Open a GitHub issue with "question" label
- **Bugs:** Open a GitHub issue with reproduction steps
- **Features:** Open a GitHub issue to discuss first

---

## See Also

- [Architecture](01_ARCHITECTURE.md) - System design
- [STYLE_GUIDE.md](../STYLE_GUIDE.md) - Documentation standards
- [TASKS.md](../TASKS.md) - Current documentation tasks
