# STYLE_GUIDE.md - Documentation Standards

Last Updated: 2026-01-02

---

## Overview

This guide defines documentation standards for the didactic-engine project.
All docstrings and comments should follow these conventions.

---

## Docstring Format: Google Style

We use **Google-style docstrings** for consistency with the scientific Python ecosystem.

### Module Docstring Template

```python
"""
Brief one-line description.

Extended description explaining the module's purpose, how it fits
into the overall architecture, and any important design decisions.

Example:
    Brief usage example if appropriate::

        from didactic_engine.module import SomeClass
        obj = SomeClass()
        result = obj.process(data)

Note:
    Any important caveats or requirements.

See Also:
    related_module: Description of relationship
"""
```

### Class Docstring Template

```python
class SomeClass:
    """
    Brief one-line description of the class.

    Extended description of the class, its purpose, and how to use it.

    Attributes:
        attr1: Description of attr1.
        attr2: Description of attr2.

    Example:
        >>> obj = SomeClass(param1=value)
        >>> obj.method()
        expected_output

    Note:
        Any important caveats about the class.

    See Also:
        RelatedClass: Description of relationship.
    """
```

### Function/Method Docstring Template

```python
def some_function(param1: str, param2: int = 10) -> dict:
    """
    Brief one-line description of what the function does.

    Extended description if needed, explaining the algorithm,
    important assumptions, or non-obvious behavior.

    Args:
        param1: Description of param1. Include valid values
            or constraints if relevant.
        param2: Description of param2. Defaults to 10.

    Returns:
        Description of what is returned. For complex return types,
        describe the structure:
            - key1: Description
            - key2: Description

    Raises:
        ValueError: When param1 is empty.
        FileNotFoundError: When the referenced file doesn't exist.

    Side Effects:
        - Writes to disk at path X
        - Modifies global state Y
        - Makes network call to Z

    Note:
        Time complexity: O(n log n)
        Not thread-safe.

    Example:
        >>> result = some_function("input", param2=20)
        >>> print(result["key"])
        'expected_value'

    See Also:
        related_function: Does something similar.
    """
```

---

## Comment Guidelines

### When to Add Comments

**DO** add comments for:
- **Why** something is done (not what)
- Non-obvious algorithms or data structures
- Invariants and constraints
- Workarounds and their reasons
- Performance considerations
- Security implications
- TODO items with context

**DON'T** add comments for:
- Obvious code (e.g., `i += 1  # increment i`)
- Restating the function name
- Dead code (delete it instead)

### Comment Format

```python
# Single-line comment for brief explanations

# Multi-line comment for longer explanations.
# Each line starts with #.
# Keep lines under 79 characters.

# TODO(username): Brief description of what needs to be done
# and why it's deferred.

# HACK: Explanation of why this workaround exists
# and what the proper fix would be.

# NOTE: Important information that's easy to miss.

# WARNING: Critical information about potential issues.
```

### Inline Comments

```python
x = calculate_value()  # Brief explanation if not obvious

# Longer explanation goes on its own line(s)
# before the code it describes
y = complex_operation(x)
```

---

## Type Hints

### Required For:
- All public function signatures
- Class attributes
- Return types

### Format:
```python
from typing import Dict, List, Optional, Union, Any
from pathlib import Path
import numpy as np
import pandas as pd

def process_audio(
    audio: np.ndarray,
    sample_rate: int,
    options: Optional[Dict[str, Any]] = None,
) -> pd.DataFrame:
    """Process audio and return results."""
    ...
```

### Type Hint Style:
- Use `Optional[X]` instead of `X | None` for 3.9 compatibility
- Use `List[X]` from typing for clarity
- For numpy arrays, use `np.ndarray` (not `numpy.ndarray`)
- For paths, accept `Union[str, Path]` and convert internally

---

## Cross-References

### In Docstrings

```python
"""
See Also:
    AudioAnalyzer: For audio analysis features.
    :func:`align_notes_to_beats`: For note alignment.
    :class:`PipelineConfig`: For configuration options.
"""
```

### In Comments

```python
# See AudioAnalyzer.analyze() for feature extraction
# Ref: https://librosa.org/doc/latest/beat.html
```

---

## Examples in Docstrings

### Basic Example

```python
"""
Example:
    >>> result = function(input)
    >>> print(result)
    expected_output
"""
```

### Complex Example

```python
"""
Example:
    Process a WAV file through the pipeline::

        from didactic_engine import PipelineConfig, AudioPipeline
        from pathlib import Path

        cfg = PipelineConfig(
            song_id="my_song",
            input_wav=Path("song.wav"),
            out_dir=Path("output"),
        )
        pipeline = AudioPipeline(cfg)
        results = pipeline.run()

        print(f"Found {results['num_bars']} bars")
"""
```

---

## Documentation Files (Markdown)

### Headings
- Use ATX-style (`#`, `##`, `###`)
- One blank line before and after headings
- Capitalize first word only (sentence case)

### Code Blocks
- Use fenced code blocks with language identifier
- ```python for Python
- ```bash for shell commands
- ```mermaid for diagrams

### Lists
- Use `-` for unordered lists
- Use `1.` for ordered lists
- Indent nested items by 2 spaces

### Links
- Use reference-style links for repeated URLs
- Use relative paths for internal docs

---

## Quality Checklist

Before marking a module as DONE:

- [ ] Module has docstring explaining purpose
- [ ] All public classes have docstrings
- [ ] All public methods have docstrings
- [ ] Args, Returns, Raises documented
- [ ] Side effects documented
- [ ] Examples provided where helpful
- [ ] Type hints on all public signatures
- [ ] Cross-references to related code
- [ ] Comments explain "why" not "what"
