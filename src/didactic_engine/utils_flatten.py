"""
Utility functions for flattening nested dictionaries.

This module provides functions to flatten nested dict structures for use
in DataFrames and Parquet files. Nested dicts are common in feature
extraction results but incompatible with tabular formats.

Key Functions:
    - :func:`flatten_dict`: Flatten with dot separator (general use)
    - :func:`flatten_dict_for_parquet`: Flatten with underscore (Parquet)
    - :func:`unflatten_dict`: Reverse flattening

Integration:
    Dictionary flattening is used when preparing feature dictionaries
    for DataFrame conversion and Parquet export. The pipeline uses
    these functions in the feature extraction and dataset writing steps.

Example:
    >>> nested = {"audio": {"tempo": 120, "beats": [1, 2, 3]}}
    >>> flat = flatten_dict(nested)
    >>> print(flat)
    {'audio.tempo': 120, 'audio.beats': [1, 2, 3]}

See Also:
    - :mod:`didactic_engine.features` for feature extraction
    - :mod:`didactic_engine.pipeline` for dataset generation
"""

from typing import Any, Dict


def flatten_dict(
    d: Dict[str, Any],
    parent_key: str = "",
    sep: str = ".",
) -> Dict[str, Any]:
    """Flatten a nested dictionary by concatenating keys.

    Recursively flattens nested dictionaries, joining keys with a
    separator. Lists are preserved as-is (not exploded).

    Args:
        d: Dictionary to flatten. Can have arbitrary nesting depth.
        parent_key: Prefix for keys (used in recursion). Usually leave
            as empty string for top-level calls.
        sep: Separator between nested keys. Default "." produces
            keys like "level1.level2.value".

    Returns:
        Flattened dictionary with dot-separated keys. All values are
        leaf values from the original nested structure.

    Example:
        >>> nested = {"a": {"b": 1, "c": 2}, "d": 3}
        >>> flatten_dict(nested)
        {'a.b': 1, 'a.c': 2, 'd': 3}
        
        >>> deep = {"level1": {"level2": {"value": 10}}}
        >>> flatten_dict(deep)
        {'level1.level2.value': 10}
        
        >>> with_list = {"features": {"mfcc": [1, 2, 3]}}
        >>> flatten_dict(with_list)
        {'features.mfcc': [1, 2, 3]}  # Lists preserved

    Note:
        Lists are NOT flattened—they're kept as list values. This is
        intentional to avoid exploding arrays. For Parquet compatibility
        where lists need conversion, use :func:`flatten_dict_for_parquet`.

    See Also:
        - :func:`unflatten_dict` to reverse the operation
        - :func:`flatten_dict_for_parquet` for Parquet-compatible output
    """
    from typing import List, Tuple
    items: List[Tuple[str, Any]] = []

    for k, v in d.items():
        new_key = f"{parent_key}{sep}{k}" if parent_key else k

        if isinstance(v, dict):
            # Recursively flatten nested dicts
            items.extend(flatten_dict(v, new_key, sep=sep).items())
        elif isinstance(v, list):
            # Keep lists as-is (don't explode them)
            # For Parquet compatibility, convert to string if needed
            items.append((new_key, v))
        else:
            items.append((new_key, v))

    return dict(items)


def flatten_dict_for_parquet(
    d: Dict[str, Any],
    parent_key: str = "",
    sep: str = "_",
) -> Dict[str, Any]:
    """Flatten a nested dictionary for Parquet storage.

    Similar to :func:`flatten_dict` but with Parquet-specific handling:
    - Uses underscore separator (Parquet column naming convention)
    - Converts lists to comma-separated strings for compatibility

    Args:
        d: Dictionary to flatten.
        parent_key: Prefix for keys (used in recursion).
        sep: Separator between nested keys. Default "_" for Parquet.

    Returns:
        Flattened dictionary suitable for Parquet storage. Lists of
        numbers are converted to comma-separated strings.

    Example:
        >>> nested = {"audio": {"tempo": 120, "beats": [1.0, 2.0, 3.0]}}
        >>> flatten_dict_for_parquet(nested)
        {'audio_tempo': 120, 'audio_beats': '1.0,2.0,3.0'}

    Note:
        Use this function when creating DataFrames for Parquet export.
        Use :func:`flatten_dict` for general-purpose flattening where
        lists can remain as lists.

    See Also:
        - :func:`flatten_dict` for general use (preserves lists)
    """
    from typing import List, Tuple
    items: List[Tuple[str, Any]] = []

    for k, v in d.items():
        new_key = f"{parent_key}{sep}{k}" if parent_key else k

        if isinstance(v, dict):
            items.extend(flatten_dict_for_parquet(v, new_key, sep=sep).items())
        elif isinstance(v, list):
            # Convert lists to comma-separated strings for Parquet
            if all(isinstance(x, (int, float)) for x in v):
                items.append((new_key, ",".join(str(x) for x in v)))
            else:
                items.append((new_key, str(v)))
        else:
            items.append((new_key, v))

    return dict(items)


def unflatten_dict(
    d: Dict[str, Any],
    sep: str = ".",
) -> Dict[str, Any]:
    """Unflatten a dictionary with separated keys back to nested structure.

    Reverses the operation of :func:`flatten_dict`, reconstructing the
    original nested dictionary structure.

    Args:
        d: Flattened dictionary with separated keys.
        sep: Separator used in keys. Must match the separator used
            during flattening.

    Returns:
        Nested dictionary structure.

    Example:
        >>> flat = {"a.b": 1, "a.c": 2, "d": 3}
        >>> unflatten_dict(flat)
        {'a': {'b': 1, 'c': 2}, 'd': 3}
        
        >>> flat = {"level1.level2.value": 10}
        >>> unflatten_dict(flat)
        {'level1': {'level2': {'value': 10}}}

    Note:
        This function assumes keys were created by :func:`flatten_dict`.
        Behavior is undefined for manually constructed keys that don't
        follow the expected pattern.

    See Also:
        - :func:`flatten_dict` for the forward operation
    """
    result: Dict[str, Any] = {}

    for key, value in d.items():
        parts = key.split(sep)
        current = result

        for part in parts[:-1]:
            if part not in current:
                current[part] = {}
            current = current[part]

        current[parts[-1]] = value

    return result
