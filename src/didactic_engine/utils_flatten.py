"""
Utility functions for flattening nested dictionaries.

Provides utilities to flatten nested dict structures for use in
DataFrames and Parquet files.
"""

from typing import Any, Dict


def flatten_dict(
    d: Dict[str, Any],
    parent_key: str = "",
    sep: str = ".",
) -> Dict[str, Any]:
    """
    Flatten a nested dictionary by concatenating keys.

    Args:
        d: Dictionary to flatten.
        parent_key: Prefix for keys (used in recursion).
        sep: Separator between nested keys.

    Returns:
        Flattened dictionary with dot-separated keys.

    Example:
        >>> flatten_dict({"a": {"b": 1, "c": 2}, "d": 3})
        {"a.b": 1, "a.c": 2, "d": 3}

        >>> flatten_dict({"level1": {"level2": {"value": 10}}})
        {"level1.level2.value": 10}
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
    """
    Flatten a nested dictionary for Parquet storage.

    Similar to flatten_dict but:
    - Uses underscore separator (Parquet column naming convention)
    - Converts lists to strings for compatibility

    Args:
        d: Dictionary to flatten.
        parent_key: Prefix for keys.
        sep: Separator between nested keys (default: underscore).

    Returns:
        Flattened dictionary suitable for Parquet storage.
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
    """
    Unflatten a dictionary with dot-separated keys.

    Args:
        d: Flattened dictionary.
        sep: Separator used in keys.

    Returns:
        Nested dictionary.

    Example:
        >>> unflatten_dict({"a.b": 1, "a.c": 2, "d": 3})
        {"a": {"b": 1, "c": 2}, "d": 3}
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
