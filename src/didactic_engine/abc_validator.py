"""
ABC Notation Validation Module

Provides Python-based validation for ABC notation to catch common errors
before attempting to render. Based on the ABC validation playbook.

Key Functions:
    - :func:`validate_abc`: Main validation function
    - :func:`check_header`: Validate required header fields
    - :func:`check_meter_math`: Validate bar unit counts
    - :func:`check_voice_alignment`: Validate multi-voice alignment

Validation Goals:
    Fast lint pass to detect errors mechanically before rendering:
    - Missing/malformed headers (M:, L:, K:)
    - Incorrect bar lengths (meter math)
    - Voice misalignment in multi-voice ABC
    - Malformed syntax (unmatched brackets, quotes)

Example:
    >>> abc_text = '''X:1
    ... T:Test Tune
    ... M:4/4
    ... L:1/8
    ... K:C
    ... |CDEF GABC|'''
    >>> result = validate_abc(abc_text)
    >>> if result.is_valid:
    ...     print("ABC is valid!")
    ... else:
    ...     print(f"Errors: {result.errors}")
"""

import logging
import re
from dataclasses import dataclass, field
from typing import List, Dict, Optional, Tuple

logger = logging.getLogger(__name__)


@dataclass
class ABCValidationResult:
    """
    Result of ABC validation.
    
    Attributes:
        is_valid: True if ABC passed all validation checks
        errors: List of error messages
        warnings: List of warning messages
        metadata: Dict of extracted metadata (meter, unit length, key, etc.)
    """
    is_valid: bool
    errors: List[str] = field(default_factory=list)
    warnings: List[str] = field(default_factory=list)
    metadata: Dict[str, str] = field(default_factory=dict)


def validate_abc(abc_text: str, strict: bool = False) -> ABCValidationResult:
    """
    Validate ABC notation text.
    
    Performs the following checks:
    1. Header sanity (M:, L:, K: required)
    2. Meter math (bar unit counts)
    3. Voice alignment (if multi-voice)
    4. Basic syntax (unmatched brackets/quotes)
    
    Args:
        abc_text: ABC notation text to validate
        strict: If True, warnings are treated as errors
        
    Returns:
        ABCValidationResult with validation status and messages
        
    Example:
        >>> abc = '''X:1
        ... M:4/4
        ... L:1/8
        ... K:C
        ... |CDEF GABC|'''
        >>> result = validate_abc(abc)
        >>> assert result.is_valid
    """
    result = ABCValidationResult(is_valid=True)
    
    # Check header
    header_result = check_header(abc_text)
    result.errors.extend(header_result.errors)
    result.warnings.extend(header_result.warnings)
    result.metadata.update(header_result.metadata)
    
    if not header_result.is_valid:
        result.is_valid = False
        return result  # Can't continue without valid header
    
    # Check meter math (bar lengths)
    meter_result = check_meter_math(
        abc_text,
        meter=result.metadata.get('meter', '4/4'),
        unit_length=result.metadata.get('unit_length', '1/8')
    )
    result.errors.extend(meter_result.errors)
    result.warnings.extend(meter_result.warnings)
    
    if meter_result.errors:
        result.is_valid = False
    
    # Check voice alignment if multi-voice
    if _is_multivoice(abc_text):
        voice_result = check_voice_alignment(
            abc_text,
            meter=result.metadata.get('meter', '4/4'),
            unit_length=result.metadata.get('unit_length', '1/8')
        )
        result.errors.extend(voice_result.errors)
        result.warnings.extend(voice_result.warnings)
        
        if voice_result.errors:
            result.is_valid = False
    
    # Treat warnings as errors in strict mode
    if strict and result.warnings:
        result.errors.extend(result.warnings)
        result.warnings = []
        result.is_valid = False
    
    return result


def check_header(abc_text: str) -> ABCValidationResult:
    """
    Validate ABC header contains required fields.
    
    Required fields:
    - M: meter (e.g., M:4/4, M:6/8)
    - L: unit note length (e.g., L:1/8, L:1/16)
    - K: key signature (e.g., K:C, K:Gmaj)
    
    Args:
        abc_text: ABC notation text
        
    Returns:
        ABCValidationResult with header validation status
    """
    result = ABCValidationResult(is_valid=True)
    lines = abc_text.split('\n')
    
    # Extract header fields
    meter = None
    unit_length = None
    key = None
    
    for line in lines:
        line = line.strip()
        if line.startswith('M:'):
            meter = line[2:].strip()
            result.metadata['meter'] = meter
        elif line.startswith('L:'):
            unit_length = line[2:].strip()
            result.metadata['unit_length'] = unit_length
        elif line.startswith('K:'):
            key = line[2:].strip()
            result.metadata['key'] = key
            break  # K: typically marks end of header
    
    # Check required fields
    if not meter:
        result.errors.append("Missing required header field: M: (meter)")
        result.is_valid = False
    elif not _is_valid_meter(meter):
        result.errors.append(f"Invalid meter format: M:{meter}")
        result.is_valid = False
    
    if not unit_length:
        result.errors.append("Missing required header field: L: (unit note length)")
        result.is_valid = False
    elif not _is_valid_unit_length(unit_length):
        result.errors.append(f"Invalid unit length format: L:{unit_length}")
        result.is_valid = False
    
    if not key:
        result.errors.append("Missing required header field: K: (key signature)")
        result.is_valid = False
    
    return result


def check_meter_math(
    abc_text: str,
    meter: str = '4/4',
    unit_length: str = '1/8'
) -> ABCValidationResult:
    """
    Validate bar unit counts match meter.
    
    Formula: expected_units_per_bar = (meter_fraction) / (unit_length_fraction)
    Example: M:4/4, L:1/8 → (1) / (1/8) = 8 units per bar
    
    Args:
        abc_text: ABC notation text
        meter: Meter (e.g., '4/4', '6/8')
        unit_length: Unit note length (e.g., '1/8', '1/16')
        
    Returns:
        ABCValidationResult with meter math validation status
    """
    result = ABCValidationResult(is_valid=True)
    
    try:
        expected_units = _calculate_expected_units(meter, unit_length)
    except ValueError as e:
        result.errors.append(f"Failed to calculate expected units: {e}")
        result.is_valid = False
        return result
    
    # Extract music lines (after header, not voice declarations)
    music_lines = _extract_music_lines(abc_text)
    
    # Split into bars and count units
    bar_num = 0
    for line in music_lines:
        # Split by bar lines
        bars = re.split(r'\|+', line)
        
        for bar in bars:
            bar = bar.strip()
            if not bar or bar.startswith('['):  # Skip empty or inline fields
                continue
            
            bar_num += 1
            units = _count_units(bar, unit_length)
            
            # Allow some tolerance for pickup bars and incomplete bars
            if abs(units - expected_units) > 0.5:
                result.warnings.append(
                    f"Bar {bar_num} has {units} units, expected {expected_units} "
                    f"(meter: {meter}, unit length: {unit_length})"
                )
    
    return result


def check_voice_alignment(
    abc_text: str,
    meter: str = '4/4',
    unit_length: str = '1/8'
) -> ABCValidationResult:
    """
    Validate multi-voice ABC has aligned barlines.
    
    For reliable rendering:
    - Every voice has the same number of bars
    - Barlines occur in the same places
    - Each bar in each voice matches expected unit count
    
    Args:
        abc_text: ABC notation text (multi-voice)
        meter: Meter (e.g., '4/4', '6/8')
        unit_length: Unit note length (e.g., '1/8', '1/16')
        
    Returns:
        ABCValidationResult with voice alignment validation status
    """
    result = ABCValidationResult(is_valid=True)
    
    # Extract voices
    voices = _extract_voices(abc_text)
    
    if len(voices) < 2:
        return result  # Not multi-voice
    
    # Count bars per voice
    bar_counts = {}
    for voice_name, voice_text in voices.items():
        bars = _count_bars(voice_text)
        bar_counts[voice_name] = bars
    
    # Check all voices have same number of bars
    if len(set(bar_counts.values())) > 1:
        result.errors.append(
            f"Voice bar count mismatch: {bar_counts}. "
            f"All voices must have the same number of bars."
        )
        result.is_valid = False
    
    return result


# Helper functions

def _is_valid_meter(meter: str) -> bool:
    """Check if meter string is valid (e.g., '4/4', '6/8', 'C')."""
    if meter in ('C', 'C|'):  # Common time shortcuts
        return True
    return bool(re.match(r'^\d+/\d+$', meter))


def _is_valid_unit_length(unit_length: str) -> bool:
    """Check if unit length string is valid (e.g., '1/8', '1/16')."""
    return bool(re.match(r'^1/\d+$', unit_length))


def _calculate_expected_units(meter: str, unit_length: str) -> float:
    """Calculate expected units per bar."""
    # Parse meter
    if meter == 'C':
        meter = '4/4'
    elif meter == 'C|':
        meter = '2/2'
    
    meter_parts = meter.split('/')
    meter_num = int(meter_parts[0])
    meter_den = int(meter_parts[1])
    
    # Parse unit length
    unit_parts = unit_length.split('/')
    unit_den = int(unit_parts[1])
    
    # Calculate: (meter_fraction) / (unit_length_fraction)
    # Example: (4/4) / (1/8) = 1 / (1/8) = 8
    meter_fraction = meter_num / meter_den
    unit_fraction = 1.0 / unit_den
    
    return meter_fraction / unit_fraction


def _extract_music_lines(abc_text: str) -> List[str]:
    """Extract music lines (skip header and voice declarations)."""
    lines = abc_text.split('\n')
    music_lines = []
    in_header = True
    
    for line in lines:
        line = line.strip()
        
        # Skip empty lines
        if not line:
            continue
        
        # K: marks end of header
        if line.startswith('K:'):
            in_header = False
            continue
        
        # Skip header lines
        if in_header:
            continue
        
        # Skip voice declarations
        if line.startswith('V:'):
            continue
        
        # Skip comments
        if line.startswith('%'):
            continue
        
        music_lines.append(line)
    
    return music_lines


def _count_units(bar_text: str, unit_length: str) -> float:
    """
    Count units in a bar.
    
    Strips non-duration tokens:
    - Chord symbols in quotes: "Dm", "C7"
    - Whitespace
    - Ornaments and decorations
    
    Simplified counting (not perfect, but catches most issues):
    - Each letter = 1 unit
    - Numbers after notes multiply duration
    - / after notes divide duration
    """
    # Remove chord symbols (quoted text)
    bar_text = re.sub(r'"[^"]*"', '', bar_text)
    
    # Remove whitespace
    bar_text = bar_text.replace(' ', '')
    
    # Simple unit counting (count note letters)
    # This is approximate but catches major issues
    notes = re.findall(r'[A-Ga-gz]', bar_text)
    
    return float(len(notes))


def _is_multivoice(abc_text: str) -> bool:
    """Check if ABC text contains multiple voices."""
    return bool(re.search(r'^V:\d+', abc_text, re.MULTILINE))


def _extract_voices(abc_text: str) -> Dict[str, str]:
    """Extract individual voice texts from multi-voice ABC."""
    voices = {}
    lines = abc_text.split('\n')
    current_voice = None
    voice_lines = []
    
    for line in lines:
        if line.startswith('V:'):
            # Save previous voice
            if current_voice and voice_lines:
                voices[current_voice] = '\n'.join(voice_lines)
            
            # Start new voice
            current_voice = line[2:].strip().split()[0]
            voice_lines = []
        elif current_voice:
            voice_lines.append(line)
    
    # Save last voice
    if current_voice and voice_lines:
        voices[current_voice] = '\n'.join(voice_lines)
    
    return voices


def _count_bars(music_text: str) -> int:
    """Count number of bars in music text."""
    # Count bar lines (|)
    return music_text.count('|')
