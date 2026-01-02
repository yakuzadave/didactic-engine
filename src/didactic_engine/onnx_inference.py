"""
ONNX Runtime inference module for ML models.

This module provides ONNX-based inference capabilities as an alternative
to TensorFlow-based inference. It's particularly useful on Python 3.12+
where TensorFlow < 2.15.1 is not available.

Key Features:
    - ONNX model loading and inference
    - Availability checking for graceful degradation
    - Support for audio-related inference tasks

Prerequisites:
    ONNX Runtime must be installed::
    
        pip install onnxruntime

Usage:
    The module provides utilities for checking ONNX runtime availability
    and running inference on ONNX models. This is useful for scenarios
    where TensorFlow-based tools (like basic-pitch) aren't available.

Example:
    >>> from didactic_engine.onnx_inference import is_onnxruntime_available
    >>> if is_onnxruntime_available():
    ...     from didactic_engine.onnx_inference import ONNXInferenceSession
    ...     session = ONNXInferenceSession("model.onnx")
    ...     output = session.run(input_data)

See Also:
    - :mod:`didactic_engine.transcription` for MIDI transcription
"""

from pathlib import Path
from typing import Any, Dict, List, Optional, Union
import numpy as np

# Check for onnxruntime availability
try:
    import onnxruntime as ort
    ONNXRUNTIME_AVAILABLE = True
except ImportError:
    ort = None  # type: ignore
    ONNXRUNTIME_AVAILABLE = False


def is_onnxruntime_available() -> bool:
    """Check if ONNX Runtime is available.
    
    Returns:
        True if onnxruntime is installed and can be imported,
        False otherwise.
    
    Example:
        >>> if is_onnxruntime_available():
        ...     print("ONNX Runtime is available")
        ... else:
        ...     print("Please install: pip install onnxruntime")
    """
    return ONNXRUNTIME_AVAILABLE


def get_onnxruntime_version() -> Optional[str]:
    """Get the installed ONNX Runtime version.
    
    Returns:
        Version string if onnxruntime is installed, None otherwise.
    
    Example:
        >>> version = get_onnxruntime_version()
        >>> if version:
        ...     print(f"ONNX Runtime version: {version}")
    """
    if ONNXRUNTIME_AVAILABLE and ort is not None:
        return ort.__version__
    return None


def get_available_providers() -> List[str]:
    """Get available ONNX Runtime execution providers.
    
    Execution providers determine how inference is performed (CPU, GPU, etc.).
    
    Returns:
        List of available provider names. Empty list if onnxruntime
        is not installed.
    
    Example:
        >>> providers = get_available_providers()
        >>> if 'CUDAExecutionProvider' in providers:
        ...     print("GPU acceleration available")
    
    Common Providers:
        - ``CPUExecutionProvider``: Always available
        - ``CUDAExecutionProvider``: NVIDIA GPU support
        - ``CoreMLExecutionProvider``: Apple Silicon support
        - ``TensorrtExecutionProvider``: TensorRT optimization
    """
    if ONNXRUNTIME_AVAILABLE and ort is not None:
        return ort.get_available_providers()
    return []


class ONNXInferenceSession:
    """ONNX Runtime inference session wrapper.
    
    Provides a simplified interface for loading ONNX models and running
    inference. Handles provider selection and input/output management.
    
    Attributes:
        model_path: Path to the loaded ONNX model.
        providers: Execution providers being used.
        session: Underlying ONNX Runtime InferenceSession.
    
    Example:
        >>> session = ONNXInferenceSession("model.onnx")
        >>> input_data = np.random.randn(1, 3, 224, 224).astype(np.float32)
        >>> outputs = session.run({"input": input_data})
    """
    
    def __init__(
        self,
        model_path: Union[str, Path],
        providers: Optional[List[str]] = None,
    ):
        """Initialize an ONNX inference session.
        
        Args:
            model_path: Path to the ONNX model file (.onnx).
            providers: List of execution providers to use. If None,
                uses all available providers in order of preference.
        
        Raises:
            RuntimeError: If onnxruntime is not installed.
            FileNotFoundError: If the model file doesn't exist.
            ValueError: If the model file is invalid.
        
        Example:
            >>> # Use default providers (CPU)
            >>> session = ONNXInferenceSession("model.onnx")
            
            >>> # Prefer GPU if available
            >>> session = ONNXInferenceSession(
            ...     "model.onnx",
            ...     providers=['CUDAExecutionProvider', 'CPUExecutionProvider']
            ... )
        """
        if not ONNXRUNTIME_AVAILABLE or ort is None:
            raise RuntimeError(
                "onnxruntime is not installed. Please install it:\n"
                "  pip install onnxruntime\n"
                "Or for GPU support:\n"
                "  pip install onnxruntime-gpu"
            )
        
        self.model_path = Path(model_path)
        
        if not self.model_path.exists():
            raise FileNotFoundError(f"Model file not found: {self.model_path}")
        
        # Use provided providers or default to all available
        if providers is None:
            providers = ort.get_available_providers()
        
        self.providers = providers
        
        try:
            self.session = ort.InferenceSession(
                str(self.model_path),
                providers=self.providers,
            )
        except Exception as e:
            raise ValueError(f"Failed to load ONNX model: {e}") from e
    
    def get_input_names(self) -> List[str]:
        """Get names of model input tensors.
        
        Returns:
            List of input tensor names.
        
        Example:
            >>> session = ONNXInferenceSession("model.onnx")
            >>> print(session.get_input_names())
            ['input_audio']
        """
        return [inp.name for inp in self.session.get_inputs()]
    
    def get_output_names(self) -> List[str]:
        """Get names of model output tensors.
        
        Returns:
            List of output tensor names.
        
        Example:
            >>> session = ONNXInferenceSession("model.onnx")
            >>> print(session.get_output_names())
            ['notes', 'onsets', 'contours']
        """
        return [out.name for out in self.session.get_outputs()]
    
    def get_input_shapes(self) -> Dict[str, List[Optional[int]]]:
        """Get shapes of model input tensors.
        
        Returns:
            Dictionary mapping input names to their shapes.
            Dynamic dimensions are represented as None.
        
        Example:
            >>> session = ONNXInferenceSession("model.onnx")
            >>> print(session.get_input_shapes())
            {'input_audio': [1, None, 1]}  # batch, time, channels
        """
        shapes = {}
        for inp in self.session.get_inputs():
            # Handle dynamic dimensions (may be strings like 'batch_size')
            shape = []
            for dim in inp.shape:
                if isinstance(dim, int):
                    shape.append(dim)
                else:
                    shape.append(None)  # Dynamic dimension
            shapes[inp.name] = shape
        return shapes
    
    def run(
        self,
        inputs: Dict[str, np.ndarray],
        output_names: Optional[List[str]] = None,
    ) -> Dict[str, np.ndarray]:
        """Run inference on the model.
        
        Args:
            inputs: Dictionary mapping input names to numpy arrays.
            output_names: Optional list of outputs to return. If None,
                returns all outputs.
        
        Returns:
            Dictionary mapping output names to numpy arrays.
        
        Raises:
            ValueError: If input shapes don't match model expectations.
        
        Example:
            >>> session = ONNXInferenceSession("model.onnx")
            >>> inputs = {"audio": audio_array.astype(np.float32)}
            >>> outputs = session.run(inputs)
            >>> notes = outputs["notes"]
        """
        if output_names is None:
            output_names = self.get_output_names()
        
        # Run inference
        results = self.session.run(output_names, inputs)
        
        # Package results as dictionary
        return dict(zip(output_names, results))
    
    def __repr__(self) -> str:
        """String representation of the session."""
        return (
            f"ONNXInferenceSession("
            f"model='{self.model_path.name}', "
            f"inputs={self.get_input_names()}, "
            f"outputs={self.get_output_names()})"
        )


def create_inference_session(
    model_path: Union[str, Path],
    prefer_gpu: bool = False,
) -> ONNXInferenceSession:
    """Create an ONNX inference session with sensible defaults.
    
    Convenience function that configures execution providers based
    on availability and preference.
    
    Args:
        model_path: Path to the ONNX model file.
        prefer_gpu: If True and GPU providers are available, use them.
            Falls back to CPU if GPU is not available.
    
    Returns:
        Configured ONNXInferenceSession.
    
    Raises:
        RuntimeError: If onnxruntime is not installed.
    
    Example:
        >>> session = create_inference_session("model.onnx", prefer_gpu=True)
        >>> print(f"Using providers: {session.providers}")
    """
    if not ONNXRUNTIME_AVAILABLE:
        raise RuntimeError(
            "onnxruntime is not installed. Please install it:\n"
            "  pip install onnxruntime"
        )
    
    available = get_available_providers()
    
    if prefer_gpu:
        # Prefer GPU providers if available
        gpu_providers = [
            'CUDAExecutionProvider',
            'TensorrtExecutionProvider',
            'CoreMLExecutionProvider',
        ]
        providers = [p for p in gpu_providers if p in available]
        providers.append('CPUExecutionProvider')  # Always have CPU as fallback
    else:
        providers = ['CPUExecutionProvider']
    
    return ONNXInferenceSession(model_path, providers=providers)
