# Environment Setup (Windows & WSL)

This project targets Python 3.11+ and runs well on Windows or WSL. The notes below cover CPU-only installs, GPU acceleration, and optional tools.

## Common prerequisites
- Python 3.11 (3.12 works, but Basic Pitch is 3.11-only)
- FFmpeg in PATH (required by `pydub`)
  - Windows: `choco install ffmpeg`
  - WSL/Ubuntu: `sudo apt-get install ffmpeg`
- Optional: `libsndfile1` for soundfile on Linux (`sudo apt-get install libsndfile1`)

Create a virtual environment in the repo root:
```bash
python3.11 -m venv .venv
source .venv/bin/activate            # PowerShell: .venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
```

## CPU-only install
- Minimal runtime: `pip install -e .`
- With dev tooling/tests: `pip install -e ".[dev]"`
- With all optional CPU features: `pip install -e ".[all,dev]"`

## GPU-enabled install (NVIDIA CUDA)
We keep CPU and GPU extras separate so `onnxruntime-gpu` can be used without pulling the CPU wheel.

1) Confirm drivers:
   - Windows: install the latest NVIDIA driver (Studio/Game Ready).
   - WSL2: install the [NVIDIA CUDA on WSL](https://learn.microsoft.com/windows/ai/directml/gpu-cuda-in-wsl) driver and verify `nvidia-smi` inside WSL.

2) Install PyTorch with CUDA 12.1 wheels (no full CUDA toolkit required):
```bash
pip install --index-url https://download.pytorch.org/whl/cu121 torch torchvision torchaudio
```

3) Install project deps with GPU extras:
```bash
pip install -e ".[ml-gpu,dev]"       # GPU ONNX Runtime + Demucs + test tooling
# If you want every optional feature: pip install -e ".[all-gpu,dev]"
```

4) Verify GPU visibility:
```bash
python - <<'PY'
import torch, onnxruntime as ort
print("PyTorch CUDA available:", torch.cuda.is_available())
print("ONNX providers:", ort.get_available_providers())
PY
```
Expect to see `CUDAExecutionProvider` in the ONNX providers list.

## Windows vs WSL tips
- WSL gives easier access to CUDA-enabled PyTorch wheels and `apt-get` for FFmpeg; use `python3.11-venv` and `python3.11-dev` packages if needed.
- On Windows, keep paths short to avoid MAX_PATH issues when Demucs writes stems.
- Basic Pitch depends on TensorFlow and only works on Python 3.11; on Python 3.12 use the ONNX path instead.

## Optional extras at a glance
- `.[ml]` / `.[ml-gpu]`: Demucs, Basic Pitch (3.11), ONNX Runtime (CPU/GPU), torchcodec
- `.[essentia]`: Essentia-based features (AGPL)
- `.[viz]`: Plotly + Kaleido visualizations
- `.[batch]`: tqdm/polars helpers for batch workflows
- `.[all]` / `.[all-gpu]`: Bundles the above CPU/GPU stacks respectively
