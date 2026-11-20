"""ONNX Runtime sessions for the locally run models."""


def onnx_session(model_path, options=None):
    """A session over `model_path` on CUDA when ONNX Runtime has it, else the CPU."""
    import onnxruntime

    if not model_path.is_file():
        raise FileNotFoundError(f"No ONNX model at {model_path}")
    # onnxruntime-gpu ships without cuDNN; load the one torch's nvidia wheels carry.
    onnxruntime.preload_dlls()
    providers = [
        provider
        for provider in ("CUDAExecutionProvider", "CPUExecutionProvider")
        if provider in onnxruntime.get_available_providers()
    ]
    session = onnxruntime.InferenceSession(str(model_path), options, providers=providers)
    if session.get_providers()[0] != "CUDAExecutionProvider":
        print(f"CUDA unavailable to ONNX Runtime; running {model_path.name} on the CPU")
    return session
