"""
ONNX Export and Runtime Inference Engine for Liver Fibrosis Staging Models.

Supports PyTorch to ONNX graph export, INT8 Dynamic Quantization for edge deployment,
and comparative latency benchmarking between native PyTorch and ONNX Runtime.
"""
import time
from pathlib import Path
from typing import Dict, Tuple, Optional, Any, List
import numpy as np

import torch
import torch.nn as nn
import torch.nn.functional as F

try:
    import onnx
    import onnxruntime as ort
    from onnxruntime.quantization import quantize_dynamic, QuantType
except ImportError:
    ort = None

import sys
sys.path.insert(0, str(__file__).rsplit('src', 1)[0])
from config import IMAGE_SIZE, DEVICE, CLASS_NAMES


def export_to_onnx(model: nn.Module,
                   output_path: Path,
                   input_shape: Tuple[int, ...] = (1, 3, IMAGE_SIZE, IMAGE_SIZE),
                   opset_version: int = 17,
                   quantize_int8: bool = False,
                   verbose: bool = True) -> Path:
    """
    Export a PyTorch model (ensemble or branch) to ONNX format.

    Args:
        model: PyTorch model instance
        output_path: Path to save the exported .onnx file
        input_shape: Input tensor dimensions (B, C, H, W)
        opset_version: ONNX opset version (default 17 for modern transformer ops)
        quantize_int8: If True, also generate an INT8 dynamically quantized model
        verbose: Whether to print export progress and file size

    Returns:
        Path to the exported ONNX model (or quantized model if quantize_int8=True)
    """
    model.eval()
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    dummy_input = torch.randn(*input_shape, device="cpu")
    model_cpu = model.to("cpu")

    if verbose:
        print(f"Exporting model to ONNX: {output_path} (opset {opset_version})...")

    with torch.no_grad():
        torch.onnx.export(
            model_cpu,
            dummy_input,
            str(output_path),
            export_params=True,
            opset_version=opset_version,
            do_constant_folding=True,
            input_names=["input"],
            output_names=["logits"],
            dynamic_axes={
                "input": {0: "batch_size"},
                "logits": {0: "batch_size"}
            }
        )

    if verbose:
        file_size_mb = output_path.stat().st_size / (1024 * 1024)
        print(f"✅ Successfully exported ONNX model: {output_path.name} ({file_size_mb:.2f} MB)")

    if quantize_int8 and ort is not None:
        quant_path = output_path.with_name(output_path.stem + "_int8.onnx")
        if verbose:
            print(f"Applying INT8 Dynamic Quantization -> {quant_path.name}...")
        
        quantize_dynamic(
            model_input=str(output_path),
            model_output=str(quant_path),
            weight_type=QuantType.QInt8
        )
        if verbose:
            q_size_mb = quant_path.stat().st_size / (1024 * 1024)
            print(f"✅ Successfully created INT8 Quantized model: {quant_path.name} ({q_size_mb:.2f} MB)")
            print(f"💡 Compression ratio: {file_size_mb / q_size_mb:.2f}x reduction in file size")
        return quant_path

    return output_path


class ONNXInferenceEngine:
    """
    High-performance ONNX Runtime Inference Engine for liver slide classification.
    """

    def __init__(self, model_path: Path, use_gpu: bool = True):
        """
        Initialize ONNX Runtime session.

        Args:
            model_path: Path to .onnx model file
            use_gpu: Whether to attempt CUDA execution provider
        """
        if ort is None:
            raise ImportError("onnxruntime is not installed. Please install via pip install onnxruntime.")

        self.model_path = Path(model_path)
        if not self.model_path.exists():
            raise FileNotFoundError(f"ONNX model file not found at: {self.model_path}")

        providers = []
        if use_gpu and "CUDAExecutionProvider" in ort.get_available_providers():
            providers.append("CUDAExecutionProvider")
        providers.append("CPUExecutionProvider")

        self.session = ort.InferenceSession(str(self.model_path), providers=providers)
        self.input_name = self.session.get_inputs()[0].name
        self.output_name = self.session.get_outputs()[0].name
        self.provider_used = self.session.get_providers()[0]

    def predict(self, x: np.ndarray) -> Dict[str, Any]:
        """
        Run forward inference on input numpy array.

        Args:
            x: Input image array of shape (B, C, H, W) in float32 format

        Returns:
            Dictionary containing logits, probabilities, predicted class, and confidence
        """
        if x.ndim == 3:
            x = np.expand_dims(x, axis=0)
        x = x.astype(np.float32)

        start_t = time.perf_counter()
        logits_list = self.session.run([self.output_name], {self.input_name: x})[0]
        exec_t_ms = (time.perf_counter() - start_t) * 1000.0

        # Compute softmax probabilities
        exp_l = np.exp(logits_list - np.max(logits_list, axis=1, keepdims=True))
        probs = exp_l / np.sum(exp_l, axis=1, keepdims=True)

        preds = np.argmax(probs, axis=1)
        confidences = np.max(probs, axis=1)

        results = []
        for i in range(len(preds)):
            results.append({
                "predicted_class": CLASS_NAMES[preds[i]],
                "predicted_prob": float(confidences[i]),
                "probabilities": probs[i].tolist(),
                "logits": logits_list[i].tolist()
            })

        return {
            "batch_results": results,
            "latency_ms": exec_t_ms,
            "execution_provider": self.provider_used
        }


def benchmark_latency(pytorch_model: nn.Module,
                      onnx_engine: ONNXInferenceEngine,
                      input_shape: Tuple[int, ...] = (1, 3, IMAGE_SIZE, IMAGE_SIZE),
                      num_warmup: int = 10,
                      num_runs: int = 50,
                      device: str = "cpu") -> Dict[str, Any]:
    """
    Compare inference latency between native PyTorch and ONNX Runtime.

    Args:
        pytorch_model: Native PyTorch model
        onnx_engine: Initialized ONNXInferenceEngine
        input_shape: Test input tensor shape
        num_warmup: Number of warmup runs
        num_runs: Number of timed runs
        device: Device for PyTorch evaluation

    Returns:
        Benchmarking comparison report with latency in ms and speedup ratio.
    """
    pytorch_model.eval()
    pytorch_model.to(device)
    test_tensor = torch.randn(*input_shape, device=device)
    test_numpy = test_tensor.cpu().numpy()

    # Warmup PyTorch
    with torch.no_grad():
        for _ in range(num_warmup):
            _ = pytorch_model(test_tensor)

    # Time PyTorch
    pt_times = []
    with torch.no_grad():
        for _ in range(num_runs):
            t0 = time.perf_counter()
            _ = pytorch_model(test_tensor)
            if device == "cuda":
                torch.cuda.synchronize()
            pt_times.append((time.perf_counter() - t0) * 1000.0)

    # Warmup ONNX
    for _ in range(num_warmup):
        _ = onnx_engine.predict(test_numpy)

    # Time ONNX
    onnx_times = []
    for _ in range(num_runs):
        res = onnx_engine.predict(test_numpy)
        onnx_times.append(res["latency_ms"])

    pt_mean = float(np.mean(pt_times))
    pt_std = float(np.std(pt_times))
    onnx_mean = float(np.mean(onnx_times))
    onnx_std = float(np.std(onnx_times))
    speedup = pt_mean / max(onnx_mean, 0.001)

    return {
        "pytorch_latency_ms": {"mean": pt_mean, "std": pt_std},
        "onnx_latency_ms": {"mean": onnx_mean, "std": onnx_std},
        "speedup_ratio": speedup,
        "onnx_provider": onnx_engine.provider_used,
        "num_runs": num_runs,
        "summary": (f"🚀 ONNX Runtime ({onnx_engine.provider_used}) achieved {speedup:.2f}x speedup! "
                    f"(PyTorch: {pt_mean:.2f}±{pt_std:.2f} ms vs ONNX: {onnx_mean:.2f}±{onnx_std:.2f} ms)")
    }
