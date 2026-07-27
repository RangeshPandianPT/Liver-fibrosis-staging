"""
Command-Line Interface (CLI) for Automated Liver Fibrosis Staging Platform.
Entry point: `liver-stage`
"""
import sys
import json
import argparse
from pathlib import Path
from PIL import Image
import numpy as np

import torch
import torch.nn.functional as F

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from config import CLASS_NAMES, DEVICE, IMAGE_SIZE
from src.models import SoftVotingEnsemble
from src.preprocessing import preprocess_image
from src.xai import UncertaintyEstimator
from src.inference.onnx_engine import export_to_onnx, ONNXInferenceEngine, benchmark_latency


def cli_predict(args):
    """Command: Predict fibrosis stage for an image or folder of images."""
    image_path = Path(args.image)
    if not image_path.exists():
        print(f"❌ Error: Image path '{image_path}' does not exist.")
        sys.exit(1)

    print(f"🔬 Loading ensemble model on {DEVICE}...")
    model = SoftVotingEnsemble(pretrained=False)
    model.eval()
    model.to(DEVICE)

    images_to_process = [image_path] if image_path.is_file() else list(image_path.glob("*.*"))
    results = []

    print(f"\nEvaluating {len(images_to_process)} slide(s)...\n")
    print(f"{'Filename':<25} | {'Stage':<6} | {'Conf':<7} | {'Entropy':<7} | {'Status'}")
    print("-" * 80)

    for img_p in images_to_process:
        try:
            pil_img = Image.open(img_p).convert("RGB")
            tensor = preprocess_image(np.array(pil_img), is_training=False)
            if tensor.ndim == 3:
                tensor = tensor.unsqueeze(0)
            
            with torch.no_grad():
                logits = model(tensor.to(DEVICE))
                probs = F.softmax(logits, dim=1)[0].cpu().numpy()

            analysis = UncertaintyEstimator.analyze_prediction(probs)
            
            status_symbol = "⚠️ AMBIGUOUS" if analysis["alert_triggered"] else "✅ CONFIDENT"
            print(f"{img_p.name:<25} | {analysis['predicted_class']:<6} | {analysis['predicted_prob']*100:6.1f}% | {analysis['entropy']:7.2f} | {status_symbol}")

            results.append({
                "file": str(img_p),
                "predicted_stage": analysis["predicted_class"],
                "confidence": float(analysis["predicted_prob"]),
                "entropy": float(analysis["entropy"]),
                "alert": analysis["alert_triggered"],
                "recommendation": analysis["recommendation"]
            })
        except Exception as e:
            print(f"{img_p.name:<25} | ERROR: {str(e)}")

    if args.output:
        out_p = Path(args.output)
        out_p.parent.mkdir(parents=True, exist_ok=True)
        with open(out_p, "w", encoding="utf-8") as f:
            json.dump(results, f, indent=2)
        print(f"\n💾 Results saved to {out_p}")


def cli_export_onnx(args):
    """Command: Export PyTorch model to ONNX format."""
    print(f"📦 Initializing model '{args.branch}' for ONNX export...")
    ensemble = SoftVotingEnsemble(pretrained=False)
    
    if args.branch.lower() == "ensemble":
        model = ensemble
    else:
        model = ensemble.get_model_branch(args.branch.lower())

    out_path = Path(args.output)
    export_to_onnx(
        model=model,
        output_path=out_path,
        input_shape=(1, 3, IMAGE_SIZE, IMAGE_SIZE),
        quantize_int8=args.int8,
        verbose=True
    )


def cli_benchmark(args):
    """Command: Benchmark PyTorch vs ONNX Runtime latency."""
    print("⏳ Initializing models for latency benchmarking...")
    pytorch_model = SoftVotingEnsemble(pretrained=False)
    
    onnx_path = Path(args.onnx_model)
    if not onnx_path.exists():
        print(f"⚠️ ONNX model not found at {onnx_path}. Auto-exporting temporary ONNX model...")
        onnx_path = Path("outputs/temp_ensemble.onnx")
        export_to_onnx(pytorch_model, onnx_path, verbose=False)

    engine = ONNXInferenceEngine(onnx_path, use_gpu=args.gpu)
    print(f"⏱️ Running {args.runs} benchmark iterations...")
    
    report = benchmark_latency(
        pytorch_model=pytorch_model,
        onnx_engine=engine,
        num_runs=args.runs,
        device=DEVICE if args.gpu else "cpu"
    )

    print("\n" + "=" * 65)
    print(" ⚡ INFERENCE LATENCY BENCHMARK REPORT")
    print("=" * 65)
    print(f"PyTorch Native Latency : {report['pytorch_latency_ms']['mean']:.2f} ± {report['pytorch_latency_ms']['std']:.2f} ms")
    print(f"ONNX Runtime Latency   : {report['onnx_latency_ms']['mean']:.2f} ± {report['onnx_latency_ms']['std']:.2f} ms")
    print(f"ONNX Provider          : {report['onnx_provider']}")
    print(f"Speedup Ratio          : {report['speedup_ratio']:.2f}x")
    print("=" * 65 + "\n")


def cli_serve(args):
    """Command: Launch FastAPI production server."""
    try:
        import uvicorn
    except ImportError:
        print("❌ Error: uvicorn is not installed. Please run: pip install uvicorn")
        sys.exit(1)

    print(f"🚀 Starting Automated Liver Staging API server on http://{args.host}:{args.port}")
    print(f"📚 Interactive Swagger UI available at http://{args.host}:{args.port}/docs")
    uvicorn.run("src.api.main:app", host=args.host, port=args.port, reload=args.reload)


def app():
    """Main CLI parser."""
    parser = argparse.ArgumentParser(
        prog="liver-stage",
        description="Automated Liver Fibrosis Staging (ALS) Clinical AI Platform CLI"
    )
    subparsers = parser.add_subparsers(title="commands", dest="command", help="Available subcommands")

    # Predict subcommand
    p_pred = subparsers.add_parser("predict", help="Classify histological slide image(s)")
    p_pred.add_argument("--image", "-i", required=True, help="Path to slide image or folder")
    p_pred.add_argument("--output", "-o", help="Path to save JSON predictions report")
    p_pred.set_defaults(func=cli_predict)

    # Export ONNX subcommand
    p_export = subparsers.add_parser("export-onnx", help="Export PyTorch model to ONNX")
    p_export.add_argument("--branch", "-b", default="ensemble", choices=["ensemble", "convnextv2", "mednext", "deit", "resnet"], help="Model branch to export")
    p_export.add_argument("--output", "-o", default="outputs/checkpoints/ensemble.onnx", help="Output path for .onnx file")
    p_export.add_argument("--int8", action="store_true", help="Also generate INT8 dynamically quantized model")
    p_export.set_defaults(func=cli_export_onnx)

    # Benchmark subcommand
    p_bench = subparsers.add_parser("benchmark", help="Benchmark PyTorch vs ONNX latency")
    p_bench.add_argument("--onnx-model", "-m", default="outputs/checkpoints/ensemble.onnx", help="Path to exported ONNX model")
    p_bench.add_argument("--runs", "-r", type=int, default=30, help="Number of benchmark runs")
    p_bench.add_argument("--gpu", action="store_true", help="Use CUDA GPU if available")
    p_bench.set_defaults(func=cli_benchmark)

    # Serve subcommand
    p_serve = subparsers.add_parser("serve", help="Launch FastAPI REST API server")
    p_serve.add_argument("--host", default="0.0.0.0", help="Host IP to bind")
    p_serve.add_argument("--port", "-p", type=int, default=8000, help="Port to bind")
    p_serve.add_argument("--reload", action="store_true", help="Enable auto-reload for development")
    p_serve.set_defaults(func=cli_serve)

    args = parser.parse_args()
    if not hasattr(args, "func"):
        parser.print_help()
        sys.exit(1)

    args.func(args)


if __name__ == "__main__":
    app()
