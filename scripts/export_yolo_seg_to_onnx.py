#!/usr/bin/env python3
"""Export a trained Ultralytics YOLO segmentation model to ONNX.

Batch size is intentionally fixed to 1 for deployment compatibility.
"""

from __future__ import annotations

import argparse
import shutil
from pathlib import Path


FIXED_BATCH_SIZE = 1


def validate_onnx_batch_size(path: Path) -> None:
    try:
        import onnx
    except ImportError as exc:
        raise RuntimeError("ONNX validation requires the 'onnx' package. Install requirements.txt first.") from exc

    model = onnx.load(str(path))
    if not model.graph.input:
        raise RuntimeError(f"ONNX model has no graph inputs: {path}")

    first_input = model.graph.input[0]
    shape = first_input.type.tensor_type.shape
    if not shape.dim:
        raise RuntimeError(f"ONNX input has no static shape information: {first_input.name}")

    batch_dim = shape.dim[0]
    batch_value = batch_dim.dim_value
    batch_param = batch_dim.dim_param
    if batch_value != FIXED_BATCH_SIZE:
        detail = f"dim_value={batch_value}" if batch_value else f"dim_param={batch_param!r}"
        raise RuntimeError(f"Expected fixed ONNX batch size 1, got {detail} on input {first_input.name}")


def resolve_output_path(exported_path: Path, output: Path | None, overwrite: bool) -> Path:
    if output is None:
        return exported_path

    output_path = output
    if output_path.suffix.lower() != ".onnx":
        output_path.mkdir(parents=True, exist_ok=True)
        output_path = output_path / exported_path.name
    else:
        output_path.parent.mkdir(parents=True, exist_ok=True)

    if output_path.exists() and not overwrite:
        raise FileExistsError(f"Output already exists: {output_path}. Use --overwrite to replace it.")

    if exported_path.resolve() != output_path.resolve():
        shutil.copy2(exported_path, output_path)
    return output_path


def export_to_onnx(args: argparse.Namespace) -> Path:
    from ultralytics import YOLO

    weights = args.weights
    if not weights.exists():
        raise FileNotFoundError(f"Trained weights not found: {weights}")

    model = YOLO(str(weights))
    export_kwargs = {
        "format": "onnx",
        "imgsz": args.imgsz,
        "batch": FIXED_BATCH_SIZE,
        "dynamic": False,
        "simplify": args.simplify,
        "nms": args.nms,
    }
    if args.opset is not None:
        export_kwargs["opset"] = args.opset
    if args.device is not None:
        export_kwargs["device"] = args.device

    exported = model.export(**export_kwargs)

    exported_path = Path(exported)
    if not exported_path.exists():
        raise RuntimeError(f"Ultralytics export did not produce an ONNX file at: {exported_path}")

    final_path = resolve_output_path(exported_path, args.output, args.overwrite)
    validate_onnx_batch_size(final_path)
    return final_path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--weights",
        type=Path,
        default=Path("runs/segment/train/weights/best.pt"),
        help="Path to trained Ultralytics .pt weights.",
    )
    parser.add_argument("--output", type=Path, default=None, help="Optional ONNX file path or output directory.")
    parser.add_argument("--imgsz", type=int, default=1280, help="Static square export image size.")
    parser.add_argument("--opset", type=int, default=None, help="Optional ONNX opset. Leave unset for Ultralytics default.")
    parser.add_argument("--device", default=None, help="Export device, e.g. 'cpu' or '0'. Leave unset for Ultralytics default.")
    parser.add_argument("--nms", action="store_true", help="Include NMS in the exported model.")
    parser.add_argument("--no-simplify", dest="simplify", action="store_false", help="Disable ONNX simplification.")
    parser.add_argument("--overwrite", action="store_true", help="Allow replacing --output if it already exists.")
    parser.set_defaults(simplify=True)
    return parser.parse_args()


if __name__ == "__main__":
    output_path = export_to_onnx(parse_args())
    print(f"Exported ONNX with fixed batch size {FIXED_BATCH_SIZE}: {output_path}")
