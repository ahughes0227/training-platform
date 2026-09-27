"""Run a small optimizer/checkpoint/reload check in the trainer image.

Use ``python -m defect_platform.trainer.validate --require-gpu`` as the
container GPU certification command. Omitting the flag permits local CPU checks.
"""

from __future__ import annotations

import argparse
import json
import tempfile

import torch

from .runtime import run_synthetic_validation


def main() -> None:
    parser = argparse.ArgumentParser(description="Validate the trainer runtime")
    parser.add_argument("--require-gpu", action="store_true",
                        help="fail unless CUDA is available (container GPU certification)")
    parser.add_argument("--output-dir", help="where validation checkpoint and report are written")
    args = parser.parse_args()
    if args.require_gpu and not torch.cuda.is_available():
        raise SystemExit("GPU validation requested but CUDA is unavailable")
    output = args.output_dir or tempfile.mkdtemp(prefix="defect-trainer-check-")
    result = run_synthetic_validation(output)
    result["cuda_name"] = torch.cuda.get_device_name(0) if torch.cuda.is_available() else None
    result["gpu_required"] = args.require_gpu
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
