#!/usr/bin/env python3
"""Check TokBench evaluation runtime dependencies."""

import importlib.util
import sys


REQUIRED_MODULES = {
    "torch": "torch",
    "torchvision": "torchvision",
    "numpy": "numpy",
    "cv2": "opencv-python",
    "PIL": "Pillow",
    "tqdm": "tqdm",
    "defusedxml": "defusedxml",
    "scipy": "scipy",
    "h5py": "h5py",
    "pypdfium2": "pypdfium2",
    "pyclipper": "pyclipper",
    "shapely": "shapely",
    "langdetect": "langdetect",
    "rapidfuzz": "rapidfuzz",
    "huggingface_hub": "huggingface-hub",
    "anyascii": "anyascii",
    "prettytable": "prettytable",
    "insightface": "insightface",
    "onnx": "onnx",
    "imageio": "imageio[ffmpeg]",
    "nltk": "nltk",
    "onnxruntime": "onnxruntime-gpu",
    "skimage": "scikit-image",
}


def main() -> int:
    missing = [
        package_name
        for module_name, package_name in REQUIRED_MODULES.items()
        if importlib.util.find_spec(module_name) is None
    ]
    if not missing:
        return 0

    print("Missing Python packages: " + ", ".join(missing), file=sys.stderr)
    print("Install TokBench eval dependencies with: pip install -r requirements.txt", file=sys.stderr)
    return 1


if __name__ == "__main__":
    raise SystemExit(main())
