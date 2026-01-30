#!/usr/bin/env python3
from setuptools import setup, find_packages
from pathlib import Path

this_directory = Path(__file__).parent
long_description = (this_directory / "README.md").read_text(encoding='utf-8') if (this_directory / "README.md").exists() else ""

CORE_DEPS = [
    "numpy>=1.19.0",
    "pandas>=1.1.0",
    "opencv-python>=4.5.0",
    "pillow>=10.0.0",
    "torch>=1.7.0",
    "torchvision>=0.8.0",
    "pyyaml>=5.4.0",
    "tqdm>=4.50.0",
]

TRACKING_DEPS = [
    "ultralytics>=8.0.0",
    "deep-sort-realtime>=1.3.0",
    "filterpy>=1.4.5",
    "transformers>=4.30.0",
]

GAZE_DEPS = [
    "transformers>=4.46.0",
    "h5py>=3.0.0",
    "torchmetrics>=0.7.0",
]

ACQUISITION_DEPS = [
    "pylsl>=1.16.0",
    "pyxdf>=1.16.0",
]

ALL_DEPS = TRACKING_DEPS + GAZE_DEPS + ACQUISITION_DEPS + [
    "mediapipe>=0.8.0",
    "matplotlib>=3.3.0",
    "seaborn>=0.11.0",
    "scipy>=1.5.0",
    "gdown>=4.4.0",
    "open-clip-torch>=2.0.0",
]

setup(
    name="naturallab",
    version="1.0.0",
    author="Anonymous",
    author_email="anonymous@example.com",
    description="Multi-modal tracking and analysis system",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/anonymous/naturallab",
    packages=find_packages(),
    classifiers=[
        "Development Status :: 4 - Beta",
        "Intended Audience :: Science/Research",
        "License :: OSI Approved :: MIT License",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.8",
        "Programming Language :: Python :: 3.9",
        "Programming Language :: Python :: 3.10",
        "Programming Language :: Python :: 3.11",
    ],
    python_requires=">=3.8",
    install_requires=CORE_DEPS,
    extras_require={
        "tracking": TRACKING_DEPS,
        "gaze": GAZE_DEPS,
        "acquisition": ACQUISITION_DEPS,
        "all": ALL_DEPS,
    },
    include_package_data=True,
)
