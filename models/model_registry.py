"""
Model registry for SurgMap3D.

Stores metadata about every AI model available in the project.
"""

from __future__ import annotations

from pathlib import Path

MODEL_REGISTRY = {
    "dinov2": {
        "name": "DINOv2",
        "module": "models.dinov2.wrapper",
        "class": "DINOv2Model",
        "task": "Feature Extraction",
        "checkpoint_dir": Path("checkpoints/dinov2"),
    },
    "endo3r": {
        "name": "Endo3R",
        "module": "models.endo3r.wrapper",
        "class": "Endo3RModel",
        "task": "Monocular 3D Reconstruction",
        "checkpoint_dir": Path("checkpoints/endo3r"),
    },
    "must3r": {
        "name": "MUSt3R",
        "module": "models.must3r.wrapper",
        "class": "MUSt3RModel",
        "task": "Dense Matching",
        "checkpoint_dir": Path("checkpoints/must3r"),
    },
    "dens3r": {
        "name": "Dens3R",
        "module": "models.dens3r.wrapper",
        "class": "Dens3RModel",
        "task": "Dense Reconstruction",
        "checkpoint_dir": Path("checkpoints/dens3r"),
    },
    "omnivggt": {
        "name": "OmniVGGT",
        "module": "models.omnivggt.wrapper",
        "class": "OmniVGGTModel",
        "task": "Geometry Estimation",
        "checkpoint_dir": Path("checkpoints/omnivggt"),
    },
    "sam2": {
        "name": "SAM2",
        "module": "models.sam2.wrapper",
        "class": "SAM2Model",
        "task": "Segmentation",
        "checkpoint_dir": Path("checkpoints/sam2"),
    },
}