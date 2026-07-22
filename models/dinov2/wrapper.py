"""
DINOv2 wrapper for SurgMap3D.

Implements the common BaseModel interface using the official
Meta DINOv2 model.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import torch
from PIL import Image
from torchvision import transforms

from models.base_model import BaseModel


class DINOv2Model(BaseModel):
    """
    DINOv2 feature extractor.
    """

    def __init__(
        self,
        checkpoint_path: Path |None = None,
        device: str | None = None,
        model_variant: str = "dinov2_vitb14",
    ) -> None:

        if device is None:
            device = "cuda" if torch.cuda.is_available() else "cpu"

        super().__init__(
            model_name="DINOv2",
            checkpoint_path=checkpoint_path,
            device=device,
        )

        self.model_variant = model_variant
        self.model = None

        self.transform = transforms.Compose([
            transforms.Resize((518, 518)),
            transforms.ToTensor(),
        ])

    def load(self) -> None:
        """
        Load pretrained DINOv2 model.
        """

        self.model = torch.hub.load(
            "facebookresearch/dinov2",
            self.model_variant,
        )

        self.model.to(self.device)
        self.model.eval()

        self.is_loaded = True

    def preprocess(self, image: Image.Image) -> torch.Tensor:
        """
        Convert PIL image to tensor.
        """

        if not isinstance(image, Image.Image):
            raise TypeError("Input must be a PIL image.")

        tensor = self.transform(image)
        tensor = tensor.unsqueeze(0)

        return tensor.to(self.device)

    def predict(self, tensor: torch.Tensor) -> torch.Tensor:
        """
        Extract DINOv2 embedding.
        """

        if not self.is_loaded:
            raise RuntimeError("Call load() first.")

        with torch.no_grad():
            embedding = self.model(tensor)

        return embedding

    def postprocess(self, embedding: torch.Tensor) -> dict[str, Any]:
        """
        Convert embedding to dictionary.
        """

        return {
            "embedding": embedding.cpu(),
            "shape": tuple(embedding.shape),
            "dimension": embedding.shape[-1],
        }

    def unload(self) -> None:
        """
        Free GPU memory.
        """

        if self.model is not None:
            del self.model

        self.model = None
        self.is_loaded = False

        if torch.cuda.is_available():
            torch.cuda.empty_cache()