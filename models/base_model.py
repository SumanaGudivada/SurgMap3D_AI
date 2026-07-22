"""
Abstract base class for all AI models used in SurgMap3D.

Every model integrated into the project must inherit from BaseModel.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from pathlib import Path
from typing import Any


class BaseModel(ABC):
    """
    Abstract interface implemented by every AI model.

    Lifecycle:
        load()
            ↓
        preprocess()
            ↓
        predict()
            ↓
        postprocess()
            ↓
        unload()
    """

    def __init__(
        self,
        model_name: str,
        checkpoint_path: Path | None = None,
        device: str = "cpu",
    ) -> None:
        self.model_name = model_name
        self.checkpoint_path = checkpoint_path
        self.device = device
        self.model: Any | None = None
        self.is_loaded = False

    @abstractmethod
    def load(self) -> None:
        """Load the model."""
        pass

    @abstractmethod
    def preprocess(self, input_data: Any) -> Any:
        """Prepare input for inference."""
        pass

    @abstractmethod
    def predict(self, input_data: Any) -> Any:
        """Run model inference."""
        pass

    @abstractmethod
    def postprocess(self, output_data: Any) -> Any:
        """Convert raw predictions into usable output."""
        pass

    @abstractmethod
    def unload(self) -> None:
        """Release model resources."""
        pass

    def __repr__(self) -> str:
        return (
            f"{self.__class__.__name__}("
            f"name='{self.model_name}', "
            f"device='{self.device}', "
            f"loaded={self.is_loaded})"
        )