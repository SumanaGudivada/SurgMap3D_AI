"""
Model factory for SurgMap3D.

Creates AI model instances dynamically using the model registry.
"""

from __future__ import annotations

import importlib
from typing import Any

from models.base_model import BaseModel
from models.model_registry import MODEL_REGISTRY


class ModelFactory:
    """
    Factory responsible for creating model instances.
    """

    @staticmethod
    def create(model_name: str, **kwargs: Any) -> BaseModel:
        """
        Create and return a model instance.

        Args:
            model_name: Name of the model registered in MODEL_REGISTRY.
            **kwargs: Additional arguments passed to the model constructor.

        Returns:
            BaseModel instance.

        Raises:
            ValueError: If the model is not registered.
            ImportError: If the wrapper cannot be imported.
            AttributeError: If the class is not found.
        """

        if model_name not in MODEL_REGISTRY:
            raise ValueError(
                f"Unknown model '{model_name}'. "
                f"Available models: {list(MODEL_REGISTRY.keys())}"
            )

        model_info = MODEL_REGISTRY[model_name]

        module = importlib.import_module(model_info["module"])

        model_class = getattr(module, model_info["class"])

        return model_class(**kwargs)