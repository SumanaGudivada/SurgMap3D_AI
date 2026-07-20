from __future__ import annotations

from pathlib import Path

import cv2
import numpy as np

from utils.logger import get_logger


class ImagePreprocessor:
    """Reusable image preprocessing utilities using OpenCV and NumPy."""

    def __init__(self) -> None:
        self.logger = get_logger(__name__)

    def load_image(self, image_path: Path) -> np.ndarray:
        """Load an image from disk into a NumPy array.

        Args:
            image_path: Path to the image file.

        Returns:
            The loaded image as a NumPy array.

        Raises:
            FileNotFoundError: If the image file does not exist.
            RuntimeError: If the image cannot be loaded.
        """
        if not image_path.exists():
            raise FileNotFoundError(f"Image file not found: {image_path}")

        image = cv2.imread(str(image_path), cv2.IMREAD_COLOR)
        if image is None:
            self.logger.error("Failed to load image: %s", image_path)
            raise RuntimeError(f"Unable to load image: {image_path}")

        self.logger.info("Loaded image from %s", image_path)
        return image

    def resize_image(self, image: np.ndarray, width: int, height: int) -> np.ndarray:
        """Resize an image to the requested dimensions.

        Args:
            image: Input image array.
            width: Target width in pixels.
            height: Target height in pixels.

        Returns:
            The resized image array.

        Raises:
            ValueError: If width or height are not positive.
            TypeError: If the input is not a NumPy array.
        """
        if not isinstance(image, np.ndarray):
            raise TypeError("image must be a numpy.ndarray")
        if image.size == 0:
            raise ValueError("Input image is empty.")
        if width <= 0 or height <= 0:
            raise ValueError("width and height must be positive integers")

        resized = cv2.resize(image, (width, height), interpolation=cv2.INTER_AREA)
        self.logger.info("Resized image to %dx%d", width, height)
        return resized

    def convert_bgr_to_rgb(self, image: np.ndarray) -> np.ndarray:
        """Convert an image from BGR color order to RGB.

        Args:
            image: Input BGR image array.

        Returns:
            The image in RGB color order.

        Raises:
            TypeError: If the input is not a NumPy array.
        """
        if not isinstance(image, np.ndarray):
            raise TypeError("image must be a numpy.ndarray")
        if image.size == 0:
            raise ValueError("Input image is empty.")

        rgb_image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        self.logger.info("Converted image from BGR to RGB")
        return rgb_image

    def normalize_image(self, image: np.ndarray) -> np.ndarray:
        """Normalize image pixel values to the range [0, 1].

        Args:
            image: Input image array.

        Returns:
            The normalized image array.

        Raises:
            TypeError: If the input is not a NumPy array.
        """
        if not isinstance(image, np.ndarray):
            raise TypeError("image must be a numpy.ndarray")
        if image.size == 0:
            raise ValueError("Input image is empty.")

        normalized = image.astype(np.float32) / 255.0
        self.logger.info("Normalized image pixel values to [0, 1]")
        return normalized

    def is_blurry(self, image: np.ndarray, threshold: float = 100.0) -> bool:
        """Determine whether an image is blurry using the variance of the Laplacian.

        This method expects an OpenCV BGR image and uses the variance of the
        Laplacian of the grayscale conversion to assess blur.

        Args:
            image: Input BGR image array.
            threshold: Variance threshold below which the image is blurry.

        Returns:
            True if the image is blurry, otherwise False.

        Raises:
            TypeError: If the input is not a NumPy array.
            ValueError: If threshold is not positive.
        """
        if not isinstance(image, np.ndarray):
            raise TypeError("image must be a numpy.ndarray")
        if image.size == 0:
            raise ValueError("Input image is empty.")
        if threshold <= 0:
            raise ValueError("threshold must be a positive number")

        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        variance = cv2.Laplacian(gray, cv2.CV_64F).var()
        blurry = variance < threshold

        self.logger.info(
            "Computed blur variance %.2f with threshold %.2f: blurry=%s",
            variance,
            threshold,
            blurry,
        )
        return blurry

    def save_image(self, image: np.ndarray, output_path: Path) -> None:
        """Save an image to disk, creating directories as needed.

        Args:
            image: Image array to save.
            output_path: Destination file path.

        Raises:
            TypeError: If the input is not a NumPy array.
            RuntimeError: If the image cannot be written.
        """
        if not isinstance(image, np.ndarray):
            raise TypeError("image must be a numpy.ndarray")
        if image.size == 0:
            raise ValueError("Input image is empty.")

        output_path.parent.mkdir(parents=True, exist_ok=True)
        success = cv2.imwrite(str(output_path), image)
        if not success:
            self.logger.error("Failed to save image to %s", output_path)
            raise RuntimeError(f"Unable to save image: {output_path}")

        self.logger.info("Saved image to %s", output_path)
