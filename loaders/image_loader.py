from __future__ import annotations

import logging
from pathlib import Path
from typing import Any, Dict, Optional

import numpy as np
from PIL import Image

from .base_loader import BaseLoader, LoaderError

logger = logging.getLogger(__name__)


# ==========================================================
# Image Loader
# ==========================================================

class ImageLoader(BaseLoader):
    """
    Production Image Loader.

    Supported Formats
    -----------------
    JPG
    JPEG
    PNG
    BMP
    TIFF
    GIF
    WEBP
    """

    SUPPORTED_EXTENSIONS = [

        ".jpg",

        ".jpeg",

        ".png",

        ".bmp",

        ".tiff",

        ".tif",

        ".gif",

        ".webp",

    ]

    # ======================================================
    # Constructor
    # ======================================================

    def __init__(
        self,
        file_path: str,
    ):

        super().__init__(file_path)

        self.validate_extension(

            self.SUPPORTED_EXTENSIONS

        )

        self.image: Optional[
            Image.Image
        ] = None

        logger.info(

            f"ImageLoader initialized "

            f"for {self.file_name}"

        )

    # ======================================================
    # Open Image
    # ======================================================

    def open_image(
        self,
    ) -> Image.Image:
        """
        Open image.
        """

        if self.image is None:

            try:

                self.image = Image.open(

                    self.file_path

                )

            except Exception as error:

                raise LoaderError(

                    f"Unable to open image: "

                    f"{error}"

                )

        return self.image

    # ======================================================
    # Close Image
    # ======================================================

    def close_image(
        self,
    ) -> None:
        """
        Close image.
        """

        if self.image:

            self.image.close()

            self.image = None

    # ======================================================
    # Image Size
    # ======================================================

    @property
    def dimensions(
        self,
    ) -> tuple[int, int]:
        """
        Width and height.
        """

        image = self.open_image()

        return image.size

    # ======================================================
    # Width
    # ======================================================

    @property
    def width(
        self,
    ) -> int:

        return self.dimensions[0]

    # ======================================================
    # Height
    # ======================================================

    @property
    def height(
        self,
    ) -> int:

        return self.dimensions[1]

    # ======================================================
    # Channels
    # ======================================================

    @property
    def channels(
        self,
    ) -> int:
        """
        Number of channels.
        """

        image = self.open_image()

        return len(

            image.getbands()

        )

    # ======================================================
    # Color Mode
    # ======================================================

    @property
    def color_mode(
        self,
    ) -> str:
        """
        RGB, RGBA, L...
        """

        image = self.open_image()

        return image.mode

    # ======================================================
    # Image Format
    # ======================================================

    @property
    def image_format(
        self,
    ) -> str:
        """
        PNG, JPEG...
        """

        image = self.open_image()

        return image.format

    # ======================================================
    # Metadata
    # ======================================================

    def image_metadata(
        self,
    ) -> Dict[str, Any]:
        """
        Basic image metadata.
        """

        return {

            "width":

                self.width,

            "height":

                self.height,

            "channels":

                self.channels,

            "mode":

                self.color_mode,

            "format":

                self.image_format,

            "file_size":

                self.file_size,

            "extension":

                self.extension,

        }

    # ======================================================
    # Convert to NumPy
    # ======================================================

    def to_numpy(
        self,
    ) -> np.ndarray:
        """
        Convert image to NumPy.
        """

        image = self.open_image()

        return np.asarray(

            image

        )

    # ======================================================
    # Statistics
    # ======================================================

    def statistics(
        self,
    ) -> Dict[str, Any]:
        """
        Basic statistics.
        """

        array = self.to_numpy()

        return {

            "width":

                self.width,

            "height":

                self.height,

            "channels":

                self.channels,

            "pixels":

                self.width

                *

                self.height,

            "minimum":

                float(

                    array.min()

                ),

            "maximum":

                float(

                    array.max()

                ),

            "mean":

                float(

                    array.mean()

                ),

            "std":

                float(

                    array.std()

                ),

        }

    # ======================================================
    # Load
    # ======================================================

    def load(
        self,
    ) -> Dict[str, Any]:
        """
        Load image.
        """

        image = self.open_image()

        return {

            "text": "",

            "image": image,

            "width": self.width,

            "height": self.height,

            "channels": self.channels,

            "metadata":

                self.image_metadata(),

        }

    # ======================================================
    # Preview
    # ======================================================

    def preview(
        self,
    ) -> Image.Image:
        """
        Return PIL image.
        """

        return self.open_image()

    # ======================================================
    # Empty Check
    # ======================================================

    def is_empty(
        self,
    ) -> bool:
        """
        Check zero-byte image.
        """

        return self.file_size == 0

    # ======================================================
    # String Representation
    # ======================================================

    def __repr__(
        self,
    ):

        return (

            "ImageLoader("

            f"file='{self.file_name}', "

            f"size={self.width}x{self.height}"

            ")"

        )
