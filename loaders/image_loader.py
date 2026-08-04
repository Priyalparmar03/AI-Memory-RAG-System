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

# ======================================================
# OCR using Tesseract
# ======================================================

def extract_text(
    self,
) -> str:
    """
    Extract text using Tesseract OCR.
    """

    try:

        import pytesseract

    except ImportError:

        raise LoaderError(

            "Install pytesseract."

        )

    image = self.open_image()

    return pytesseract.image_to_string(

        image

    )


# ======================================================
# OCR using EasyOCR
# ======================================================

def easyocr_text(
    self,
    languages: Optional[list[str]] = None,
) -> str:
    """
    OCR using EasyOCR.
    """

    try:

        import easyocr

    except ImportError:

        raise LoaderError(

            "Install easyocr."

        )

    if languages is None:

        languages = ["en"]

    reader = easyocr.Reader(

        languages,

        gpu=False,

    )

    result = reader.readtext(

        str(self.file_path),

        detail=0,

    )

    return "\n".join(result)


# ======================================================
# Grayscale
# ======================================================

def grayscale(
    self,
) -> Image.Image:
    """
    Convert image to grayscale.
    """

    image = self.open_image()

    return image.convert(

        "L"

    )


# ======================================================
# Resize
# ======================================================

def resize(
    self,
    width: int,
    height: int,
) -> Image.Image:
    """
    Resize image.
    """

    image = self.open_image()

    return image.resize(

        (

            width,

            height,

        )

    )


# ======================================================
# Rotate
# ======================================================

def rotate(
    self,
    angle: float,
) -> Image.Image:
    """
    Rotate image.
    """

    image = self.open_image()

    return image.rotate(

        angle,

        expand=True,

    )


# ======================================================
# Crop
# ======================================================

def crop(
    self,
    left: int,
    upper: int,
    right: int,
    lower: int,
) -> Image.Image:
    """
    Crop image.
    """

    image = self.open_image()

    return image.crop(

        (

            left,

            upper,

            right,

            lower,

        )

    )


# ======================================================
# Thumbnail
# ======================================================

def thumbnail(
    self,
    size: tuple[int, int] = (256, 256),
) -> Image.Image:
    """
    Generate thumbnail.
    """

    image = self.open_image().copy()

    image.thumbnail(

        size

    )

    return image


# ======================================================
# Gaussian Blur
# ======================================================

def blur(
    self,
    radius: float = 2.0,
) -> Image.Image:
    """
    Apply Gaussian blur.
    """

    from PIL import ImageFilter

    image = self.open_image()

    return image.filter(

        ImageFilter.GaussianBlur(

            radius

        )

    )


# ======================================================
# Sharpen
# ======================================================

def sharpen(
    self,
) -> Image.Image:
    """
    Sharpen image.
    """

    from PIL import ImageFilter

    image = self.open_image()

    return image.filter(

        ImageFilter.SHARPEN

    )


# ======================================================
# Enhance Contrast
# ======================================================

def enhance_contrast(
    self,
    factor: float = 1.5,
) -> Image.Image:
    """
    Increase image contrast.
    """

    from PIL import ImageEnhance

    image = self.open_image()

    enhancer = ImageEnhance.Contrast(

        image

    )

    return enhancer.enhance(

        factor

    )


# ======================================================
# Preprocess for OCR
# ======================================================

def preprocess(
    self,
) -> Image.Image:
    """
    OCR preprocessing pipeline.
    """

    image = self.grayscale()

    image = image.point(

        lambda pixel:

            255

            if pixel > 160

            else 0

    )

    return image


# ======================================================
# Save Image
# ======================================================

def save(
    self,
    image: Image.Image,
    output_path: str,
) -> str:
    """
    Save processed image.
    """

    image.save(

        output_path

    )

    return output_path

# ======================================================
# Extract EXIF Metadata
# ======================================================

def extract_exif(
    self,
) -> Dict[str, Any]:
    """
    Extract EXIF metadata.
    """

    from PIL.ExifTags import TAGS

    image = self.open_image()

    exif = image.getexif()

    if not exif:

        return {}

    metadata = {}

    for tag_id, value in exif.items():

        tag = TAGS.get(

            tag_id,

            tag_id,

        )

        metadata[tag] = value

    return metadata


# ======================================================
# Histogram
# ======================================================

def histogram(
    self,
) -> list[int]:
    """
    Image histogram.
    """

    image = self.open_image()

    return image.histogram()


# ======================================================
# Dominant Color
# ======================================================

def dominant_color(
    self,
) -> tuple[int, int, int]:
    """
    Compute dominant RGB color.
    """

    image = self.open_image()

    image = image.convert(

        "RGB"

    )

    image = image.resize(

        (

            150,

            150,

        )

    )

    pixels = np.array(

        image

    ).reshape(

        -1,

        3,

    )

    unique, counts = np.unique(

        pixels,

        axis=0,

        return_counts=True,

    )

    dominant = unique[

        np.argmax(

            counts

        )

    ]

    return tuple(

        int(v)

        for v in dominant

    )


# ======================================================
# Average Color
# ======================================================

def average_color(
    self,
) -> tuple[int, int, int]:
    """
    Compute average RGB color.
    """

    image = self.open_image()

    image = image.convert(

        "RGB"

    )

    pixels = np.asarray(

        image

    )

    avg = pixels.mean(

        axis=(0, 1)

    )

    return tuple(

        int(v)

        for v in avg

    )


# ======================================================
# Brightness
# ======================================================

def brightness(
    self,
) -> float:
    """
    Average brightness.
    """

    gray = np.asarray(

        self.grayscale()

    )

    return round(

        float(

            gray.mean()

        ),

        2,

    )


# ======================================================
# Contrast
# ======================================================

def contrast(
    self,
) -> float:
    """
    Image contrast.
    """

    gray = np.asarray(

        self.grayscale()

    )

    return round(

        float(

            gray.std()

        ),

        2,

    )


# ======================================================
# Blur Detection
# ======================================================

def blur_detection(
    self,
    threshold: float = 100.0,
) -> Dict[str, Any]:
    """
    Detect blurry image using
    Laplacian variance.
    """

    try:

        import cv2

    except ImportError:

        raise LoaderError(

            "Install opencv-python."

        )

    image = cv2.imread(

        str(

            self.file_path

        )

    )

    gray = cv2.cvtColor(

        image,

        cv2.COLOR_BGR2GRAY,

    )

    variance = cv2.Laplacian(

        gray,

        cv2.CV_64F,

    ).var()

    return {

        "variance":

            round(

                float(

                    variance

                ),

                2,

            ),

        "is_blurry":

            variance < threshold,

    }


# ======================================================
# Face Detection
# ======================================================

def face_detection(
    self,
) -> Dict[str, Any]:
    """
    Detect faces using
    OpenCV Haar Cascade.
    """

    try:

        import cv2

    except ImportError:

        raise LoaderError(

            "Install opencv-python."

        )

    image = cv2.imread(

        str(

            self.file_path

        )

    )

    gray = cv2.cvtColor(

        image,

        cv2.COLOR_BGR2GRAY,

    )

    detector = cv2.CascadeClassifier(

        cv2.data.haarcascades

        +

        "haarcascade_frontalface_default.xml"

    )

    faces = detector.detectMultiScale(

        gray,

        scaleFactor=1.1,

        minNeighbors=5,

    )

    return {

        "count":

            len(

                faces

            ),

        "faces":

            [

                {

                    "x": int(x),

                    "y": int(y),

                    "width": int(w),

                    "height": int(h),

                }

                for (

                    x,

                    y,

                    w,

                    h,

                )

                in faces

            ],

    }


# ======================================================
# Edge Detection
# ======================================================

def edge_detection(
    self,
):
    """
    Detect edges using Canny.
    """

    try:

        import cv2

    except ImportError:

        raise LoaderError(

            "Install opencv-python."

        )

    image = cv2.imread(

        str(

            self.file_path

        )

    )

    gray = cv2.cvtColor(

        image,

        cv2.COLOR_BGR2GRAY,

    )

    return cv2.Canny(

        gray,

        100,

        200,

    )


# ======================================================
# Save Current Image
# ======================================================

def save_image(
    self,
    output_path: str,
) -> str:
    """
    Save current image.
    """

    image = self.open_image()

    image.save(

        output_path

    )

    return output_path


# ======================================================
# Image Analysis
# ======================================================

def analyze(
    self,
) -> Dict[str, Any]:
    """
    Comprehensive image analysis.
    """

    return {

        "metadata":

            self.image_metadata(),

        "statistics":

            self.statistics(),

        "brightness":

            self.brightness(),

        "contrast":

            self.contrast(),

        "dominant_color":

            self.dominant_color(),

        "average_color":

            self.average_color(),

        "blur":

            self.blur_detection(),

        "faces":

            self.face_detection(),

        "exif":

            self.extract_exif(),

    }

# ======================================================
# Diagnostics
# ======================================================

def diagnostics(
    self,
) -> Dict[str, Any]:
    """
    Image diagnostics.
    """

    return {

        "loader":

            self.__class__.__name__,

        "file":

            self.file_name,

        "extension":

            self.extension,

        "mime_type":

            self.mime_type,

        "file_size":

            self.file_size,

        "dimensions":

            (

                self.width,

                self.height,

            ),

        "channels":

            self.channels,

        "format":

            self.image_format,

        "mode":

            self.color_mode,

        "statistics":

            self.statistics(),

    }


# ======================================================
# Benchmark
# ======================================================

def benchmark(
    self,
) -> Dict[str, Any]:
    """
    Benchmark image loading.
    """

    import time

    start = time.perf_counter()

    image = self.to_numpy()

    elapsed = (

        time.perf_counter()

        -

        start

    )

    pixels = (

        self.width

        *

        self.height

    )

    return {

        "execution_time":

            round(

                elapsed,

                4,

            ),

        "pixels_per_second":

            round(

                pixels

                /

                max(

                    elapsed,

                    1e-9,

                ),

                2,

            ),

        "megapixels":

            round(

                pixels

                /

                1_000_000,

                2,

            ),

    }


# ======================================================
# Validate Image
# ======================================================

def validate(
    self,
) -> bool:
    """
    Validate image integrity.
    """

    try:

        image = Image.open(

            self.file_path

        )

        image.verify()

        return True

    except Exception:

        return False


# ======================================================
# Reload
# ======================================================

def reload(
    self,
) -> None:
    """
    Reload image.
    """

    self.close_image()

    self.open_image()


# ======================================================
# Export NumPy
# ======================================================

def export_numpy(
    self,
    output_path: str,
) -> str:
    """
    Save image array.
    """

    array = self.to_numpy()

    np.save(

        output_path,

        array,

    )

    return output_path


# ======================================================
# Export Metadata
# ======================================================

def export_metadata(
    self,
) -> Dict[str, Any]:
    """
    Export metadata.
    """

    metadata = self.metadata()

    metadata.update(

        self.image_metadata()

    )

    metadata.update(

        self.statistics()

    )

    metadata["exif"] = (

        self.extract_exif()

    )

    return metadata


# ======================================================
# Summary
# ======================================================

def summary(
    self,
) -> Dict[str, Any]:
    """
    Human-readable summary.
    """

    stats = self.statistics()

    return {

        "file":

            self.file_name,

        "width":

            self.width,

        "height":

            self.height,

        "channels":

            self.channels,

        "format":

            self.image_format,

        "brightness":

            self.brightness(),

        "contrast":

            self.contrast(),

        "faces":

            self.face_detection()[

                "count"

            ],

        "blurry":

            self.blur_detection()[

                "is_blurry"

            ],

        "pixels":

            stats["pixels"],

    }


# ======================================================
# Cleanup
# ======================================================

def cleanup(
    self,
) -> None:
    """
    Release resources.
    """

    self.close_image()

    logger.info(

        "Image resources released."

    )


# ======================================================
# Context Manager
# ======================================================

def __enter__(
    self,
):

    self.open_image()

    return self


def __exit__(
    self,
    exc_type,
    exc_value,
    traceback,
):

    self.cleanup()


# ======================================================
# Python Protocols
# ======================================================

def __len__(
    self,
):

    return (

        self.width

        *

        self.height

    )


def __iter__(
    self,
):
    """
    Iterate over rows.
    """

    image = self.to_numpy()

    return iter(

        image

    )


def __getitem__(
    self,
    index,
):
    """
    Access image row.
    """

    image = self.to_numpy()

    return image[index]


def __contains__(
    self,
    value,
):
    """
    Check pixel value.
    """

    image = self.to_numpy()

    return (

        value

        in

        image

    )


def __repr__(
    self,
):

    return (

        "ImageLoader("

        f"file='{self.file_name}', "

        f"size={self.width}x{self.height}, "

        f"mode='{self.color_mode}'"

        ")"

    )
