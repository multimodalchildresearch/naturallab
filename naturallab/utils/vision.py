import io
from pathlib import Path
from typing import Generator, Any, Optional

import cv2
import numpy as np
from PIL import Image
from numpy.typing import NDArray

from naturallab.utils.misc import PathLike

VIDEO_FILETYPES = ['.mp4', '.avi', '.mov']
IMAGE_FILETYPES = ['.jpg', '.jpeg', '.png']


def is_video(path: PathLike) -> bool:
    """
    Check if a file is a video file.
    """
    return Path(path).suffix.lower() in VIDEO_FILETYPES


def is_image(path: PathLike) -> bool:
    """
    Check if a file is an image file.
    """
    return Path(path).suffix.lower() in IMAGE_FILETYPES


def encode_jpg(frame: NDArray[np.uint8], quality: int = 95) -> NDArray[np.uint8]:
    """
    Encode a frame as a jpg image.

    Args:
        frame: The frame to encode.
        quality: The quality of the jpg image.

    Returns:
        The encoded jpg image.
    """
    params = [int(cv2.IMWRITE_JPEG_QUALITY), quality]
    ret, jpg = cv2.imencode('.jpg', frame, params)
    if not ret:
        raise ValueError("Error encoding frame to jpg.")
    return jpg


def laplacian_var(image: NDArray[np.uint8]) -> float:
    """
    Computes the variance of Laplacian
    """
    return cv2.Laplacian(cv2.cvtColor(image, cv2.COLOR_BGR2GRAY), cv2.CV_64F).var()


def open_image_from_bytes(byte_array: bytes) -> Image:
    """
    Open an image from a byte array.
    """
    return Image.open(io.BytesIO(byte_array)).convert("RGB")


def video_iter(path: PathLike, total: Optional[int] = None) -> Generator[NDArray[np.uint8], Any, None]:
    """
    Iterate over the frames of a video file.

    Args:
        path: The path to the video file.
        total: The total number of frames to read. If None, read all frames.

    Yields:
        The frames of the video as numpy arrays.
    """
    if not Path(path).exists():
        raise FileNotFoundError(f"File {path} does not exist.")

    cap = cv2.VideoCapture(str(path))
    if not cap.isOpened():
        raise ValueError("Error opening video stream or file.")
    num_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

    if total is None:
        total = num_frames

    if total > num_frames:
        raise ValueError("Total frames greater than video frames.")

    for _ in range(total):
        ret, frame = cap.read()
        if not ret:
            raise ValueError("Error reading video stream or file.")

        yield frame

    cap.release()
