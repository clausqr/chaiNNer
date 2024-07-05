from __future__ import annotations

import os
import platform
from pathlib import Path
from typing import Callable, Iterable, Union

import cv2
import numpy as np
import pillow_avif  # type: ignore # noqa: F401
from PIL import Image
from sanic.log import logger

from nodes.impl.dds.texconv import dds_to_png_texconv
from nodes.impl.image_formats import (
    get_available_image_formats,
    get_opencv_formats,
    get_pil_formats,
)
from nodes.properties.inputs import ImageFileInput
from nodes.properties.outputs import DirectoryOutput, FileNameOutput, LargeImageOutput, NumberOutput
from nodes.utils.utils import get_h_w_c, split_file_path

from .. import io_group

_Decoder = Callable[[Path], Union[np.ndarray, None]]
"""
An image decoder.

Of the given image is naturally not supported, the decoder may return `None`
instead of raising an exception. E.g. when the file extension indicates an
unsupported format.
"""


def get_ext(path: Path | str) -> str:
    return split_file_path(path)[2].lower()


def remove_unnecessary_alpha(img: np.ndarray) -> np.ndarray:
    """
    Removes the alpha channel from an image if it is not used.
    """
    if get_h_w_c(img)[2] != 4:
        return img

    unnecessary = (
        (img.dtype == np.uint8 and np.all(img[:, :, 3] == 255))
        or (img.dtype == np.uint16 and np.all(img[:, :, 3] == 65536))
        or (img.dtype == np.float32 and np.all(img[:, :, 3] == 1.0))
        or (img.dtype == np.float64 and np.all(img[:, :, 3] == 1.0))
    )

    if unnecessary:
        return img[:, :, :3]
    return img


def _read_cv(path: Path) -> np.ndarray | None:
    if get_ext(path) not in get_opencv_formats():
        # not supported
        return None

    img = None
    try:
        img = cv2.imdecode(np.fromfile(path, dtype=np.uint8), cv2.IMREAD_UNCHANGED)
    except Exception as cv_err:
        logger.warning(f"Error loading image, trying with imdecode: {cv_err}")

    if img is None:
        try:
            img = cv2.imread(str(path), cv2.IMREAD_UNCHANGED)
        except Exception as e:
            raise RuntimeError(
                f'Error reading image image from path "{path}". Image may be corrupt.'
            ) from e

    if img is None:  # type: ignore
        raise RuntimeError(
            f'Error reading image image from path "{path}". Image may be corrupt.'
        )

    return img


def _read_pil(path: Path) -> np.ndarray | None:
    if get_ext(path) not in get_pil_formats():
        # not supported
        return None

    im = Image.open(path)
    if im.mode == "P":
        # convert color palette to actual colors
        im = im.convert(im.palette.mode)

    img = np.array(im)
    _, _, c = get_h_w_c(img)
    if c == 3:
        img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
    elif c == 4:
        img = cv2.cvtColor(img, cv2.COLOR_RGBA2BGRA)
    return img


def _read_dds(path: Path) -> np.ndarray | None:
    if get_ext(path) != ".dds":
        # not supported
        return None

    if platform.system() != "Windows":
        # texconv is only supported on Windows.
        return None

    png = dds_to_png_texconv(path)
    try:
        img = _read_cv(png)
        if img is not None:
            img = remove_unnecessary_alpha(img)
        return img
    finally:
        os.remove(png)


def _for_ext(ext: str | Iterable[str], decoder: _Decoder) -> _Decoder:
    ext_set: set[str] = set()
    if isinstance(ext, str):
        ext_set.add(ext)
    else:
        ext_set.update(ext)

    return lambda path: decoder(path) if get_ext(path) in ext_set else None


_decoders: list[tuple[str, _Decoder]] = [
    ("pil-jpeg", _for_ext([".jpg", ".jpeg"], _read_pil)),
    ("cv", _read_cv),
    ("texconv-dds", _read_dds),
    ("pil", _read_pil),
]

valid_formats = get_available_image_formats()

@io_group.register(
    schema_id="chainner:image:load",
    name="Get Frame from Capture Device",
    description=(
        "Get a frame from the specified capture device. This node will output the captured frame"
        " and the name of the capture device."
    ),
    icon="BsFillImageFill",
    inputs=[
        ImageFileInput(primary_input=True).with_docs(
            "Specify the capture device to get the frame from (e.g. /dev/video0)."
        )
    ],
    outputs=[
        LargeImageOutput()
        .with_docs(
            "The node will display a preview of the captured frame as well as type"
            " information for it. Connect this output to the input of another node to"
            " pass the frame to it."
        )
        .suggest(),
        NumberOutput("Frames Available").with_docs(
            "The number of frames available from the capture device."
        ),
        FileNameOutput("Capture Device Path", of_input=0).with_docs(
            "The path of the capture device."
        ),
    ],
    side_effects=True,
)
def get_frame_from_capture_device_node(path: Path) -> tuple[np.ndarray, int, str]:
    logger.debug(f"Getting frame from capture device: {path}")

    img = None
    error = None
    frame_count = 0
    for name, decoder in _decoders:
        try:
            img = decoder(Path(path))
        except Exception as e:
            error = e
            logger.warning(f"Decoder {name} failed")

        if img is not None:
            frame_count += 1

    if img is None:
        if error is not None:
            raise error
        raise RuntimeError(
            f'The capture device "{path}" you are trying to get the frame from cannot be read by chaiNNer.'
        )

    return img, frame_count, str(path)
