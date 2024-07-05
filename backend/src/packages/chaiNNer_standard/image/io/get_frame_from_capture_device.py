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
from nodes.properties.inputs import ImageFileInput, FileInput, DirectoryInput
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


def _read_cap(path: Path) -> np.ndarray | None:
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


@io_group.register(
    schema_id="chainner:image:load",
    name="Get Frame from Capture Device",
    description=(
        "Get a frame from the specified capture device. This node will output the captured frame"
        " and the name of the capture device."
    ),
    icon="BsFillImageFill",
    inputs=[
        FileInput(primary_input=True,
                  label="Capture Device",
                  file_kind="capture_device",
                  filetypes=["capture_device"],
                  has_handle=True).with_docs(
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
    try:
        cap = cv2.VideoCapture(str(path))
        if not cap.isOpened():
            raise RuntimeError(
                f'The capture device "{path}" you are trying to get the frame from cannot be read by chaiNNer.'
            )

        frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        ret, img = cap.read()
        cap.release()
    except Exception as e:
        error = e

    if img is None:
        if error is not None:
            raise error
        raise RuntimeError(
            f'The capture device "{path}" you are trying to get the frame from cannot be read by chaiNNer.'
        )

    return img, frame_count, str(path)
