from __future__ import annotations

from enum import Enum
from pathlib import Path

import numpy as np
from sanic.log import logger

from api import Generator, IteratorOutputInfo, NodeContext
from nodes.groups import Condition, if_group
from nodes.impl.ffmpeg import FFMpegEnv
from nodes.impl.video import VideoCapture
from nodes.properties.inputs import BoolInput, EnumInput, FileInput, NumberInput
from nodes.properties.outputs import (
    DirectoryOutput,
    FileNameOutput,
    ImageOutput,
    NumberOutput,
)
from nodes.utils.utils import split_file_path
from progress_controller import Aborted

from .. import video_frames_group


class PixelFormat(str, Enum):
    MJPEG = "mjpeg"
    BGR24 = "bgr24"
    YUYV422 = "yuyv422"
    GRAY8 = "gray"


class Resolution(str, Enum):
    R160x120 = "160x120"
    R320x240 = "320x240"
    R640x480 = "640x480"
    R1280x720 = "1280x720"


class VideoCaptureState:
    def __init__(
        self, path: Path, ffmpeg_env: FFMpegEnv, pix_fmt: str, resolution: str
    ):
        self.path = path
        self.ffmpeg_env = ffmpeg_env
        self.pix_fmt = pix_fmt
        self.resolution = resolution
        self.loader = None
        self.frame_index = 0
        self._create_loader()

    def _create_loader(self):
        """Create a new VideoCapture instance"""
        self.loader = VideoCapture(
            self.path, self.ffmpeg_env, self.pix_fmt, self.resolution
        )
        self.frame_index = 0

    def reset(self):
        """Reset the state for a new capture session"""
        self._create_loader()

    def cleanup(self):
        """Clean up resources"""
        # The new VideoCapture class manages its own cleanup internally
        # via the stream_frames generator's finally block. No explicit
        # close call is needed here anymore.
        self.loader = None

    def update_settings(self, pix_fmt: str, resolution: str):
        """Update settings and recreate loader if changed"""
        if self.pix_fmt != pix_fmt or self.resolution != resolution:
            self.pix_fmt = pix_fmt
            self.resolution = resolution
            self._create_loader()


# Global dictionary for storing state
VIDEO_LOADER_STATES = {}


@video_frames_group.register(
    schema_id="chainner:image:get_stream_from_capture_device",
    name="Get Stream from Capture Device",
    description=[
        "Get a stream of frames from a capture device.",
        "Uses FFMPEG to read capture device.",
        "(experimental)",
    ],
    icon="MdVideoCameraBack",
    inputs=[
        FileInput(
            primary_input=True,
            label="Capture Device",
            file_kind="capture_device",
            filetypes=["capture_device"],
            has_handle=True,
        ).with_docs(
            "Specify the capture device to get the stream from (e.g. /dev/video0)."
        ),
        BoolInput("Use limit", default=True).with_id(1),
        if_group(Condition.bool(1, True))(
            NumberInput("Limit", default=200, min=1)
            .with_docs("Limit the number of frames to iterate over.")
            .with_id(2)
        ),
        EnumInput(
            PixelFormat,
            label="Pixel Format",
            default=PixelFormat.BGR24,
        ).with_id(3),
        EnumInput(
            Resolution,
            label="Resolution",
            default=Resolution.R640x480,
        ).with_id(4),
    ],
    outputs=[
        ImageOutput("Frame", channels=3),
        NumberOutput(
            "Frame Index",
            output_type="if Input1 { min(uint, Input2 - 1) } else { uint }",
        ).with_docs("A counter that starts at 0 and increments by 1 for each frame."),
        DirectoryOutput("Capture Device Directory", of_input=0),
        FileNameOutput("Capture Device Name", of_input=0),
        NumberOutput("FPS", output_type="0.."),
    ],
    iterator_outputs=IteratorOutputInfo(
        outputs=[0, 1], length_type="if Input1 { min(uint, Input2) } else { uint }"
    ),
    node_context=True,
    kind="generator",
)
def get_stream_from_capture_device_node(
    node_context: NodeContext,
    path: Path,
    use_limit: bool,
    limit: int,
    pixel_format: PixelFormat,
    resolution: Resolution,
) -> tuple[Generator[tuple[np.ndarray, int]], Path, str, float]:
    video_dir, video_name, _ = split_file_path(path)

    # Retrieve or create state object
    if path not in VIDEO_LOADER_STATES:
        VIDEO_LOADER_STATES[path] = VideoCaptureState(
            path,
            FFMpegEnv.get_integrated(node_context.storage_dir),
            pixel_format.value,
            resolution.value,
        )

    # `state` is now guaranteed to exist
    state = VIDEO_LOADER_STATES[path]

    # Update settings if the pixel format or resolution has changed
    if state.pix_fmt != pixel_format.value or state.resolution != resolution.value:
        state.update_settings(pixel_format.value, resolution.value)

    # Add cleanup function to node context - only clean up after the node is done
    def cleanup_state():
        if path in VIDEO_LOADER_STATES:
            VIDEO_LOADER_STATES[path].cleanup()
            VIDEO_LOADER_STATES.pop(path, None)
            logger.info(f"Cleaned up VideoCapture state for path {path}")

    # Only add cleanup after the node execution is complete
    node_context.add_cleanup(cleanup_state, after="chain")

    logger.info(f"VideoCapture state loaded for path {path}.")
    # The frame count for a live stream is conceptually infinite.
    # Use the user-defined limit if provided, otherwise a large number.
    frame_count = limit if use_limit else 1000000

    def iterator():
        try:
            # Ensure we have a valid loader
            if state.loader is None:
                state._create_loader()

            for index, frame in enumerate(state.loader.stream_frames()):
                # Check if execution was aborted
                if node_context.aborted:
                    logger.info("Video capture aborted by user")
                    break

                yield frame, index
                state.frame_index += 1
                if use_limit and state.frame_index >= limit:
                    break
        except Aborted:
            logger.info("Video capture aborted")
            raise
        except Exception as e:
            logger.error(f"Error during video capture: {e}")
            # Clean up state on error
            state.cleanup()
            VIDEO_LOADER_STATES.pop(path, None)
            raise

    # Hardcode a reasonable FPS, as probing is unreliable for live devices.
    fps = 30.0

    return (
        Generator.from_iter(supplier=iterator, expected_length=frame_count),
        video_dir,
        video_name,
        fps,
    )
