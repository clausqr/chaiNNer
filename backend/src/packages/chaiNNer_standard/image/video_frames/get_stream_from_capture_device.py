from __future__ import annotations

from pathlib import Path

import numpy as np
from sanic.log import logger

from api import Generator, IteratorOutputInfo, NodeContext
from nodes.groups import Condition, if_group
from nodes.impl.ffmpeg import FFMpegEnv
from nodes.impl.video import VideoCapture
from nodes.properties.inputs import BoolInput, FileInput, NumberInput
from nodes.properties.outputs import (
    DirectoryOutput,
    FileNameOutput,
    ImageOutput,
    NumberOutput,
)
from nodes.utils.utils import split_file_path

from .. import video_frames_group


class VideoCaptureState:
    def __init__(self, path: Path, ffmpeg_env: FFMpegEnv):
        self.loader = VideoCapture(path, ffmpeg_env)
        self.frame_index = 0


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
            .with_docs(
                "Limit the number of frames to iterate over. NOTE: Pressing STOP button breaks things, and makes this node ."
            )
            .with_id(2)
        ),
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
) -> tuple[Generator[tuple[np.ndarray, int]], Path, str, float]:
    video_dir, video_name, _ = split_file_path(path)

    # Retrieve or create state object
    if path not in VIDEO_LOADER_STATES:
        VIDEO_LOADER_STATES[path] = VideoCaptureState(
            path, FFMpegEnv.get_integrated(node_context.storage_dir)
        )
    state = VIDEO_LOADER_STATES[path]

    logger.info(f"VideoCapture state loaded for path {path}.")
    frame_count = state.loader.metadata.frame_count
    if use_limit:
        frame_count = min(frame_count, limit)

    def iterator():
        for index, frame in enumerate(state.loader.stream_frames()):
            yield frame, index
            state.frame_index += 1
            if use_limit and state.frame_index >= limit:
                state.loader.close()
                VIDEO_LOADER_STATES.pop(path)
                break

    return (
        Generator.from_iter(supplier=iterator, expected_length=frame_count),
        video_dir,
        video_name,
        state.loader.metadata.fps,
    )
