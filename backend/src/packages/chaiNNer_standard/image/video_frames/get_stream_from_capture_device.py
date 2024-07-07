from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np

from api import Generator, IteratorOutputInfo, NodeContext
from nodes.groups import Condition, if_group
from nodes.impl.video import VideoCapture
from nodes.properties.inputs import BoolInput, FileInput, NumberInput
from nodes.properties.outputs import (
    ImageOutput,
    NumberOutput,
)
from nodes.utils.utils import split_file_path

from .. import video_frames_group

# TODO: get the frames from the capture device and put them into the generator


@video_frames_group.register(
    schema_id="chainner:image:get_frame_from_capture_device",
    name="Get Stream from Capture Device",
    description=[
        "Capture frames from a video capture device.",
        "This iterator captures frames from a video capture device.",
    ],
    icon="MdVideoCameraBack",
    inputs=[
        FileInput(
            primary_input=True,
            label="Capture Device",
            file_kind="capture_device",
            filetypes=["capture_device"],
            has_handle=True,
        ),
        BoolInput("Use limit", default=False).with_id(1),
        if_group(Condition.bool(1, True))(
            NumberInput("Limit", default=10, min=1)
            .with_docs(
                "Limit the number of frames to capture. This can be useful for testing the iterator without capturing all frames from the device."
            )
            .with_id(2)
        ),
    ],
    outputs=[
        ImageOutput("Frame", channels=3),
        NumberOutput(
            "Frame Index",
            output_type="if Input1 { min(uint, Input2 - 1) } else { uint }",
        ).with_docs(
            "A counter that starts at 0 and increments by 1 for each captured frame."
        ),
    ],
    iterator_outputs=IteratorOutputInfo(
        outputs=[0, 1], length_type="if Input1 { min(uint, Input2) } else { uint }"
    ),
    node_context=True,
    kind="generator",
)
def get_stream_from_capture_device_node(
    node_context: NodeContext,
    capture_device: Path,
    use_limit: bool,
    limit: int,
) -> tuple[Generator[tuple[np.ndarray, int]], Path, str, float, Any]:
    video_dir, video_name, _ = split_file_path(capture_device)

    loader = VideoCapture(capture_device)

    frame_count = -1
    if use_limit:
        frame_count = min(frame_count, limit)

    audio_stream = loader.get_audio_stream()

    def iterator():
        for index, frame in enumerate(loader.stream_frames()):
            yield frame, index

            if use_limit and index + 1 >= limit:
                break

    return (
        Generator.from_iter(supplier=iterator, expected_length=frame_count),
        video_dir,
        video_name,
        loader.metadata.fps,
        audio_stream,
    )
