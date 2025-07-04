from __future__ import annotations

import subprocess
from dataclasses import dataclass
from io import BufferedIOBase
from pathlib import Path
from typing import Generator

import cv2
import ffmpeg
import numpy as np
from sanic.log import logger

from .ffmpeg import FFMpegEnv


@dataclass(frozen=True)
class VideoMetadata:
    width: int
    height: int
    fps: float
    frame_count: int

    @staticmethod
    def from_file(path: Path, ffmpeg_env: FFMpegEnv):
        probe = ffmpeg.probe(path, cmd=ffmpeg_env.ffprobe)
        video_format = probe.get("format", None)
        if video_format is None:
            raise RuntimeError("Failed to get video format. Please report.")
        video_stream = next(
            (stream for stream in probe["streams"] if stream["codec_type"] == "video"),
            None,
        )

        if video_stream is None:
            raise RuntimeError("No video stream found in file")

        width = video_stream.get("width", None)
        if width is None:
            raise RuntimeError("No width found in video stream")
        width = int(width)

        height = video_stream.get("height", None)
        if height is None:
            raise RuntimeError("No height found in video stream")
        height = int(height)

        fps = video_stream.get("r_frame_rate", None)
        if fps is None:
            raise RuntimeError("No fps found in video stream")
        fps = int(fps.split("/")[0]) / int(fps.split("/")[1])

        frame_count = video_stream.get("nb_frames", None)
        if frame_count is None:
            duration = video_stream.get("duration", None)
            if duration is None:
                duration = video_format.get("duration", None)
            if duration is not None:
                frame_count = float(duration) * fps
            else:
                raise RuntimeError(
                    "No frame count or duration found in video stream. Unable to determine video length. Please report."
                )
        frame_count = int(frame_count)

        return VideoMetadata(
            width=width,
            height=height,
            fps=fps,
            frame_count=frame_count,
        )

    @staticmethod
    def from_device(path: Path, ffmpeg_env: FFMpegEnv):
        probe = ffmpeg.probe(path, format="v4l2", cmd=ffmpeg_env.ffprobe)
        video_format = probe.get("format", None)
        if video_format is None:
            raise RuntimeError("Failed to get video format. Please report.")
        video_stream = next(
            (stream for stream in probe["streams"] if stream["codec_type"] == "video"),
            None,
        )

        if video_stream is None:
            raise RuntimeError("No video stream found in file")

        width = video_stream.get("width", None)
        if width is None:
            raise RuntimeError("No width found in video stream")
        width = int(width)

        height = video_stream.get("height", None)
        if height is None:
            raise RuntimeError("No height found in video stream")
        height = int(height)

        fps = video_stream.get("r_frame_rate", None)
        if fps is None:
            raise RuntimeError("No fps found in video stream")
        fps = int(fps.split("/")[0]) / int(fps.split("/")[1])

        # frame_count = video_stream.get("nb_frames", None)
        # if frame_count is None:
        #     duration = video_stream.get("duration", None)
        #     if duration is None:
        #         duration = video_format.get("duration", None)
        #     if duration is not None:
        #         frame_count = float(duration) * fps
        #     else:
        #         raise RuntimeError(
        #             "No frame count or duration found in video stream. Unable to determine video length. Please report."
        #         )
        frame_count = 10000

        return VideoMetadata(
            width=width,
            height=height,
            fps=fps,
            frame_count=frame_count,
        )


class VideoLoader:
    def __init__(self, path: Path, ffmpeg_env: FFMpegEnv):
        self.path = path
        self.ffmpeg_env = ffmpeg_env
        self.metadata = VideoMetadata.from_file(path, ffmpeg_env)
        logger.info(
            f"VideoLoader instantiated with metadata {self.metadata}, id: {id(self)}"
        )

    def get_audio_stream(self):
        return ffmpeg.input(self.path).audio

    def stream_frames(self):
        """
        Returns an iterator that yields frames as BGR uint8 numpy arrays.
        """

        ffmpeg_reader = (
            ffmpeg.input(self.path)
            .output(
                "pipe:",
                format="rawvideo",
                pix_fmt="bgr24",
                sws_flags="lanczos+accurate_rnd+full_chroma_int+full_chroma_inp+bitexact",
                loglevel="error",
            )
            .run_async(pipe_stdout=True, pipe_stderr=False, cmd=self.ffmpeg_env.ffmpeg)
        )
        assert isinstance(ffmpeg_reader, subprocess.Popen)

        with ffmpeg_reader:
            assert isinstance(ffmpeg_reader.stdout, BufferedIOBase)

            width = self.metadata.width
            height = self.metadata.height

            while True:
                in_bytes = ffmpeg_reader.stdout.read(width * height * 3)
                if not in_bytes:
                    logger.debug("Can't receive frame (stream end?). Exiting ...")
                    break

                yield np.frombuffer(in_bytes, np.uint8).reshape([height, width, 3])


class VideoCapture:
    """
    Handles capturing video from a v4l2 device using OpenCV.
    This class is stateful and is reused across multiple chain runs.
    """

    def __init__(
        self,
        path: Path,
        ffmpeg_env: FFMpegEnv,
        pix_fmt: str = "bgr24",
        resolution: str = "640x480",
    ):
        # path is expected to be like '/dev/video0' or an integer index
        try:
            device_index = int(str(path).replace("/dev/video", ""))
        except Exception:
            device_index = 0
        self.cap = cv2.VideoCapture(device_index, cv2.CAP_V4L2)
        try:
            w_str, h_str = resolution.lower().split("x")
            width = int(w_str)
            height = int(h_str)
        except ValueError:
            width, height = 640, 480
        self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, width)
        self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, height)
        self.width = int(self.cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        self.height = int(self.cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        logger.info(f"OpenCV VideoCapture initialized at {self.width}x{self.height}")

    def stream_frames(self) -> Generator[np.ndarray, None, None]:
        while True:
            ret, frame = self.cap.read()
            if not ret:
                logger.warning("OpenCV failed to read frame from camera.")
                break
            yield frame

    def cleanup(self):
        self.cap.release()
        logger.info("OpenCV VideoCapture released.")
