import subprocess
from dataclasses import dataclass
from io import BufferedIOBase
from pathlib import Path
from typing import Generator

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
    Handles capturing video from a v4l2 device using FFmpeg.
    This class is stateful and is reused across multiple chain runs.
    """

    def __init__(self, path: Path, ffmpeg_env: FFMpegEnv, pix_fmt: str = "bgr24"):
        self.path = path
        self.ffmpeg_env = ffmpeg_env
        self.pix_fmt = (
            pix_fmt  # ffmpeg pixel format to output (e.g. bgr24, yuyv422, gray)
        )
        # Do not probe the device, as it can hang. Instead, create mock
        # metadata to satisfy the node's requirements for properties like
        # frame_count, which is conceptually infinite for a live stream.
        self.metadata = VideoMetadata(
            width=640, height=480, fps=30.0, frame_count=10000
        )
        logger.info(f"VideoCapture stateful instance initialized for {self.path}")

    def stream_frames(self) -> Generator[np.ndarray, None, None]:
        """
        A generator that yields video frames. It handles the entire lifecycle
        of the ffmpeg process for a single streaming session and ensures
        it is cleaned up afterward.
        """
        process: subprocess.Popen | None = None
        width = 640
        height = 480

        # bytes per pixel mapping for constant-bpp formats
        _bpp_map = {
            "bgr24": 3,
            "rgb24": 3,
            "gray": 1,
            "gray8": 1,
            "yuyv422": 2,
        }
        bytes_per_pixel = _bpp_map.get(self.pix_fmt, 3)
        frame_size = width * height * bytes_per_pixel

        try:
            # 1. Start the FFmpeg subprocess for this specific stream session.
            logger.info(f"Attempting to start new FFmpeg stream for {self.path}...")
            process = (
                ffmpeg.input(
                    self.path,
                    f="v4l2",
                    input_format="yuyv422",
                    video_size=f"{width}x{height}",
                )
                .output("pipe:", format="rawvideo", pix_fmt=self.pix_fmt)
                .run_async(
                    pipe_stdout=True, pipe_stderr=True, cmd=self.ffmpeg_env.ffmpeg
                )
            )
            assert process is not None  # For the type checker
            logger.info(f"FFmpeg stream started with PID: {process.pid}")

            # 2. Enter the capture loop.
            while True:
                # Check for termination.
                if process.poll() is not None:
                    break

                # Read a single frame from stdout.
                assert process.stdout is not None
                in_bytes = process.stdout.read(frame_size)
                if not in_bytes:
                    break

                if len(in_bytes) == frame_size:
                    yield np.frombuffer(in_bytes, np.uint8).reshape([height, width, 3])

        except ffmpeg.Error as e:
            stderr = e.stderr.decode(errors="ignore") if e.stderr else "N/A"
            logger.error(f"FFmpeg failed to start for {self.path}. Stderr: {stderr}")
            raise RuntimeError(f"FFmpeg startup failed: {stderr}") from e

        finally:
            # 3. Ensure the process for THIS stream is cleaned up.
            if process:
                logger.info(f"Closing FFmpeg stream with PID: {process.pid}")
                try:
                    if process.poll() is None:
                        process.terminate()
                        process.wait(timeout=1.0)
                except subprocess.TimeoutExpired:
                    logger.warning(
                        f"FFmpeg stream {process.pid} did not terminate, killing."
                    )
                    process.kill()
                except Exception as e:
                    logger.error(f"Error during FFmpeg stream cleanup: {e}")
            logger.info("FFmpeg stream session ended.")
