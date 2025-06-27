from __future__ import annotations

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

    @staticmethod
    def detect_device_capabilities(
        path: Path, ffmpeg_env: FFMpegEnv
    ) -> tuple[str, str]:
        """
        Detect the best pixel format and resolution for a v4l2 device.
        Returns (pixel_format, resolution) tuple.
        """
        try:
            # Try to probe the device to see what formats are available
            probe = ffmpeg.probe(str(path), f="v4l2", cmd=ffmpeg_env.ffprobe)
            video_stream = next(
                (
                    stream
                    for stream in probe["streams"]
                    if stream["codec_type"] == "video"
                ),
                None,
            )

            if video_stream:
                # Get the actual pixel format from the device
                actual_pix_fmt = video_stream.get("pix_fmt", "yuyv422")
                width = int(video_stream.get("width", 160))
                height = int(video_stream.get("height", 120))

                # Map ffmpeg pixel formats to our internal format names
                format_map = {
                    "yuyv422": "yuyv422",
                    "mjpeg": "mjpeg",
                    "gray": "gray",
                    "bgr24": "bgr24",
                }

                detected_format = format_map.get(actual_pix_fmt, "yuyv422")
                detected_resolution = f"{width}x{height}"

                logger.info(
                    f"Detected device format: {detected_format}, resolution: {detected_resolution}"
                )
                return detected_format, detected_resolution

        except Exception as e:
            logger.warning(f"Could not detect device capabilities for {path}: {e}")

        # Fallback defaults
        return "yuyv422", "160x120"

    def __init__(
        self,
        path: Path,
        ffmpeg_env: FFMpegEnv,
        pix_fmt: str = "bgr24",
        resolution: str = "640x480",
    ):
        self.path = path
        self.ffmpeg_env = ffmpeg_env

        self.requested_fmt = (
            pix_fmt  # user-selected format (mjpeg, yuyv422, gray, bgr24)
        )

        # Determine v4l2 input format and ffmpeg output pixel format
        if self.requested_fmt == "mjpeg":
            self.input_format = "mjpeg"
            self.output_pix_fmt = "bgr24"  # decode JPEG to BGR
        elif self.requested_fmt in ("yuyv422",):
            self.input_format = "yuyv422"
            self.output_pix_fmt = "bgr24"  # convert to BGR inside ffmpeg
        elif self.requested_fmt in ("gray", "gray8"):
            self.input_format = "gray"
            self.output_pix_fmt = "gray"
        else:
            # default assume raw BGR24 supported (rare)
            self.input_format = "bgr24"
            self.output_pix_fmt = "bgr24"

        # Parse resolution string WxH
        try:
            w_str, h_str = resolution.lower().split("x")
            self.width = int(w_str)
            self.height = int(h_str)
        except ValueError:
            self.width, self.height = 640, 480  # fallback

        # Dummy metadata (mainly for other nodes that access fps/frame_count)
        self.metadata = VideoMetadata(
            width=self.width, height=self.height, fps=30.0, frame_count=1000000
        )
        logger.info(f"VideoCapture stateful instance initialized for {self.path}")

    def stream_frames(self) -> Generator[np.ndarray, None, None]:
        """
        A generator that yields video frames. It handles the entire lifecycle
        of the ffmpeg process for a single streaming session and ensures
        it is cleaned up afterward.
        """
        process: subprocess.Popen | None = None
        width = self.width
        height = self.height

        # bytes per pixel mapping for constant-bpp formats
        _bpp_map = {
            "bgr24": 3,
            "rgb24": 3,
            "gray": 1,
            "gray8": 1,
        }
        # For YUYV422, we need to handle it specially since it's 2 bytes per pixel
        if self.requested_fmt == "yuyv422":
            bytes_per_pixel = 2
        else:
            bytes_per_pixel = _bpp_map.get(self.output_pix_fmt, 3)
        frame_size = width * height * bytes_per_pixel

        first_frame = True

        try:
            # 1. Start the FFmpeg subprocess for this specific stream session.
            logger.info(f"Attempting to start new FFmpeg stream for {self.path}...")
            process = (
                ffmpeg.input(
                    self.path,
                    f="v4l2",
                    input_format=self.input_format,
                    video_size=f"{width}x{height}",
                )
                .output("pipe:", format="rawvideo", pix_fmt=self.output_pix_fmt)
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
                if first_frame and in_bytes and len(in_bytes) != frame_size:
                    # auto-detect actual resolution
                    pixels = len(in_bytes) // bytes_per_pixel
                    common = [
                        (160, 120),
                        (320, 240),
                        (640, 480),
                        (1280, 720),
                        (1920, 1080),
                    ]
                    for w, h in common:
                        if w * h == pixels:
                            width, height = w, h
                            frame_size = len(in_bytes)
                            logger.info(f"Auto-detected resolution {width}x{height}")
                            break
                    first_frame = False

                if len(in_bytes) == frame_size:
                    arr = np.frombuffer(in_bytes, np.uint8)

                    if self.output_pix_fmt in ("gray", "gray8"):
                        # Gray → expand to 3-channel
                        arr = arr.reshape((height, width))
                        arr = np.repeat(arr[:, :, None], 3, axis=2)
                    elif self.requested_fmt == "yuyv422":
                        # Convert YUYV422 → BGR using OpenCV if available, else skip frame
                        try:
                            import cv2

                            arr = arr.reshape((height, width, 2))
                            arr = cv2.cvtColor(arr, cv2.COLOR_YUV2BGR_YUY2)
                        except Exception as e:
                            logger.warning(
                                f"OpenCV conversion failed for YUYV422 → BGR: {e}. Skipping frame."
                            )
                            continue
                    else:
                        # Assume 3-byte BGR/RGB already
                        arr = arr.reshape((height, width, 3))

                    yield arr

                first_frame = False

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
