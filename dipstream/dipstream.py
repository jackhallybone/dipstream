import os
import threading
import time
from enum import Enum

import numpy as np

os.environ["SD_ENABLE_ASIO"] = "1"

import sounddevice as sd


def query_devices():
    return sd.query_devices()


class _Source:

    def __init__(
        self, name: str, fs: int, data: np.ndarray, channel_mapping: list[int]
    ):
        self.name = name
        self.fs = fs
        self.data = data
        self.channel_mapping = channel_mapping

        self.duration = data.shape[0] / fs

        self.playing = False
        self.loop = False
        self.read_idx = 0
        self.start_time = None
        self.start_event = threading.Event()
        self.end_time = None
        self.end_event = threading.Event()

    def start(self, starting_idx: int = 0, loop: bool = False):
        """Set flags to indicate the source should output data when requested.

        NOTE: start_time is recorded when playback actually starts, not when this function is called.
        """
        self.playing = True
        self.loop = loop
        self.read_idx = starting_idx
        self.start_time = None
        self.start_event.clear()
        self.end_time = None
        self.end_event.clear()

    def stop(self):
        """Set flags to indicate the source should not output data when requested.

        NOTE: end_time is recorded when playback actually stops, not when this function is called.
        """
        self.playing = False

    @staticmethod
    def get_n_frames_from_data(
        data: np.ndarray, read_idx: int, n_frames: int, loop: bool
    ):
        """Get the next n_frames from data, wrapping at the end if looping or padding if ended early."""

        # Case 1: full block available
        if read_idx + n_frames <= len(data):
            block = data[read_idx : read_idx + n_frames]
            next_read_idx = read_idx + n_frames
            return block, next_read_idx, n_frames

        # Case 2: looping
        if loop:
            part1 = data[read_idx:]  # remaining frames from the end
            part2 = data[: n_frames - len(part1)]  # frames from the start
            block = np.concatenate((part1, part2))
            next_read_idx = (read_idx + n_frames) % len(data)
            return block, next_read_idx, n_frames

        # Case 3: paddings
        part1 = data[read_idx:]
        pad = np.zeros((n_frames - len(part1), data.shape[1]))
        block = np.concatenate((part1, pad))
        next_read_idx = len(data)  # indicates the end
        return block, len(data), len(part1)

    def get_next_block(self, n_frames: int, block_time: float, fs: int):
        """Get the next block of audio data, or return None if not currently playing."""
        if not self.playing:
            if self.end_time is None:
                self.end_time = block_time
                self.end_event.set()
            return None

        if self.start_time is None:
            self.start_time = block_time
            self.start_event.set()

        block, self.read_idx, n_frames_with_data = self.get_n_frames_from_data(
            self.data, self.read_idx, n_frames, self.loop
        )

        if not self.loop and self.read_idx >= len(self.data):
            self.playing = False
            if self.end_time is None:
                self.end_time = block_time + (n_frames_with_data / fs)
                self.end_event.set()

        return block


class DipStream:
    """A multi-source audio stream manager for soundevice.

    NOTE: times are relative to the sounddevice stream clock.
    """

    def __init__(self, fs: int, device: str | None, channels: list[int]):
        self._lock = threading.Lock()

        # Check that the sounddevice settings are valid
        sd.check_output_settings(samplerate=fs, device=device, channels=max(channels))

        self._sources = {}
        self._stream = sd.OutputStream(
            samplerate=fs,
            device=device,
            channels=max(channels),
            blocksize=0,  # use default or optimal blocksize provided by the device
            callback=self._callback,
        )
        self._current_blocksize = 0

    # Stream

    def _callback(self, outdata, frames, time, status):
        """Trigger the getting of the next block of audio from all sources and pass it to the sound device."""
        if status:
            print(f"Stream status: {status}")

        with self._lock:
            self._current_blocksize = frames

        # NOTE: it would be better to use time["outputBufferDacTime"] but it is not provided by ASIO
        mix_block = self._mix_and_map_sources(
            frames, self._stream.channels, self.now, self.fs
        )

        if mix_block.shape != outdata.shape:
            raise ValueError(
                f"Output block shape error ({mix_block.shape} should be {outdata.shape})."
            )

        if np.any(np.abs(mix_block) > 1):
            raise ValueError(
                f"Output block contains amplitude values above 1 ({np.max(np.abs(mix_block))})"
            )

        outdata[:] = mix_block

    def __enter__(self):
        """Forward the enter function to the sounddevice stream. Implies self.start_stream()."""
        self._stream.__enter__()
        return self

    def __exit__(self, exc_type, exc_value, traceback):
        """Forward the enter function to the sounddevice stream. Implies self.stop_stream()."""
        self._stream.__exit__(exc_type, exc_value, traceback)

    def start_stream(self):
        """Forward the start function to the sounddevice stream."""
        self._stream.start()

    def stop_stream(self):
        """Forward the stop function to the sounddevice stream."""
        self._stream.stop()

    @property
    def now(self) -> float:
        """Get the current stream time."""
        return self._stream.time

    @property
    def fs(self) -> int:
        """Get the sample rate of the stream."""
        return self._stream.samplerate

    @property
    def current_blocksize(self) -> int:
        """Get the blocksize of the stream (gets the current/latest value of variable streams)."""
        with self._lock:
            return self._current_blocksize

    # Source

    def _mix_and_map_sources(
        self, n_frames: int, n_channels: int, block_time: float, fs: int
    ) -> np.ndarray:
        """Mix the output of all sources, mapped to channels, into one output block."""
        with self._lock:
            mix = np.zeros((n_frames, n_channels))
            for source in self._sources.values():
                src_block = source.get_next_block(n_frames, block_time, fs)
                # Pass if source is not playing (and therefore returns None)
                if src_block is None:
                    continue
                # Map source channels to output channels (allowing mono signals to be mapped to multiple channels)
                is_mono = src_block.shape[1] == 1
                for i, out_ch in enumerate(source.channel_mapping):
                    src_ch = 0 if is_mono else i
                    mix[:, out_ch - 1] += src_block[:, src_ch]
        return mix

    def __contains__(self, name: str) -> bool:
        """Allow lookup of stream names directly from the instance."""
        with self._lock:
            return name in self._sources

    def add(
        self,
        name: str,
        fs: int,
        data: np.ndarray,
        channel_mapping: list[int],
        replace: bool = False,
    ) -> str:
        """Add a source to the collection (which starts off inactive).

        NOTE: mono audio can be mapped to multiple channels
        """
        if fs != self.fs:
            raise ValueError(f"Sample rate mismatch for source '{name}'.")
        if data.ndim != 2:
            raise ValueError(
                f"Data shape must be (n_samples, n_channels), even for mono audio."
            )
        if not all(1 <= ch <= self._stream.channels for ch in channel_mapping):
            raise ValueError(f"Invalid channel mapping for source '{name}'.")
        if data.shape[1] > 1 and data.shape[1] != len(channel_mapping):
            raise ValueError(f"Incorrect number of channels for source '{name}'.")

        with self._lock:
            if name in self._sources and not replace:
                raise ValueError(f"Source '{name}' already exists.")
            self._sources[name] = _Source(name, fs, data, channel_mapping)

        return name

    def remove(self, name: str):
        """Remove a source from the collection."""
        with self._lock:
            if name in self._sources:
                del self._sources[name]

    def clear_sources(self):
        """Remove all sources from the collection."""
        with self._lock:
            self._sources = {}

    def _get_source_by_name(self, name: str) -> _Source:
        """Lookup a source by name. Must be called inside a lock."""
        source = self._sources.get(name)
        if source is None:
            raise KeyError(f"No source named '{name}'")
        return source

    def start(self, name: str, loop: bool = False, timeout: float | None = 0.5):
        """Start playback of a source."""
        with self._lock:
            source = self._get_source_by_name(name)
            source.start(loop=loop)
        # Wait for the actual start (which should happen in the next callback)
        if not source.start_event.wait(timeout=timeout):
            raise RuntimeError(
                f"Source '{name}' did not start on time (after {timeout}s)"
            )

    def stop(self, name: str, timeout: float | None = 0.5):
        """Stop playback of a source."""
        with self._lock:
            source = self._get_source_by_name(name)
            source.stop()
        # Wait for the actual stop (which should happen in the next callback)
        if not source.end_event.wait(timeout=timeout):
            raise RuntimeError(
                f"Source '{name}' did not stop on time (after {timeout}s)"
            )

    # Timing

    def start_time(self, name: str) -> float | None:
        """Get the start time of a source. This will be None if it hasn't started yet."""
        with self._lock:
            source = self._get_source_by_name(name)
            return source.start_time

    def end_time(self, name: str) -> float | None:
        """Get the end time of a source. This will be None if it hasn't started yet."""
        with self._lock:
            source = self._get_source_by_name(name)
            return source.end_time

    def elapsed_between(self, start: float, end: float) -> float:
        """Calculate the elapsed time between two times."""
        return end - start

    def data_duration(self, name: str) -> float:
        """Get the duration of a source's audio data in seconds."""
        with self._lock:
            source = self._get_source_by_name(name)
            return source.duration

    def playback_duration(self, name: str) -> float | None:
        """Get the duration of a source's playback in seconds"""
        start = self.start_time(name)
        end = self.end_time(name)
        if start is None or end is None:
            return None
        return self.elapsed_between(start, end)

    def wait_until_time(
        self, target_time: float, plus: float = 0, sleep_in_loop: float = 0.005
    ):
        """Wait in a loop until a specified time, with optional additional delay."""
        target_time += plus
        while self.now < target_time:
            time.sleep(sleep_in_loop)

    def wait_until_start(
        self,
        name: str,
        plus: float = 0,
        timeout: float | None = None,
        sleep_in_loop: float = 0.005,
    ):
        """Wait until the start event has fired, with optional additional delay."""
        with self._lock:
            source = self._get_source_by_name(name)
        fired = source.start_event.wait(timeout=timeout)
        if not fired:
            raise RuntimeError(
                f"Source '{name}' start event did not fire in time (after {timeout}s)"
            )
        start_time = self.start_time(name)
        if start_time is None:
            raise ValueError(f"Source '{name}' has not started.")
        self.wait_until_time(start_time, plus, sleep_in_loop)

    def wait_until_end(
        self,
        name: str,
        plus: float = 0,
        timeout: float | None = None,
        sleep_in_loop: float = 0.005,
    ):
        """Wait until the end event has fired, with optional additional delay."""
        with self._lock:
            source = self._get_source_by_name(name)
        fired = source.end_event.wait(timeout=timeout)
        if not fired:
            raise RuntimeError(
                f"Source '{name}' end event did not fire in time (after {timeout}s)"
            )
        end_time = self.end_time(name)
        if end_time is None:
            raise ValueError(f"Source '{name}' has not ended.")
        self.wait_until_time(end_time, plus, sleep_in_loop)
