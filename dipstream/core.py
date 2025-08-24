import os
import threading
import time

import numpy as np

os.environ["SD_ENABLE_ASIO"] = "1"

import sounddevice as sd


def query_devices():
    """Use sounddevice.query_devices() to list the available audio devices."""
    return sd.query_devices()


class DipStream:
    """A multi-source stream controller using sounddevice and _Source objects."""

    def __init__(
        self, fs: int, device: str | None, channels: list[int], blocksize: int = 0
    ):
        """Initialise a Dipstream instance, validating and setting up the underlying sounddevice.OutputStream."""

        self._lock = threading.Lock()

        sd.check_output_settings(samplerate=fs, device=device, channels=max(channels))

        self._sources: set[_Source] = set()  # track the sources in the stream
        self._stream = sd.OutputStream(
            samplerate=fs,
            device=device,
            channels=max(channels),
            blocksize=blocksize,
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

        mixed_block = self._mix_and_map_sources(
            frames,
            self._stream.channels,
            self.now,
            self.fs,  # time["outputBufferDacTime"] is not provided by ASIO
        )

        if mixed_block.shape != outdata.shape:
            raise ValueError(
                f"Output block shape error ({mixed_block.shape} should be {outdata.shape})."
            )

        if np.any(np.abs(mixed_block) > 1):
            raise ValueError(
                f"Output block contains amplitude values above 1 ({np.max(np.abs(mixed_block))})"
            )

        outdata[:] = mixed_block

    def __enter__(self):
        """Forward the enter function to the sounddevice stream. Implies self.start_stream()."""
        self._stream.__enter__()
        return self

    def __exit__(self, exc_type, exc_value, traceback):
        """Forward the enter function to the sounddevice stream. Implies self.stop_stream()."""
        self._stream.__exit__(exc_type, exc_value, traceback)

    def start(self):
        """Forward the start function to the sounddevice stream."""
        self._stream.start()

    def stop(self):
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
        """Get the most recent blocksize of the stream (fixes or variable streams)."""
        with self._lock:
            return self._current_blocksize

    # Sources

    def add(self, fs: int, data: np.ndarray, channel_mapping: list[int]) -> "_Source":
        """Add a source to the collection and validate the signal and playback properties."""
        if fs != self.fs:
            raise ValueError("Source sample rate does match stream sample rate.")
        if data.ndim != 2:
            raise ValueError(
                "Audio signal must always be of shape (n_samples, n_channels)."
            )
        if min(channel_mapping) < 1 or max(channel_mapping) > self._stream.channels:
            raise ValueError(
                f"Channel numbers in mapping must be between 1 and {self._stream.channels}"
            )
        if len(channel_mapping) != len(set(channel_mapping)):
            raise ValueError("Channel numbers cannot be repeated in mapping.")
        if data.shape[1] > 1 and data.shape[1] != len(channel_mapping):
            raise ValueError(
                "Number of channels mapped must equal the number of channels in the audio signal."
            )

        source = _Source(self, fs, data, channel_mapping)

        with self._lock:
            self._sources.add(source)

        return source

    def __contains__(self, source: "_Source") -> bool:
        with self._lock:
            return source in self._sources

    def remove(self, source: "_Source"):
        """Remove a source from the stream."""
        source._must_exist_in_stream()
        with self._lock, source._lock:
            if source._playing:
                raise RuntimeError("Cannot remove a source while it is playing.")
            self._sources.remove(source)

    def clear_sources(self):
        """Remove all sources from the stream."""
        with self._lock:
            self._sources.clear()

    def _mix_and_map_sources(
        self, n_frames: int, n_channels: int, block_time: float, fs: int
    ) -> np.ndarray:
        """Mix the output of all sources, mapped to channels, into one output block."""

        mix = np.zeros((n_frames, n_channels))

        with self._lock:
            sources = list(
                self._sources
            )  # shallow snapshot of set, sources handle their internal locking

        # Accumulate blocks of audio from the sources that are currently playing
        for source in sources:
            # _get_next_block also handles the actual starting and stopping of sources
            block = source._get_next_block(n_frames, block_time, fs)
            if block is None:
                continue

            if block.shape[1] == 1:
                # Mono sources are repeated on each output channel
                mix[:, source._channel_mapping_indices] += block
            else:
                # Multichannel sources map source channel to output channel
                mix[:, source._channel_mapping_indices] += block[:, :]

        return mix

    # Timing

    def elapsed_between(self, start: float, end: float) -> float:
        """Calculate the elapsed time between two times."""
        return end - start

    def wait_until_time(self, target_time: float, sleep_in_loop: float = 0.005):
        """Wait in a loop until a specified time, with optional additional delay."""
        while self.now < target_time:
            time.sleep(sleep_in_loop)


class _Source:
    """A source for the Dipstream multi-source stream controller."""

    def __init__(
        self,
        dipstream: "DipStream",
        fs: int,
        data: np.ndarray,
        channel_mapping: list[int],
    ):
        """Initialise a _Source instance, which begins inactive."""

        self._lock = threading.Lock()

        self._dipstream = dipstream
        self._fs = fs
        self._data = data
        self._channel_mapping = channel_mapping

        # To match sounddevice, mapping starts with 1, so convert to numpy index
        self._channel_mapping_indices = np.array(self._channel_mapping) - 1

        self._playing = False
        self._looping = False
        self._read_idx = 0
        self._start_time = None
        self._start_event = threading.Event()
        self._end_time = None
        self._end_event = threading.Event()

    def _must_exist_in_stream(self):
        """Raise an error if this source does not exist in the Dipstream stream collection."""
        if self not in self._dipstream:  # Dipstream __contains__ handles the lock
            raise RuntimeError(
                "This source does not exist in Dipstream. Has it already been removed?"
            )

    @property
    def fs(self):
        """The source's sample rate."""
        return self._fs

    @property
    def data_duration(self):
        """The duration of the source's signal in seconds."""
        return self._data.shape[0] / self._fs

    @property
    def playback_duration(self):
        """The actual duration the source was played back for in seconds."""
        with self._lock:
            if self._start_time is None or self._end_time is None:
                return None
            return self._end_time - self._start_time

    @property
    def channel_mapping(self):
        """The source's channel mapping."""
        return list(self._channel_mapping)

    @property
    def start_time(self):
        """The time that playback of the source started, in stream time."""
        with self._lock:
            return self._start_time

    @property
    def end_time(self):
        """The time that playback of the source ended, in stream time."""
        with self._lock:
            return self._end_time

    @property
    def is_playing(self):
        """True if the source is currently playing."""
        with self._lock:
            return self._playing

    @property
    def is_looping(self):
        """True if the source is currently playing and playback is set to loop."""
        with self._lock:
            return self._playing and self._looping

    def start(
        self, starting_idx: int = 0, loop: bool = False, timeout: float | None = 0.5
    ):
        """Start source playback.

        Sets flags to indicate that the source should playback when the stream requests it.
        """

        self._must_exist_in_stream()

        # Set flags to schedule the source to start playing on the stream
        self._start_event.clear()
        self._end_event.clear()
        with self._lock:
            self._playing = True
            self._looping = loop
            self._read_idx = starting_idx
            self._start_time = None
            self._end_time = None

        # Wait for the actual start to happen
        if not self._start_event.wait(timeout=timeout):
            raise RuntimeError(f"Source did not start on time (after {timeout}s)")

    def stop(self, timeout: float | None = None):
        """Stop source playback.

        Sets flags to indicate that the source should not playback when the stream requests it.
        """
        self._must_exist_in_stream()

        # Set flags to schedule the source to stop playing on the stream
        with self._lock:
            self._playing = False
            self._looping = False

        # Wait for the actual stop to happen
        if not self._end_event.wait(timeout=timeout):
            raise RuntimeError(f"Source did not stop on time (after {timeout}s).")

    def wait_until_start(self, plus: float = 0, timeout: float | None = None):
        """Wait until the source has started playback, with optional additional further wait time."""

        self._must_exist_in_stream()

        with self._lock:
            if not self._playing:
                raise RuntimeError(
                    "Cannot wait for a source start when start() has not been called."
                )

        self._start_event.wait(timeout=timeout)

        if plus:
            self._dipstream.wait_until_time(self._start_time + plus)

    def wait_until_end(self, plus: float = 0, timeout: float | None = None):
        """Wait until the source has completed playback, with optional additional further wait time."""

        self._must_exist_in_stream()

        with self._lock:
            if not self._playing:
                raise RuntimeError(
                    "Cannot wait for a source to end when it has not yet started."
                )
            if self._looping:
                raise RuntimeError("Cannot wait until a looping source has ended.")

        self._end_event.wait(timeout=timeout)

        if plus:
            self._dipstream.wait_until_time(self._end_time + plus)

    @staticmethod
    def _get_n_frames_from_data(
        data: np.ndarray, read_idx: int, n_frames: int, wrap: bool
    ) -> tuple[np.ndarray, int, int]:
        """Get the next n_frames from data, padding with zeros or wrapping the signal at the end of the buffer."""

        data_n_frames, data_n_channels = data.shape
        outdata = np.zeros((n_frames, data_n_channels))
        n_frames_from_end = min(data_n_frames - read_idx, n_frames)

        # Get as many frames as from the signal in data possible
        outdata[:n_frames_from_end, :] = data[
            read_idx : read_idx + n_frames_from_end, :
        ]
        new_read_idx = read_idx + n_frames_from_end
        n_frames_with_data = n_frames_from_end

        # If the signal in data has ended, but outdata is not complete
        if (
            n_frames_from_end < n_frames
        ):  # NOTE: if len(data) divisible by n_frames, end will fire next callback

            # Complete the signal with frames from the start of the signal in data
            if wrap:
                n_frames_from_start = n_frames - n_frames_from_end
                outdata[n_frames_from_end:, :] = data[:n_frames_from_start, :]
                new_read_idx = n_frames_from_start
                n_frames_with_data = n_frames

            # Mark the signal as ended
            else:
                new_read_idx = data_n_frames
                n_frames_with_data = n_frames_from_end

        return outdata, new_read_idx, n_frames_with_data

    def _get_next_block(
        self, n_frames: int, block_time: float, fs: int
    ) -> np.ndarray | None:
        """Get the next block of audio data, or return None if not currently playing."""

        with self._lock:

            # If not playing but no end time, stop() has been called so set the end and return None
            if not self._playing:
                if self._end_time is None:
                    self._end_time = block_time
                    self._end_event.set()
                return None

            # If playing, but no start time, start() has been called so set the start
            if self._start_time is None:
                self._start_time = block_time
                self._start_event.set()

            # Copy within lock
            read_idx = self._read_idx
            looping = self._looping

        # Get the next frame of audio
        block, new_read_idx, n_frames_with_data = self._get_n_frames_from_data(
            self._data, read_idx, n_frames, looping
        )

        with self._lock:
            self._read_idx = new_read_idx

            # If the signal has ended, update source attributes to indicate a stop/end
            if n_frames_with_data < n_frames:
                self._playing = False
                if self._end_time is None:
                    self._end_time = block_time + (n_frames_with_data / fs)
                    self._end_event.set()

        return block
