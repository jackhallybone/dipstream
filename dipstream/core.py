from __future__ import annotations

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

    def __init__(self, **kwargs):
        """Initialise a Dipstream instance, validating and setting up the underlying sounddevice.OutputStream."""

        if "callback" in kwargs:
            raise ValueError(
                "Cannot accept a callback. The callback is defined by DipStream."
            )

        sd.check_output_settings(
            samplerate=kwargs.get("samplerate"),
            device=kwargs.get("device"),
            channels=kwargs.get("channels"),
        )

        self._lock = threading.Lock()

        self._sources: set[_Source] = set()  # to track the sources
        self._stream = sd.OutputStream(
            **kwargs,
            callback=self._callback,
        )
        self._current_blocksize = 0

        # Preallocate an oversized output block
        self._outdata_mix = np.zeros((16384, self.channels))

    # Stream

    def _callback(self, outdata, frames, time, status):
        """Trigger the getting of the next block of audio from all sources and pass it to the sound device."""
        with self._lock:
            self._current_blocksize = frames

        # Zero the output arrays
        outdata[:] = 0
        self._outdata_mix[:] = 0

        # Estimate the time that the block will hit the DAC (time.outputBufferDacTime is not provided by ASIO)
        block_time = self.now + self.latency

        sources = list(self._sources)  # shallow copy snapshot of the current sources

        for source in sources:

            with source._lock:
                # If the source is not playing, skip it
                if not source._playing:
                    # But first, mark the source as ended if it is not already (ie, `stop()` has just been called)
                    if source._end_time is None:
                        source._end_time = block_time
                        source._end_event.set()
                    continue

                # Mark the source as started if it is not already (ie, `start()` has just been called)
                if source._start_time is None:
                    source._start_time = block_time
                    source._start_event.set()

            # Mix a full block of frames from the source's data into outdata, if there are enough frames left
            if source._read_idx + frames < source._data.shape[0]:
                self._outdata_mix[
                    :frames, source._channel_mapping_indices
                ] += source._data[source._read_idx : source._read_idx + frames, :]
                source._read_idx += frames

            else:
                # Else, mix all remaining frames from the source's data into outdata
                frames_remaining = source._data.shape[0] - source._read_idx
                self._outdata_mix[
                    :frames_remaining, source._channel_mapping_indices
                ] += source._data[
                    source._read_idx : source._read_idx + frames_remaining, :
                ]

                # If the source should loop, complete the block by mixing the outstanding frames from the data into outdata
                if source._loop:
                    frames_outstanding = frames - frames_remaining
                    self._outdata_mix[
                        frames_remaining : frames_remaining + frames_outstanding,
                        source._channel_mapping_indices,
                    ] += source._data[:frames_outstanding, :]
                    source._read_idx = frames_outstanding

                # Else, mark the source as not playing back and ended
                else:
                    end_time = block_time + (frames_remaining / self.samplerate)
                    with source._lock:
                        source._playing = False
                        source._end_time = end_time
                        source._end_event.set()

        if np.max(self._outdata_mix) > 1.0 or np.min(self._outdata_mix) < -1.0:
            raise ValueError(
                f"Output block contains amplitude values above 1 ({np.max(np.abs(self._outdata_mix))})"
            )

        outdata[:] = self._outdata_mix[:frames, :]

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
    def samplerate(self) -> int:
        """Get the sample rate of the stream."""
        return self._stream.samplerate

    @property
    def channels(self) -> int:
        """Get the number of channels in the stream."""
        return self._stream.channels

    @property
    def latency(self) -> float:
        """Get the latency of the system."""
        return self._stream.latency

    @property
    def current_blocksize(self) -> int:
        """Get the most recent blocksize of the stream (fixes or variable streams)."""
        with self._lock:
            return self._current_blocksize

    # Sources

    def add(
        self, samplerate: int, data: np.ndarray, channel_mapping: list[int]
    ) -> "_Source":
        """Add a source to the collection and validate the signal and playback properties."""

        if samplerate != self.samplerate:
            raise ValueError("Source sample rate does match stream sample rate.")

        if min(channel_mapping) < 1 or max(channel_mapping) > self.channels:
            raise ValueError(
                f"Channel numbers in mapping must be between 1 and {self.channels}"
            )

        if len(channel_mapping) != len(set(channel_mapping)):
            raise ValueError("Channel numbers cannot be repeated in mapping.")

        if not np.issubdtype(data.dtype, np.floating):
            raise TypeError("Audio signal must be of type/subtype float.")

        # Internally expand mono (n,) shape to (n, 1) to allow multichannel (repeated) broadcasting
        if data.ndim == 1:
            data = data[:, np.newaxis]

        if data.shape[1] > data.shape[0]:
            raise ValueError("Audio signal must be of shape (n_samples, n_channels)")

        if data.shape[1] > 1 and data.shape[1] != len(channel_mapping):
            raise ValueError(
                "Number of channels in channel_mapping must equal the number of channels in data unless it is mono."
            )

        source = _Source(self, samplerate, data, channel_mapping)

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
        samplerate: int,
        data: np.ndarray,
        channel_mapping: list[int],
    ):
        """Initialise a _Source instance, which begins inactive."""

        self._lock = threading.Lock()

        self._dipstream = dipstream
        self._samplerate = samplerate
        self._data = data
        self._channel_mapping = channel_mapping

        # To match sounddevice, mapping starts with 1, so convert to numpy index
        self._channel_mapping_indices = np.array(self._channel_mapping) - 1

        self._playing = False
        self._loop = False
        self._read_idx = 0
        self._start_time: float | None = None
        self._start_event = threading.Event()
        self._end_time: float | None = None
        self._end_event = threading.Event()

    def _must_exist_in_stream(self):
        """Raise an error if this source does not exist in the Dipstream stream collection."""
        if self not in self._dipstream:  # Dipstream __contains__ handles the lock
            raise RuntimeError(
                "This source does not exist in Dipstream. Has it already been removed?"
            )

    @property
    def samplerate(self):
        """The source's sample rate."""
        return self._samplerate

    @property
    def data_duration(self):
        """The duration of the source's signal in seconds."""
        return self._data.shape[0] / self._samplerate

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
            return self._playing and self._loop

    def start(
        self,
        at: float | None = None,
        start_with: _Source | None = None,
        with_offset: float | None = None,
        loop: bool = False,
        starting_idx: int = 0,
        timeout: float | None = 0.5,
    ):
        """Start source playback; immediately, at a given time, or with another source. Blocks until the start hits the DAC."""

        if at is not None and (start_with is not None or with_offset is not None):
            raise ValueError(
                "Cannot schedule to start both at a time and with another source."
            )

        if start_with == self:
            raise ValueError("Cannot schedule to start with self.")

        self._must_exist_in_stream()

        if at is not None:
            # Wait for the scheduled starting time, accounting for starting latency
            self._dipstream.wait_until_time(at - self._dipstream.latency)
        elif start_with is not None:
            # Wait for the start event of the other source (when it's start is scheduled)
            start_with._start_event.wait()
            if with_offset:
                # Wait for a further offset after the other source has been scheduled to start
                self._dipstream.wait_until_time(self._dipstream.now + with_offset)

        # Set flags to schedule the source to start playing on the stream
        self._start_event.clear()
        self._end_event.clear()
        with self._lock:
            self._playing = True
            self._loop = loop
            self._read_idx = starting_idx
            self._start_time = None
            self._end_time = None

        # Wait for the start to actually be scheduled in the callback
        if not self._start_event.wait(timeout=timeout):
            raise TimeoutError(f"Source did not start on time (after {timeout}s).")

        # Wait for the start to actually hit the DAC (ie, wait for latency)
        with self._lock:
            start_time = self._start_time  # includes the latency of the system
        self._dipstream.wait_until_time(start_time)

    def stop(
        self,
        at: float | None = None,
        stop_with: _Source | None = None,
        with_offset: float | None = None,
        timeout: float | None = 0.5,
    ):
        """Stop source playback; immediately, at a given time, or with another source. Blocks until the end hits the DAC."""

        if at is not None and (stop_with is not None or with_offset is not None):
            raise ValueError(
                "Cannot schedule a stop both at a time and with another source."
            )

        self._must_exist_in_stream()

        if at is not None:
            # Wait for the scheduled stopping time, accounting for stopping latency
            self._dipstream.wait_until_time(at - self._dipstream.latency)
        elif stop_with is not None:
            # Wait for the end event of the other source (when it's stop is scheduled)
            stop_with._end_event.wait()
            if with_offset:
                # Wait for a further offset after the other source has been scheduled to stop
                self._dipstream.wait_until_time(self._dipstream.now + with_offset)

        # Set flags to schedule the source to stop playing on the stream
        with self._lock:
            self._playing = False
            self._loop = False

        # Wait for the stop to actually be scheduled in the callback
        if not self._end_event.wait(timeout=timeout):
            raise TimeoutError(f"Source did not stop on time (after {timeout}s).")

        # Wait for the stop to actually hit the DAC (ie, wait for latency)
        with self._lock:
            end_time = self._end_time  # includes the latency of the system
        self._dipstream.wait_until_time(end_time)
