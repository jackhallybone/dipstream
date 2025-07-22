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

    def __init__(self, name, fs, data, channel_mapping):
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

    def start(self, starting_idx=0, loop=False):
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
    def get_n_frames_from_data(data, read_idx, n_frames, loop):
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

    def get_next_block(self, n_frames, block_time, fs):
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

    class Event(str, Enum):
        """Types of events a source can generate."""

        START = "start"
        END = "end"

    def __init__(self, fs, device, channels):
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
        self._current_blocksize = None

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
    def now(self):
        """Get the current stream time."""
        return self._stream.time

    @property
    def fs(self):
        """Get the sample rate of the stream."""
        return self._stream.samplerate

    @property
    def current_blocksize(self):
        """Get the blocksize of the stream (gets the current/latest value of variable streams)."""
        with self._lock:
            return self._current_blocksize

    # Source

    def _mix_and_map_sources(self, n_frames, n_channels, block_time, fs):
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

    def __contains__(self, name):
        """Allow lookup of stream names directly from the instance."""
        with self._lock:
            return name in self._sources

    def add(self, name, fs, data, channel_mapping, replace=False):
        """Add a source to the collection (which starts off inactive).

        NOTE: mono signals can be mapped to multiple channels
        """
        if fs != self.fs:
            raise ValueError(f"Sample rate mismatch for source '{name}'.")
        if data.ndim != 2:
            raise ValueError(
                f"Data shape must be (n_samples, n_channels), even for mono signals."
            )
        if not all(1 <= ch <= self._stream.channels for ch in channel_mapping):
            raise ValueError(f"Invalid channel mapping for source '{name}'.")
        if data.shape[1] > 1 and data.shape[1] != len(channel_mapping):
            raise ValueError(f"Incorrect number of channels for source '{name}'.")

        with self._lock:
            if name in self._sources and not replace:
                raise ValueError(f"Source '{name}' already exists.")
            self._sources[name] = _Source(name, fs, data, channel_mapping)

    def remove(self, name):
        """Remove a source from the collection."""
        with self._lock:
            if name in self._sources:
                del self._sources[name]

    def clear_sources(self):
        """Remove all sources from the collection."""
        with self._lock:
            self._sources = {}

    def _get_source_by_name(self, name):
        """Lookup a source by name. Must be called inside a lock."""
        source = self._sources.get(name)
        if source is None:
            raise KeyError(f"No source named '{name}'")
        return source

    def start(self, name, loop=False):
        """Start playback of a source."""
        with self._lock:
            source = self._get_source_by_name(name)
            source.start(loop=loop)
        # Wait for the actual start (which should happen in the next callback)
        if not source.start_event.wait(
            timeout=0.5
        ):  # TODO: timeout based on next callback time?
            raise RuntimeError(f"Source '{name}' did not start on time")

    def stop(self, name):
        """Stop playback of a source."""
        with self._lock:
            source = self._get_source_by_name(name)
            source.stop()
        # Wait for the actual stop (which should happen in the next callback)
        if not source.end_event.wait(
            timeout=0.5
        ):  # TODO: timeout based on next callback time?
            raise RuntimeError(f"Source '{name}' did not stop on time")

    # TODO: consider adding play() which calls add() then start()

    def duration(self, name):
        """Get the duration of a source in seconds."""
        with self._lock:
            source = self._get_source_by_name(name)
            return source.duration

    # Timing

    def _get_time_of_event(self, source, event_name):
        """Must be called within a lock."""
        if event_name == self.Event.START:
            return source.start_time
        elif event_name == self.Event.END:
            return source.end_time
        else:
            raise ValueError(f"Unknown event name ({event_name})")

    def _wait_until_event(self, source, event_name, timeout=None):
        """Must not be called within a lock."""
        if event_name == self.Event.START:
            return source.start_event.wait(timeout=timeout)
        elif event_name == self.Event.END:
            return source.end_event.wait(timeout=timeout)
        else:
            raise ValueError(f"Unknown event tag ({event_name})")

    def get_event_time(self, event):
        """Get the stream time of an event (eg, source start or stop), or pass a time through."""
        if isinstance(event, tuple):
            name, event_name = event
            with self._lock:
                source = self._get_source_by_name(name)
                return self._get_time_of_event(source, event_name)
        return event

    def wait_until(self, event, plus=0, timeout=None, sleep_in_loop=0.005):
        """Wait (and/or sleep) until a given event has fired or time has passed."""
        if isinstance(event, tuple):
            name, event_name = event
            with self._lock:
                source = self._get_source_by_name(name)
            fired = self._wait_until_event(source, event_name, timeout)
            if not fired:
                raise RuntimeError(f"Event '{event}' did not fire in time")
            with self._lock:
                target_time = self._get_time_of_event(source, event_name)
        else:
            target_time = event

        target_time += plus
        while self.now < target_time:
            time.sleep(sleep_in_loop)

    def elapsed_between(self, start, end):
        """Calculate the elapsed stream time between two events or times."""
        return self.get_event_time(end) - self.get_event_time(start)
