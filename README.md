# dipstream

A thin wrapper for a [sounddevice](https://python-sounddevice.readthedocs.io/en/latest/) stream to allow signals to "dip in and out" independently of each other.

A `DipStream` stream, is an [`OutputStream`](https://python-sounddevice.readthedocs.io/en/0.3.15/api/streams.html#sounddevice.OutputStream) that allows audio sources to be added, played and removed at any time while the stream is running. This means that it is not necessary to pre-compute the complete mixed output signal, and allows sources to start and stop in response to undefined timing triggers like user input.

## Getting started

```
pip install -e .
```

The following snippet shows how `DipStream` can start and stop playback of several audio signals ("sources") independently of each other.

```python
import numpy as np

import dipstream as ds


samplerate = 48000

# Create two example signals, both 1s long
t = np.linspace(0, 1, int(np.ceil(samplerate * 1)), endpoint=False)
tone_500 = 0.75 * np.sin(2 * np.pi * 500 * t)
tone_1000 = 0.25 * np.sin(2 * np.pi * 1000 * t)

dipstream = ds.DipStream(samplerate=samplerate, channels=2)  # use default stereo device

with dipstream:

    # Add the 500Hz tone on the left channel and the 1000Hz tone on the right
    source_500Hz = dipstream.add(
        samplerate=samplerate, data=tone_500, channel_mapping=[1]
    )
    source_1000Hz = dipstream.add(
        samplerate=samplerate, data=tone_1000, channel_mapping=[2]
    )

    # Start the 500Hz tone, looping until stopped
    source_500Hz.start(loop=True)

    # After 2 seconds of playback, start the 1000Hz tone
    source_1000Hz.start(at=source_500Hz.start_time + 2)

    # Stop the 500Hz tone 2 seconds after the 1000Hz tone ends
    source_500Hz.stop(stop_with=source_1000Hz, with_offset=2)

    # Print and example of the timing
    print(f" 500Hz: expected 5s, played for {source_500Hz.playback_duration:.6f}s")
    print(f"1000Hz: expected 1s, played for {source_1000Hz.playback_duration:.6f}s")

    # Remove the signals from the stream
    dipstream.remove(source_500Hz)
    dipstream.remove(source_1000Hz)
```

### Output Mix

Each active source is mixed (+=) into the output on the channels specified. It is necessary to **make sure the output mix does not clip** (exceed the -1:+1 range) by controlling the amplitude of each source. Currently clipping will throw and error and end the stream.

The output channels which sources are mixed into and played back on can be set using the `channel_mapping` argument. This does not need to be continuous, for example a stereo source could be mapped to `[1, 5]`. Mono sources can be mapped to multiple channels, and their data will be reproduced on each one.

## Timing

All timing in `DipStream` uses the [sounddevice stream time](https://python-sounddevice.readthedocs.io/en/0.3.15/api/streams.html#sounddevice.Stream.time) clock. A timestamp can be taken using `dipstream.now`.

### Source Start and Stop

Source `start()` and `stop()` functions schedule the start or stop to be handled in the next audio callback (after any delays using the `at` or `..._with` arguments, see their API section). Additionally, the latency of the system means that there will be a delay between scheduling and the actual change on the audio output.

For consistency, the source `start_time` and `stop_time` properties use the estimated time that change is seen on the audio output. This is their scheduled time (callback time) + the latency of the system ([`dipstream.latency`](https://python-sounddevice.readthedocs.io/en/0.3.12/api.html?highlight=latency#sounddevice.Stream.latency)).

The `start()` and `stop()` functions block until change is estimated on the output. When called without any delay arguments, it should block a little longer than the latency (`dipstream.latency`).

When called with the `at` time argument, the delay time is reduced by the latency in an attempt to schedule the actual change as close to the target time as possible. When called with the `..._with` argument, the calling source should be scheduled in the callback after its target so the delta should be one callback duration. In these cases, the blocking should be only a little longer than the intentional delay.

Consequently, if the start and stop are scheduled correctly, the error in `playback_duration` compared to the expected duration should be kept to a minimum -- in theory a maximum of 1 block duration, although the latency correction is not definite.

### Other Events

Since the effect of latency is accounted for, it should be possible to compare source timing events with other events, such as [user input](https://github.com/jackhallybone/quick-tk-gui), if they are timestamped using the `dipstream.now` clock.

## API

### `DipStream`

Instantiate using `DipStream(...)`, where the arguments match the [sounddevice OutputStream](https://python-sounddevice.readthedocs.io/en/0.3.15/api/streams.html#sounddevice.OutputStream) arguments. For example `DipStream(samplerate=48000, device="ASIO Fireface", channels=8)`.

**Methods:**
- Preferably use the `with` context to manage the lifecycle. Alternatively,
    - `start()` starts the stream.
    - `stop()` stops the stream.
- `add(samplerate: int, data: np.ndarray, channel_mapping: list[int])` adds a new audio signal to the stream and returns the source instance.
    - `data` must be of type/subtype float and shape (n_samples,) or (n_samples, n_channels).
    - `channel_mapping` is the mapping between source data channels and output channels and follows the format of the [sounddevice `play()` `mapping` argument](https://python-sounddevice.readthedocs.io/en/0.3.15/api/convenience-functions.html#sounddevice.play). Mono audio can be mapped (repeated on) multiple channels.
- `remove(source: _Source)` removes a source from the stream.
- `clear_sources()` removes all sources from the stream.
- `elapsed_between(start: float, end: float)` calculates the time between two timestamps.
- `wait_until_time(time: float, sleep: float)` sleeps in a loop until the target time.

**Properties (readonly):**
- `now: float` is the current [sounddevice stream time](https://python-sounddevice.readthedocs.io/en/0.3.15/api/streams.html#sounddevice.Stream.time).
- `samplerate: int` is the stream sample rate.
- `channels: int` is the number of channels available in the stream.
- `latency: float` is the [latency of the stream](https://python-sounddevice.readthedocs.io/en/0.3.12/api.html?highlight=latency#sounddevice.Stream.latency) in seconds
- `current_blocksize: int` is the blocksize of the audio callback ([which could vary between callbacks](https://python-sounddevice.readthedocs.io/en/0.3.15/api/streams.html#sounddevice.Stream.blocksize)).

### Sources

Sources should **not** be instantiated directly, only by using the `add()` method the stream.

**Methods:**
- `start(at: float | None, starts_with: _Source | None, with_offset: float | None, loop: bool, starting_idx: int, timeout: float)` starts the playback of a source, which can `loop` and start from a `starting_idx` in the signal. Schedules the start immediately or delays to start `at` a time, or `start_with` the start of another source plus an option `with_offset`. See the **Timing** section for more information.
- `stop(at: float | None, starts_with: _Source | None, with_offset: float | None, timeout: float)` stops the playback of a source. Schedules the stop immediately or delays to stop `at` a time, or `stop_with` the stop of another source plus an option `with_offset`. See the **Timing** section for more information.

**Properties (readonly):**
- `samplerate: int` the sample rate of the source, which must match that of the stream.
- `channel_mapping: list[int]` the mapping between source data channels and output channels.
- `start_time: float` the timestamp of the start of playback, which will be None if it has not yet started.
- `end_time: float` the timestamp of the end of playback, which will be None if it has not yet ended.
- `data_duration: float` the duration of the audio signal in seconds (ie, duration of the samples).
- `playback_duration: float` the duration that the source was played back for (ie, between the start and end timestamps).
- `is_playing: bool` is True if the source is currently playing.
- `is_looping: bool` is True if the source is currently playing and set to loop.
