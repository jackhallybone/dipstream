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

from dipstream import DipStream


samplerate = 48000

# Create two example signals, 1 second long each
t = np.linspace(0, 1, int(np.ceil(samplerate * 1)), endpoint=False)
tone_500 = 0.75 * np.sin(2 * np.pi * 500 * t)
tone_1000 = 0.25 * np.sin(2 * np.pi * 1000 * t)

dipstream = DipStream(samplerate=samplerate, channels=2)  # use default stereo device

with dipstream:

    # Add the 500Hz tone on the left channel and the 1000Hz tone on the right
    source_500Hz = dipstream.add(
        samplerate=samplerate, data=tone_500, channel_mapping=[1]
    )
    source_1000Hz = dipstream.add(
        samplerate=samplerate, data=tone_1000, channel_mapping=[2]
    )

    # Start the 500Hz tone, looping until stopped, and wait for 2 seconds of playback
    source_500Hz.start(loop=True)
    source_500Hz.wait_until_start(plus=2)  # block until the start event + 2 seconds

    # Start the 1000Hz tone and wait until 2 seconds after it has finished playing
    source_1000Hz.start()
    source_1000Hz.wait_until_end(plus=2)  # block until the end event + 2 seconds

    # Stop the looping 500Hz signal
    source_500Hz.stop()

    # Print and example of the timing
    print(f" 500Hz tone: expected 5s, played for {source_500Hz.playback_duration:.6f}s")
    print(
        f"1000Hz tone: expected 1s, played for {source_1000Hz.playback_duration:.6f}s"
    )

    # Remove the signals from the stream
    dipstream.remove(source_500Hz)
    dipstream.remove(source_1000Hz)
```

## Timing

All timing in `DipStream` uses the [sounddevice stream time](https://python-sounddevice.readthedocs.io/en/0.3.15/api/streams.html#sounddevice.Stream.time). This can be accessed using `dipstream.now`.

For compatibility with ASIO drivers, output block time is taken as the time the callback is fired, rather than the time that that block of audio will play (ie, not the callback `time["outputBufferDacTime"]`).

### Playback timestamps

When a source is started with `start()`, the source indicates to the stream that playback should start. In practice this does not happen immediately, but usually in the next audio block callback which introduces some latency. The `start()` function is blocking until the stream actually starts playback of the source.

The same is true for `stop()`. The `playback_duration` property holds the actual duration the source was playing for.

## API

### `DipStream`

Instantiate using `DipStream(...)`, where the arguments match the [sounddevice OutputStream](https://python-sounddevice.readthedocs.io/en/0.3.15/api/streams.html#sounddevice.OutputStream) arguments.

**Methods:**
- Preferably use the `with` context to manage the lifecycle. Alternatively,
    - `start()` starts the stream.
    - `stop()` stops the stream.
- `add(samplerate, data, channel_mapping)` adds a new audio signal to the stream as a new source with its own data and channel mapping.
    - `channel_mapping` is the mapping between source data channels and output channels and follows the format of the [sounddevice `play()` `mapping` argument](https://python-sounddevice.readthedocs.io/en/0.3.15/api/convenience-functions.html#sounddevice.play).
- `remove(source)` removes a source from the stream.
- `clear_sources()` removes all sources from the stream.
- `elapsed_between(start, end)` calculates the time between two timestamps.
- `wait_until_time(time, sleep)` sleeps in a loop until the target time.

**Properties (readonly):**
- `now` is the current [sounddevice stream time](https://python-sounddevice.readthedocs.io/en/0.3.15/api/streams.html#sounddevice.Stream.time).
- `samplerate` is the stream sample rate.
- `current_blocksize` is the most recent blocksize in the audio callback ([which could vary between callbacks](https://python-sounddevice.readthedocs.io/en/0.3.15/api/streams.html#sounddevice.Stream.blocksize)).

### Sources

Sources should **not** be instantiated directly, only by using the `add()` method the stream.

**Methods:**
- `start(idx, loop)` starts playback of the source, with an optional starting position in the signal and flag to indicate if the signals should loop or stop at the end.
- `stop()` stops playback of the source.
- `wait_until_start(plus)` blocks until playback of the source has started, with optional additional delay after the start time.
- `wait_until_start(plus)` blocks until playback of the source has stopped, with optional additional delay after the stop time.

**Properties (readonly):**
- `samplerate` the sample rate of the source, which must match that of the stream.
- `channel_mapping` the mapping between source data channels and output channels.
- `start_time` the timestamp of the start of playback, which will be None if it has not yet started.
- `end_time` the timestamp of the end of playback, which will be None if it has not yet ended.
- `data_duration` the duration of the audio signal in seconds (ie, duration of the samples).
- `playback_duration` the duration that the source was played back for (ie, between the start and end timestamps).
- `is_playing` is True if the source is currently playing.
- `is_looping` is True if the source is currently playing and set to loop.