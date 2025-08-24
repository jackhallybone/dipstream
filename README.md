# dipstream

A thin wrapper for a [sounddevice](https://python-sounddevice.readthedocs.io/en/latest/) `OutputStream` to allow signals to "dip in and out" independently of each other.

A `DipStream` stream allows output audio sources to be added, played and removed at any time while the stream is running. This means that it is not necessary to pre-compute the complete output mix signal, and allows sources to start and stop in response to undefined timing triggers like user input.

## Getting started

```
pip install -e .
```

The following snippet shows how `DipStream` can start and stop playback of several audio signals ("sources") independently of each other.

```python
import numpy as np

from dipstream import DipStream


fs = 48000

# Create two example signals
t = np.linspace(0, 1, int(np.ceil(fs * 1)), endpoint=False)
tone_500 = 0.75 * np.sin(2 * np.pi * 500 * t)[:, np.newaxis]
tone_1000 = 0.25 * np.sin(2 * np.pi * 1000 * t)[:, np.newaxis]

dipstream = DipStream(fs=fs, device=None, channels=[1, 2])  # default stereo device

with dipstream:

    # Add a noise signal on the left channel and a tone signal on the right
    noise = dipstream.add(fs=fs, data=tone_500, channel_mapping=[1])  # left
    tone = dipstream.add(fs=fs, data=tone_1000, channel_mapping=[2])  # right

    # Start the noise, looping until stopped, and wait for 2 seconds of playback
    noise.start(loop=True)
    noise.wait_until_start(plus=2)

    # Start the tone and wait until 2 seconds after it has finished playing
    tone.start()
    tone.wait_until_end(plus=2)

    # Stop the noise signal
    noise.stop()

    # Remove the signals from the stream
    dipstream.remove(noise)
    dipstream.remove(tone)
```

## Timing

All timing in `DipStream` uses the `sounddevice` [stream time](https://python-sounddevice.readthedocs.io/en/0.3.15/api/streams.html). This can be accessed using `dipstream.now`.

For compatibility with ASIO drivers, output block time is taken as the time the callback is fired, rather than the time that that block of audio will play (ie, `stream.time` not `time["outputBufferDacTime"]`).

### Playback timestamps

The source `start()` and `stop()` functions do not directly start or stop playback themselves. Actual playback is handled in the stream callback. Therefore, there will be some latency between the function and the audio (in theory, the action takes effect in the next callback). The actual source playback times can be accessed from the source properties `start_time` and `end_time`. This is why source `data_duration` may not equal `playback_duration`.

## API

### `DipStream`

Methods:
- Preferably use the `with` context to manage the lifecycle. Alternatively,
    - `start()` starts the stream, using a [sounddevice stream](https://python-sounddevice.readthedocs.io/en/0.3.15/api/streams.html).
    - `stop()` stops the stream.
- `add(fs, data, channel_mapping)` adds a new audio signal to the stream as a new source with its own data and channel mapping.
    - *`channel_mapping` is indexed from 1 to match sounddevice and audio hardware.*
- `remove(source)` removes a source from the stream.
- `clear_sources()` removes all sources from the stream.
- `elapsed_between(start, end)` calculates the time between two timestamps.
- `wait_until_time(time, sleep)` sleeps in a loop until the target time.

Properties (readonly):
- `now` is the current [sounddevice stream](https://python-sounddevice.readthedocs.io/en/0.3.15/api/streams.html) time.
- `fs` is the stream sample rate.
- `current_blocksize` is the block size of the audio callback (of the most recent callback for [variable streams](https://python-sounddevice.readthedocs.io/en/0.3.15/api/streams.html)).

### Sources

Sources should not be created directly, but with `DipStream.add()`.

Methods:
- `start(starting_idx, loop)` schedules playback to start (usually in the next callback), with optional starting position in the signal, and if the signal should loop or end.
- `stop()` schedules playback to stop (usually in the next callback).
- `wait_until_start(plus)` waits (blocking) until playback has actually started, with optional additional delay after the start, `plus`.
- `wait_until_start(plus)` waits (blocking) until playback has actually stopped, with optional additional delay after the end, `plus`.

Properties (readonly):
- `fs` the sample rate of the source (which must match that of the stream).
- `channel_mapping` the channel mapping.
- `start_time` the timestamp of the start of playback, which will be None if it has not yet started.
- `end_time` the timestamp of the end of playback, which will be None if it has not yet ended.
- `data_duration` the duration of the audio signal in seconds (ie, based on length and sample rate).
- `playback_duration` the duration that the source was played back for (ie, between the start and end timestamps).
- `is_playing` is True if the source is currently playing.
- `is_looping` is True if the source is currently playing and set to loop.