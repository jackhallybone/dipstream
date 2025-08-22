# dipstream

A thin wrapper for a [sounddevice](https://python-sounddevice.readthedocs.io/en/latest/) `OutputStream` to allow signals to "dip in and out" independently of each other.

## Getting started

```
pip install -e .
```

The following snippet shows how `DipStream` can start and stop playback of several audio signals ("sources") independently of each other.

```python
from dipstream import DipStream

dipstream = DipStream(fs=fs, device=None, channels=[1, 2]) # default stereo device

with dipstream:

    # Add a noise signal on the left channel and a tone signal on the right
    noise = dipstream.add(fs=fs, data=my_noise, channel_mapping=[1]) # left
    tone = dipstream.add(fs=fs, data=my_tone, channel_mapping=[2]) # right

    # Start the noise, looping until stopped, and wait for 3 seconds of playback
    noise.start(loop=True)
    noise.wait_until_start(plus=3)

    # Start the tone and wait until 3 seconds after it has finished playing
    tone.start()
    tone.wait_until_end(plus=3)

    # Stop the noise signal
    noise.stop()

    # Remove the signals from the stream
    dipstream.remove(noise)
    dipstream.remove(tone)
```

## Timing

`DipStream` uses the `sounddevice` stream time as it's clock, accessed using `dipstream.now`.