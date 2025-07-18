# dipstream

A thin wrapper for a [sounddevice](https://python-sounddevice.readthedocs.io/en/latest/) `OutputStream` to allow signals to "dip in and out" independently of each other.

For example, the following will continually loop background noise on the left channel for 3 seconds, then add a 2 second tone on the right channel, then continue the background noise for 3 more seconds:

```python
dipstream = DipStream(fs=fs, device="Realtek ASIO", channels=[1, 2])

with dipstream:
    dipstream.add("noise", fs=fs, data=noise, channel_mapping=[1])
    dipstream.start("noise", loop=True)

    dipstream.wait_until(("noise", dipstream.Event.START), plus=3)  # blocking

    dipstream.add("tone", fs=fs, data=tone, channel_mapping=[2])
    dipstream.start("tone")

    dipstream.wait_until(("tone", dipstream.Event.END), plus=3)  # blocking

    dipstream.stop("noise")
```

For now, sources are added and controlled via string names inside the dipstream instance so that the audio callback can manage their data.

Timings are based on stream time and start/stop actions are calculated in each callback which should keep latency and timing errors low.