# dipstream

A thin wrapper for a [sounddevice](https://python-sounddevice.readthedocs.io/en/latest/) `OutputStream` to allow signals to "dip in and out" independently of each other.

## Useage

```
pip install -e .
```

```python
from dipstream import DipStream

dipstream = DipStream(fs=fs, device=None, channels=[1, 2]) # default stereo device

with dipstream

    # Add and start a noise signal that will loop until stopped
    dipstream.add("noise", fs=fs, data=my_noise, channel_mapping=[1]) # left
    dipstream.start("noise", loop=True)

    # Wait for the noise to start then play for 3 seconds
    dipstream.wait_until_start("noise", plus=3)

    # Add and start a tone signal that will play until it ends
    dipstream.add("tone", fs=fs, data=my_tone, channel_mapping=[2]) # right
    dipstream.start("tone")

    # Wait for the tone to end, then play the noise for 3 more seconds
    dipstream.wait_until_end("tone", plus=3)

    # Stop the noise signal
    dipstream.stop("noise")
```

Sources are accessed by a string (`name`). To avoid repetition of a string in the code, this is returned by the `add()` function and can be used to address the source later.

```python
tone_name = dipstream.add("tone", ...)
dipstream.start(tone_name)
...
```

## API

### Stream

Preferably use

```python
with dipstream:
    # do something
```

or
```python
dipstream.start_stream()
# do something
dipstream.stop_stream()
```


### Adding and removing sources

- `add(name, fs, data, channel_mapping, replace)`: add a new source to the stream
    - `name: str`: a name to access the source by
    - `fs: int`: sampling rate of the audio data which is checked against the stream rate
    - `data: np.ndarray`: mono or multichannel audio data in the shape `(n_samples, n_channels)`
    - `channel_mapping: list[int]`: a list of channel numbers to play the source back on
        - (the number of channels must match n_channels in the audio data unless the source is mono in which case it is repeated on the specified channels)
    - `replace: bool`: allow or disallow the new source to overwrite an existing source with the same name
- `remove(name: str)`: remove the named source from the stream
- `clear_sources()`: remove all sources from the stream

### Controlling playback

- `start(name: str)`: start the playback of the named source
- `stop(name: str)`: stop the playback of the named source

These functions will block for a very short period until the playback of the source is actually started or stopped (usually in the next stream callback some milliseconds later).

### Timings

All times are based on the `sounddevice` stream clock.

#### Event times

- `now`: get the current stream time
- `start_time(name: str)`: get the start time of a source, which is None if the source has not yet started
- `end_time(name: str)`: get the end time of a source, which is None if the source has not yet ended

#### Durations

- `elapsed_between(start: float, end: float)`: get the elapsed time between two times
- `data_duration(name: str)`: get the duration of a source's audio data
- `playback_duration(name: str)`: get the duration that a source was actually playing for

#### Waiting

The user can manually handle waiting for playback, or the following functions can be used to wait for events and time delays.

- `wait_until_time(target_time: float, plus: float)`: wait until the target time, with an optional additional delay `plus`
- `wait_until_start(name: str, plus: float)`: wait until a source has started with an optional delay `plus`
- `wait_until_end(name: str, plus: float)`: wait until a source has ended with an optional delay `plus`