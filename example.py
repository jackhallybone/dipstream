import time

import numpy as np

from dipstream import DipStream, query_devices


def audio_sequence(fs, dipstream):

    noise_duration_before_tone_s = 3
    tone_duration_s = 2
    noise_duration_after_tone_s = 3
    expected_noise_duration_s = (
        noise_duration_before_tone_s + tone_duration_s + noise_duration_after_tone_s
    )

    # Example mono noise
    noise = np.random.randn(1 * fs, 1)
    noise /= np.max(np.abs(noise))
    noise /= 2

    # Example mono tone
    t = np.linspace(0, tone_duration_s, int(fs * tone_duration_s), endpoint=False)
    tone = 0.25 * np.sin(2 * np.pi * 1000 * t)
    tone = tone.reshape(-1, 1)

    # Add and start background noise
    noise = dipstream.add(fs, noise, channel_mapping=[1])  # left channel
    noise.start(loop=True)

    # Play noise only for 3 seconds (wait option #1: user handles the loop)
    while (
        dipstream.elapsed_between(noise.start_time, dipstream.now)
        < noise_duration_before_tone_s
    ):
        time.sleep(0.005)

    # Add and start a tone on top of the noise
    tone = dipstream.add(fs, tone, channel_mapping=[2])  # right channel
    tone.start()

    # Wait until the tone ends then play noise for 3 seconds more (wait option #2: dipstream handles loop)
    tone.wait_until_end(plus=noise_duration_after_tone_s)

    # Stop the noise
    noise.stop()

    dipstream.remove(noise)
    dipstream.remove(tone)

    ## Print some timing info for assessing latency and timing errors
    print(
        "NOISE: start={}, end={}, playback_duration={}s, expected={}s, error={}s".format(
            noise.start_time,
            noise.end_time,
            noise.playback_duration,
            expected_noise_duration_s,
            noise.playback_duration - expected_noise_duration_s,
        )
    )
    print(
        "TONE: start={}, end={}, duration={}s, expected={}s, error={}s".format(
            tone.start_time,
            tone.end_time,
            tone.playback_duration,
            tone.data_duration,  # based on the data itself
            tone.playback_duration - tone.data_duration,
        )
    )
    print(f"Current blocksize = {dipstream.current_blocksize}")
    print(f"Current block duration = {dipstream.current_blocksize / dipstream.fs}s")
    print(f"Sample duration = {1/ dipstream.fs:.8f}s")


def main():

    # print(query_devices())

    fs = 48000
    device = None  # use the default output device
    channels = [1, 2]

    dipstream = DipStream(fs=fs, device=device, channels=channels)
    with dipstream:
        audio_sequence(fs, dipstream)


if __name__ == "__main__":
    main()