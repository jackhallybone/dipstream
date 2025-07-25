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
    dipstream.add("noise", fs, noise, channel_mapping=[1])  # left channel
    dipstream.start("noise", loop=True)

    # Play noise only for 3 seconds (wait option #1: user handles the loop)
    while (
        dipstream.elapsed_between(dipstream.start_time("noise"), dipstream.now)
        < noise_duration_before_tone_s
    ):
        time.sleep(0.005)

    # Add and start a tone on top of the noise
    dipstream.add("tone", fs, tone, channel_mapping=[2])  # right channel
    dipstream.start("tone")

    # Wait untl the tone ends then play noise for 3 seconds more (wait option #2: dipstream handles loop)
    dipstream.wait_until_end("tone", plus=noise_duration_after_tone_s)

    # Stop the noise
    dipstream.stop("noise")

    ## Print some timing info for assessing latency and timing errors
    print(
        "NOISE: start={}, end={}, playback_duration={}s, expected={}s, error={}s".format(
            dipstream.start_time("noise"),
            dipstream.end_time("noise"),
            dipstream.playback_duration("noise"),
            expected_noise_duration_s,
            dipstream.playback_duration("noise") - expected_noise_duration_s,
        )
    )
    print(
        "TONE: start={}, end={}, duration={}s, expected={}s, error={}s".format(
            dipstream.start_time("tone"),
            dipstream.end_time("tone"),
            dipstream.playback_duration("tone"),
            dipstream.data_duration("tone"),  # based on the data itself
            dipstream.playback_duration("tone") - dipstream.data_duration("tone"),
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
