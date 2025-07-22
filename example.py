import time

import numpy as np

from dipstream import DipStream, query_devices


def audio_sequence(fs, dipstream):

    noise_duration_before_tone_s = 3
    tone_duration_s = 2
    noise_duration_after_tone_s = 3
    expected_noise_duration_s = noise_duration_before_tone_s+tone_duration_s+noise_duration_after_tone_s

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
        dipstream.elapsed_between(("noise", dipstream.Event.START), dipstream.now) < noise_duration_before_tone_s
    ):
        time.sleep(0.005)

    # Add and start a tone on top of the noise, and wait until it has finished
    dipstream.add("tone", fs, tone, channel_mapping=[2])  # right channel
    dipstream.start("tone")
    dipstream.wait_until(("tone", dipstream.Event.END))

    # Continue playing the noise only for 3 seconds after the tone ends (wait option #2: dipstream handles loop)
    dipstream.wait_until(("tone", dipstream.Event.END), plus=noise_duration_after_tone_s)

    # Stop the noise
    dipstream.stop("noise")

    ## Print some timing info for assessing latency and timing errors
    print("NOISE: start={}, end={}, duration={}s, expected={}s, error={}s".format(
        dipstream.get_event_time(("noise", dipstream.Event.START)),
        dipstream.get_event_time(("noise", dipstream.Event.END)),
        dipstream.elapsed_between(
            ("noise", dipstream.Event.START), ("noise", dipstream.Event.END)
        ),
        expected_noise_duration_s,
        dipstream.elapsed_between(
            ("noise", dipstream.Event.START), ("noise", dipstream.Event.END)
        ) - expected_noise_duration_s
    ))
    print("TONE: start={}, end={}, duration={}s, expected={}s, error={}s".format(
        dipstream.get_event_time(("tone", dipstream.Event.START)),
        dipstream.get_event_time(("tone", dipstream.Event.END)),
        dipstream.elapsed_between(
            ("tone", dipstream.Event.START), ("tone", dipstream.Event.END)
        ),
        tone_duration_s,
        dipstream.elapsed_between(
            ("tone", dipstream.Event.START), ("tone", dipstream.Event.END)
        ) - tone_duration_s
    ))
    print(f"Current blocksize = {dipstream.current_blocksize}")
    print(f"Current block duration = {dipstream.current_blocksize / fs}s")


def main():

    # print(query_devices())

    fs = 48000
    device = None # use the default output device
    channels = [1, 2]

    dipstream = DipStream(fs=fs, device=device, channels=channels)
    with dipstream:
        audio_sequence(fs, dipstream)


if __name__ == "__main__":
    main()
