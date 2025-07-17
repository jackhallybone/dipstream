import time

import numpy as np

from dipstream import DipStream


def run_protocol(fs, dipstream):

    # Example mono noise
    noise = np.random.randn(1 * fs, 1)
    noise /= np.max(np.abs(noise))
    noise /= 2

    # Example mono tone
    t = np.linspace(0, 2, int(fs * 2), endpoint=False)
    tone = 0.5 * np.sin(2 * np.pi * 1000 * t)
    tone = tone.reshape(-1, 1)

    # Add and start background noise
    dipstream.add("noise", fs, noise, [1])  # left channel
    dipstream.start("noise", loop=True)

    # Play noise only for 3 seconds (wait option #1: user handles the loop)
    while (
        dipstream.elapsed_between(("noise", dipstream.Event.START), dipstream.now) < 3
    ):
        time.sleep(0.005)

    # Add and start a tone on top of the noise, and wait until it has finished
    dipstream.add("tone", fs, tone, [2])  # right channel
    dipstream.start("tone")
    dipstream.wait_until(("tone", dipstream.Event.END))

    # Continue playing the noise only for 3 seconds after the tone ends (wait option #2: dipstream handles loop)
    dipstream.wait_until(("tone", dipstream.Event.END), plus=3)

    # Stop the noise
    dipstream.stop("noise")

    ## Print some timing info for dev

    noise_start = dipstream.get_event_time(("noise", dipstream.Event.START))
    noise_end = dipstream.get_event_time(("noise", dipstream.Event.END))
    noise_duration_got = dipstream.elapsed_between(
        ("noise", dipstream.Event.START), ("noise", dipstream.Event.END)
    )
    noise_duration_calculated = noise_end - noise_start
    expected_noise_duration = 3 + 2 + 3

    tone_start = dipstream.get_event_time(("tone", dipstream.Event.START))
    tone_end = dipstream.get_event_time(("tone", dipstream.Event.END))
    tone_duration_got = dipstream.elapsed_between(
        ("tone", dipstream.Event.START), ("tone", dipstream.Event.END)
    )
    tone_duration_calculated = tone_end - tone_start
    expected_tone_duration = 2

    print(
        f"NOISE: start={noise_start}, end={noise_end}, dur_got={noise_duration_got}, dur_calc={noise_duration_calculated}, dur_expected={expected_noise_duration}, error={noise_duration_got-expected_noise_duration}"
    )
    print(
        f"TONE: start={tone_start}, end={tone_end}, dur_got={tone_duration_got}, dur_calc={tone_duration_calculated}, dur_expected={expected_tone_duration}, error={tone_duration_got-expected_tone_duration}"
    )
    print("current blocksize", dipstream.current_blocksize)
    print("current block duration", dipstream.current_blocksize / fs)


def main():

    # print(query_devices())

    fs = 48000
    device = "Realtek ASIO"
    channels = [1, 2]

    dipstream = DipStream(fs=fs, device=device, channels=channels)
    with dipstream:
        run_protocol(fs, dipstream)


if __name__ == "__main__":
    main()
