import mne
import numpy as np
from scipy.stats import ttest_ind
import matplotlib.pyplot as plt

VHDR_PATH = "../fyp-test-bv/flicker-fyp(p1-10trials-120hz).vhdr"
ROI = ["O1", "Oz", "O2", "PO3", "PO4"]

FMIN = 1
FMAX = 80

BUFFER = 2  # seconds removed from each boundary
FIXED_DURATION = 26.0

# =====================================
# LOAD
# =====================================

raw = mne.io.read_raw_brainvision(VHDR_PATH, preload=True, verbose=False)
raw.set_eeg_reference("average")
raw.notch_filter(50, verbose=False)
raw.filter(1, 80, verbose=False)

events, event_id = mne.events_from_annotations(raw)

p1_code = event_id['Comment/p1']
stop_code = event_id['Comment/stop']

sfreq = raw.info['sfreq']

p1_times = events[events[:,2] == p1_code][:,0] / sfreq
stop_times = events[events[:,2] == stop_code][:,0] / sfreq
print("\n==============================")
print("MARKER TIMES")
print("==============================")

for i in range(len(p1_times)):
    print(f"Trial {i+1}")
    print("  p1   :", round(p1_times[i], 3), "sec")
    print("  stop :", round(stop_times[i], 3), "sec")
    print("  duration (stop - p1):",
          round(stop_times[i] - p1_times[i], 3), "sec")
# =====================================
# BUILD SEGMENTS
# =====================================

flicker_segments = []
rest_segments = []

print("\n==============================")
print("FLICKER + REST SEGMENTS")
print("==============================")

for i in range(len(p1_times)):

    flick_start = p1_times[i] + BUFFER
    flick_end   = stop_times[i] - BUFFER

    print(f"\nTrial {i+1}")
    print("  Flicker raw start:", round(p1_times[i], 3))
    print("  Flicker raw end  :", round(stop_times[i], 3))
    print("  Buffered start   :", round(flick_start, 3))
    print("  Buffered end     :", round(flick_end, 3))
    print("  Flicker duration :", round(flick_end - flick_start, 3))

    if flick_end > flick_start:
        fixed_end = flick_start + FIXED_DURATION
        flicker_segments.append(
            raw.copy().crop(flick_start, fixed_end).pick(ROI)
        )

    # REST (only if not last trial)
    if i < len(p1_times) - 1:

        rest_start = stop_times[i] + BUFFER
        rest_end   = p1_times[i+1] - BUFFER

        print("  Rest raw start   :", round(stop_times[i], 3))
        print("  Rest raw end     :", round(p1_times[i+1], 3))
        print("  Rest buffered start:", round(rest_start, 3))
        print("  Rest buffered end  :", round(rest_end, 3))
        print("  Rest duration    :", round(rest_end - rest_start, 3))

        if rest_end > rest_start:
            rest_segments.append(
                raw.copy().crop(rest_start, rest_end).pick(ROI)
            )
from mne.time_frequency import psd_array_multitaper

print("\n==============================")
print("FLICKER-ONLY PSD")
print("==============================")

flick_psds = []

for idx, seg in enumerate(flicker_segments):

    data = seg.get_data()  # shape: (channels, samples)

    psd, freqs = psd_array_multitaper(
        data,
        sfreq=seg.info["sfreq"],
        fmin=1,
        fmax=80,
        adaptive=True,
        normalization="full",
        verbose=False
    )

    psd = psd.mean(axis=0)  # average across channels

    flick_psds.append(psd)

    print(f"Trial {idx+1} PSD shape:", psd.shape)

flick_psds = np.stack(flick_psds)
# Convert to log10 power
log_psds = np.log10(flick_psds)
mean_log_psd = log_psds.mean(axis=0)
print("\n==============================")
print("LOCAL SNR ANALYSIS")
print("==============================")

signal_width = 0.5
noise_gap = 1.0
noise_width = 3.0

snr_trials = []

for trial_psd in flick_psds:

    snr = []

    for i, f in enumerate(freqs):

        # define masks
        signal_mask = (
            (freqs >= f - signal_width) &
            (freqs <= f + signal_width)
        )

        noise_mask = (
            (freqs >= f - noise_width) &
            (freqs <= f + noise_width) &
            ~(
                (freqs >= f - noise_gap) &
                (freqs <= f + noise_gap)
            )
        )

        if np.any(noise_mask):
            signal_power = trial_psd[signal_mask].mean()
            noise_power = trial_psd[noise_mask].mean()
            snr.append(signal_power / noise_power)
        else:
            snr.append(np.nan)

    snr_trials.append(snr)

snr_trials = np.array(snr_trials)
mean_snr = np.nanmean(snr_trials, axis=0)
plt.figure(figsize=(10,5))
plt.plot(freqs, mean_snr)
plt.xlabel("Frequency (Hz)")
plt.ylabel("Local SNR")
plt.title("Local SNR Across Frequencies (Flicker Only)")
plt.show()
# plt.figure(figsize=(8,5))

# for trial in log_psds:
#     plt.plot(freqs[mask], trial[mask], alpha=0.4)

# plt.plot(freqs[mask], mean_log_psd[mask], linewidth=3)

# plt.xlabel("Frequency (Hz)")
# plt.ylabel("Log10 Power")
# plt.title("Zoomed 35–45 Hz Region")
# plt.show()
