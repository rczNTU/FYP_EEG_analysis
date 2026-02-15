import mne
import numpy as np

VHDR_PATH = "../fyp-test-bv/flicker-fyp(p5-stronger-10trials).vhdr"

print("\nLoading file...")
raw = mne.io.read_raw_brainvision(VHDR_PATH, preload=True, verbose=False)

print("\n==============================")
print("Basic Info")
print("==============================")
print("Sampling rate:", raw.info["sfreq"])
print("Channels:", raw.ch_names)

# =====================================
# EXTRACT TRIALS
# =====================================

events, event_id = mne.events_from_annotations(raw)

p_code = event_id["Comment/p5-stronger"]
stop_code = event_id["Comment/stop"]

sfreq = raw.info["sfreq"]

p_times = events[events[:,2] == p_code][:,0] / sfreq
stop_times = events[events[:,2] == stop_code][:,0] / sfreq

durations = stop_times - p_times

# Remove short trials
valid_idx = durations > 20
p_times = p_times[valid_idx]
stop_times = stop_times[valid_idx]

print("\nValid trials:", len(p_times))

# =====================================
# PREPROCESS
# =====================================

raw.set_eeg_reference("average", verbose=False)
raw.notch_filter(50, verbose=False)
raw.filter(1, 80, verbose=False)

# =====================================
# BUILD FIXED-LENGTH SEGMENTS
# =====================================

ROI = ["O1", "Oz", "O2", "PO3", "PO4"]
BUFFER = 2
FIXED_LEN = 26

segments = []

for i in range(len(p_times)):
    start = p_times[i] + BUFFER
    end = start + FIXED_LEN

    if end < stop_times[i]:
        seg = raw.copy().crop(start, end).pick(ROI)
        segments.append(seg)

print("Segments built:", len(segments))

# =====================================
# PSD PER TRIAL
# =====================================

psds = []

for seg in segments:
    data = seg.get_data()
    data = data.mean(axis=0, keepdims=True)

    psd, freqs = mne.time_frequency.psd_array_multitaper(
        data,
        sfreq=raw.info["sfreq"],
        fmin=1,
        fmax=80,
        verbose=False
    )

    psds.append(psd[0])

psds = np.stack(psds)
mean_psd = psds.mean(axis=0)

# =====================================
# LOCAL SNR
# =====================================

snr = []

for i, f in enumerate(freqs):

    if f < 5 or f > 75:
        snr.append(np.nan)
        continue

    center_power = mean_psd[i]

    mask = (
        (freqs >= f - 3) &
        (freqs <= f + 3) &
        (np.abs(freqs - f) > 1)
    )

    noise_power = mean_psd[mask].mean()
    snr.append(center_power / noise_power)

snr = np.array(snr)

# =====================================
# REMOVE LINE NOISE REGION
# =====================================

exclude = (
    (freqs > 48) & (freqs < 52)  # remove 50 Hz
)

snr_clean = snr.copy()
snr_clean[exclude] = np.nan

# =====================================
# NUMERICAL PEAK RANKING
# =====================================

valid_idx = ~np.isnan(snr_clean)
freqs_valid = freqs[valid_idx]
snr_valid = snr_clean[valid_idx]

top_n = 10
top_indices = np.argsort(snr_valid)[-top_n:][::-1]

print("\n==============================")
print("Top 10 SNR Peaks (excluding 50 Hz)")
print("==============================")

for rank, idx in enumerate(top_indices, 1):
    print(f"{rank}. {freqs_valid[idx]:.2f} Hz  |  SNR = {snr_valid[idx]:.3f}")

# =====================================
# SPECIFICALLY CHECK 40 Hz
# =====================================

idx_40 = np.argmin(np.abs(freqs - 40))
print("\n40 Hz Analysis:")
print("Frequency:", freqs[idx_40])
print("SNR:", snr[idx_40])

# =====================================
# 1/f FIT & RESIDUAL ANALYSIS
# =====================================

import numpy as np
import matplotlib.pyplot as plt

log_freqs = np.log10(freqs)
log_psd = np.log10(mean_psd)

# exclude alpha (8–12) and line noise (48–52)
mask = (
    (freqs > 3) &
    (freqs < 80) &
    ~((freqs > 8) & (freqs < 12)) &
    ~((freqs > 48) & (freqs < 52))
)

# linear fit
coef = np.polyfit(log_freqs[mask], log_psd[mask], 1)
fit_line = np.polyval(coef, log_freqs)

residual = log_psd - fit_line

plt.figure(figsize=(8,4))
plt.plot(freqs, residual)
plt.axhline(0, color="black", linestyle="--")
plt.title("1/f Corrected Residual Spectrum")
plt.xlabel("Frequency (Hz)")
plt.ylabel("Residual (log power)")
plt.xlim(1,80)
plt.show()

# average residual in gamma band
gamma_mask = (freqs >= 35) & (freqs <= 45)
gamma_residual = residual[gamma_mask].mean()

print("\nGamma Residual (35–45 Hz):", gamma_residual)