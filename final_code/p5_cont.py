import mne
import numpy as np
import matplotlib.pyplot as plt

# ==========================================
# SETTINGS
# ==========================================

VHDR_PATH = "../fyp-test-bv/flicker-fyp(p5-stronger-continuous-article-120Hz).vhdr"

TARGET_FREQ = 40  # Hz
BUFFER = 2        # seconds after trigger
ROI = ["O1", "Oz", "O2", "PO3", "PO4"]

# ==========================================
# LOAD FILE
# ==========================================

print("\nLoading file...")
raw = mne.io.read_raw_brainvision(VHDR_PATH, preload=True, verbose=False)

sfreq = raw.info["sfreq"]
print("Sampling rate:", sfreq)

# ==========================================
# EXTRACT EVENTS
# ==========================================

events, event_id = mne.events_from_annotations(raw)

p_code = event_id["Comment/p5-stronger"]
stop_code = event_id["Comment/stop"]

p_times = events[events[:,2] == p_code][:,0] / sfreq
stop_times = events[events[:,2] == stop_code][:,0] / sfreq

durations = stop_times - p_times
print("Valid trials:", len(durations))

# ==========================================
# PREPROCESS
# ==========================================

raw.set_eeg_reference("average", verbose=False)
raw.notch_filter(50, verbose=False)
raw.filter(1, 100, verbose=False)

# ==========================================
# INTEGER CYCLE WINDOWING
# ==========================================

# Determine max usable duration across trials
usable_durations = durations - BUFFER
min_duration = np.min(usable_durations)

# Convert duration to integer number of 40 Hz cycles
cycles = int(np.floor(min_duration * TARGET_FREQ))
window_length = cycles / TARGET_FREQ

print("\nUsing:")
print("Cycles:", cycles)
print("Window length (sec):", window_length)

segments = []

for i in range(len(p_times)):

    start = p_times[i] + BUFFER
    end = start + window_length

    if end <= stop_times[i]:
        seg = raw.copy().crop(start, end).pick(ROI)
        segments.append(seg.get_data().mean(axis=0))

segments = np.stack(segments)

print("Segments built:", segments.shape)

# ==========================================
# AVERAGE TIME-DOMAIN SIGNAL
# ==========================================

mean_signal = segments.mean(axis=0)

# Remove DC offset
mean_signal = mean_signal - np.mean(mean_signal)

# ==========================================
# FFT
# ==========================================

n = len(mean_signal)
freqs = np.fft.rfftfreq(n, d=1/sfreq)
fft_vals = np.fft.rfft(mean_signal)

amplitude = np.abs(fft_vals) / n
power = amplitude ** 2
log_power = np.log10(power + 1e-20)

# Limit frequency range
mask = (freqs >= 1) & (freqs <= 80)
freqs = freqs[mask]
power = power[mask]
log_power = log_power[mask]

# ==========================================
# PLOT
# ==========================================

plt.figure(figsize=(8,4))
plt.plot(freqs, log_power)
plt.title("Log Power Spectrum (Integer-Cycle FFT)")
plt.xlabel("Frequency (Hz)")
plt.ylabel("Log10 Power")
plt.xlim(1, 80)
plt.show()

# ==========================================
# 40 Hz SNR
# ==========================================

idx_40 = np.argmin(np.abs(freqs - TARGET_FREQ))
f_40 = freqs[idx_40]

center_power = power[idx_40]

# local noise ±3 Hz excluding ±1 Hz
noise_mask = (
    (freqs >= f_40 - 3) &
    (freqs <= f_40 + 3) &
    (np.abs(freqs - f_40) > 1)
)

noise_power = power[noise_mask].mean()
snr_40 = center_power / noise_power

print("\n40 Hz Analysis")
print("Frequency:", f_40)
print("SNR:", snr_40)

# ==========================================
# BROADBAND GAMMA (35–45 Hz)
# ==========================================

gamma_mask = (freqs >= 35) & (freqs <= 45)

gamma_mean = log_power[gamma_mask].mean()
global_mean = log_power.mean()

print("\nBroadband Gamma (35–45 Hz)")
print("Mean gamma power:", gamma_mean)
print("Global mean power:", global_mean)
print("Gamma relative difference:", gamma_mean - global_mean)

# ==========================================
# OPTIONAL: TOP 10 PEAKS (excluding 50 Hz)
# ==========================================

snr_list = []

for i, f in enumerate(freqs):

    if f < 5 or f > 75:
        snr_list.append(np.nan)
        continue

    center = power[i]
    mask_noise = (
        (freqs >= f - 3) &
        (freqs <= f + 3) &
        (np.abs(freqs - f) > 1)
    )

    noise = power[mask_noise].mean()
    snr_list.append(center / noise)

snr_array = np.array(snr_list)

# remove 50 Hz
snr_array[np.abs(freqs - 50) < 1] = np.nan

top_idx = np.argsort(snr_array)[-10:][::-1]
# ==========================================
# TOP FREQUENCIES (BASED ON POWER)
# ==========================================

# Exclude 50 Hz line noise ±1 Hz
valid_mask = (freqs >= 5) & (freqs <= 75) & (np.abs(freqs - 50) > 1)

valid_freqs = freqs[valid_mask]
valid_power = power[valid_mask]

top_idx = np.argsort(valid_power)[-15:][::-1]

print("\nTop 15 Power Peaks (excluding 50 Hz)")
print("=====================================")

for i in top_idx:
    print(f"{valid_freqs[i]:.3f} Hz  |  Power = {valid_power[i]:.6e}")