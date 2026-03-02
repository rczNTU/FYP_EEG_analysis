import mne
import numpy as np
import matplotlib.pyplot as plt

# ===============================
# CONFIG
# ===============================

# VHDR_PATH = "../fri_ptn6_20hz/flicker - p6_20Hz flicker vs baseline.vhdr"
VHDR_PATH = "../fri_ptn6/flicker - p6_60Hz vs baseline.vhdr"

FMIN = 1
FMAX = 80

# ===============================
# LOAD RAW
# ===============================

print("\n[INFO] Loading EEG...")
raw = mne.io.read_raw_brainvision(VHDR_PATH, preload=True, verbose=False)

# Fix misc channels
misc = [ch for ch in raw.ch_names if raw.get_channel_types(picks=ch)[0] == "misc"]
if misc:
    raw.set_channel_types({ch: "eeg" for ch in misc})

raw.set_montage("standard_1020", on_missing="ignore")
raw.set_eeg_reference("average")

raw.notch_filter(50, verbose=False)
raw.filter(1, 80, verbose=False)

print("\n[INFO] Channels:", raw.ch_names)

# ===============================
# 1️⃣ VARIANCE CHECK
# ===============================

data = raw.get_data(picks="eeg")
channel_var = np.var(data, axis=1)

print("\n==============================")
print("Channel Variance")
print("==============================")

for ch, var in zip(raw.ch_names, channel_var):
    print(f"{ch:6s} | Variance: {var:.2e}")

mean_var = np.mean(channel_var)
std_var = np.std(channel_var)

print("\nMean variance:", mean_var)
print("Std variance :", std_var)

# Flag extreme channels
print("\nPossible Noisy Channels (>2 std):")
for ch, var in zip(raw.ch_names, channel_var):
    if var > mean_var + 2 * std_var:
        print("⚠️", ch)

print("\nPossible Flat Channels (<0.5x mean):")
for ch, var in zip(raw.ch_names, channel_var):
    if var < 0.5 * mean_var:
        print("⚠️", ch)

# ===============================
# 2️⃣ PSD PER CHANNEL
# ===============================

print("\n[INFO] Computing PSD per channel...")

psd = raw.compute_psd(method="welch", fmin=FMIN, fmax=FMAX, verbose=False)
psds, freqs = psd.get_data(return_freqs=True)

mean_psd = psds

plt.figure(figsize=(10, 6))

for ch_idx in range(len(raw.ch_names)):
    plt.plot(freqs, 10 * np.log10(mean_psd[ch_idx] + 1e-30),
             label=raw.ch_names[ch_idx], alpha=0.5)

plt.xlabel("Frequency (Hz)")
plt.ylabel("Power (dB)")
plt.title("PSD per Channel")
plt.xlim(FMIN, FMAX)
plt.legend(fontsize=8)
plt.tight_layout()
plt.show()

print("\n[INFO] Visual inspection complete.")