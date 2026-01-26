import mne
import numpy as np
import matplotlib.pyplot as plt
from scipy.signal import detrend
from collections import Counter
from scipy.signal import hilbert

def compute_plv_roi(epochs, f0, roi_chs):
    bw = 1.0
    epochs_filt = epochs.copy().filter(
        f0 - bw, f0 + bw, fir_design="firwin", verbose=False
    )

    data = epochs_filt.get_data()
    ch_names = epochs_filt.ch_names
    plvs = []

    for ch in roi_chs:
        if ch not in ch_names:
            continue

        ch_idx = ch_names.index(ch)
        phases = []

        for ep in data:
            analytic = hilbert(ep[ch_idx])
            phases.append(np.angle(analytic))

        phases = np.array(phases)
        plv = np.abs(np.mean(np.exp(1j * phases), axis=0)).mean()
        plvs.append(plv)

    return np.mean(plvs), plvs

def local_snr(psd, freqs, f0, band=3.0):
    df = freqs[1] - freqs[0]
    signal = psd[np.argmin(np.abs(freqs - f0))]

    noise_mask = (
        (freqs >= f0 - band) &
        (freqs <= f0 + band) &
        (np.abs(freqs - f0) > df)
    )
    return signal / psd[noise_mask].mean()

# CONFIG — MATCH 240 Hz PIPELINE
VHDR_PATH = "./60hz_3_patterns/single-flicker-fyp.vhdr"

# Filtering (RAW LEVEL — IMPORTANT)
FMIN, FMAX = 1, 80
NOTCH = 50

# PSD
PSD_FMIN, PSD_FMAX = 10, 60

# Gamma (EXCLUDE LINE NOISE EDGE)
GAMMA_BAND = (30, 45)

# Expected entrainment band
EXPECTED_BAND = (38.5, 41.5)

# Epoching
EDGE_START = 0.4
EDGE_END   = 0.4
EPOCH_LEN = 2.8
EPOCH_OVERLAP = 0.0

# ROI — identical intent to 240 Hz script
ROI_CHS = ["O1", "Oz", "O2", "PO3", "POz", "PO4"]

DROP_CHS = ["Cz"]

#==================================
# HELPERS (IDENTICAL LOGIC)
#==================================
def make_fixed_epochs(raw, t0, t1, duration, overlap):
    seg = raw.copy().crop(tmin=t0, tmax=t1, include_tmax=False)
    if seg.times[-1] <= duration:
        return None

    ep = mne.make_fixed_length_epochs(
        seg,
        duration=duration,
        overlap=overlap,
        preload=True,
        verbose=False
    )

    ep._data = detrend(ep.get_data(), axis=-1, type="linear")
    return ep

#==================================
# LOAD RAW
#==================================
raw = mne.io.read_raw_brainvision(VHDR_PATH, preload=True, verbose=False)
sfreq = raw.info["sfreq"]

misc_chs = [ch for ch in raw.ch_names if ch.startswith("na")]
raw.set_channel_types({ch: "misc" for ch in misc_chs})
raw.info["bads"] = misc_chs + DROP_CHS

print("Channel types:", Counter(raw.get_channel_types()))
print("Bad channels :", raw.info["bads"])

raw.set_montage(
    mne.channels.make_standard_montage("standard_1020"),
    on_missing="ignore"
)

#==================================
# RAW-LEVEL FILTERING (MATCHES 240 Hz)
#==================================
raw.filter(FMIN, FMAX, verbose=False)
raw.notch_filter(NOTCH, verbose=False)
raw.set_eeg_reference("average", verbose=False)

#==================================
# EVENTS
#==================================
events, event_id = mne.events_from_annotations(raw)

baseline_segments = []
pattern_segments  = []

for i in range(1, len(events)):
    if events[i, 2] != event_id["Comment/stop"]:
        continue

    prev = events[i - 1, 2]
    t0 = events[i - 1, 0] / sfreq + EDGE_START
    t1 = events[i, 0] / sfreq - EDGE_END

    if (t1 - t0) < EPOCH_LEN:
        continue

    if prev == event_id["Comment/baseline"]:
        baseline_segments.append((t0, t1))
    elif prev == event_id["Comment/pattern_3"]:
        pattern_segments.append((t0, t1))

#==================================
# EPOCHING
#==================================
epochs_base = mne.concatenate_epochs([
    make_fixed_epochs(raw, t0, t1, EPOCH_LEN, EPOCH_OVERLAP)
    for t0, t1 in baseline_segments
    if make_fixed_epochs(raw, t0, t1, EPOCH_LEN, EPOCH_OVERLAP)
]).pick("eeg")

epochs_pat = mne.concatenate_epochs([
    make_fixed_epochs(raw, t0, t1, EPOCH_LEN, EPOCH_OVERLAP)
    for t0, t1 in pattern_segments
    if make_fixed_epochs(raw, t0, t1, EPOCH_LEN, EPOCH_OVERLAP)
]).pick("eeg")

epochs_base.drop_bad()
epochs_pat.drop_bad()

print(f"Baseline epochs: {len(epochs_base)}")
print(f"Pattern 3 epochs: {len(epochs_pat)}")

#==================================
# PSD (IDENTICAL TO 240 Hz)
#==================================
n_times = epochs_base.get_data().shape[-1]

psd_base = epochs_base.compute_psd(
    method="welch",
    fmin=PSD_FMIN,
    fmax=PSD_FMAX,
    n_fft=n_times,
    n_per_seg=n_times,
    average="mean",
    verbose=False
)

psd_pat = epochs_pat.compute_psd(
    method="welch",
    fmin=PSD_FMIN,
    fmax=PSD_FMAX,
    n_fft=n_times,
    n_per_seg=n_times,
    average="mean",
    verbose=False
)

freqs = psd_base.freqs
roi_idx = [psd_base.ch_names.index(ch) for ch in ROI_CHS if ch in psd_base.ch_names]

base_mean = psd_base.get_data()[:, roi_idx, :].mean(axis=(0, 1))
pat_mean  = psd_pat.get_data()[:, roi_idx, :].mean(axis=(0, 1))

#==================================
# DELTA PSD — CLEAN GAMMA
#==================================
delta_power = pat_mean - base_mean

gamma_mask = (freqs >= GAMMA_BAND[0]) & (freqs <= GAMMA_BAND[1])
expected_mask = (freqs >= EXPECTED_BAND[0]) & (freqs <= EXPECTED_BAND[1])

freqs_gamma = freqs[gamma_mask]
delta_gamma = delta_power[gamma_mask]

# Peak inside expected band ONLY
f_peak = freqs[expected_mask][
    np.argmax(pat_mean[expected_mask])
]

# Top induced gamma frequencies
top_k = 10
top_idx = np.argsort(delta_gamma)[-top_k:][::-1]

print("\nTop induced gamma frequencies (Pattern 3 − Baseline):")
for i in top_idx:
    print(f"{freqs_gamma[i]:.2f} Hz | ΔPower = {delta_gamma[i]:.3e}")


gamma_base = base_mean[gamma_mask].mean()
gamma_pat  = pat_mean[gamma_mask].mean()

gamma_delta = gamma_pat - gamma_base
gamma_pct   = 100 * gamma_delta / gamma_base

print("\n===== BAND-AVERAGED GAMMA POWER =====")
print(f"Baseline gamma power: {gamma_base:.3e}")
print(f"Pattern 3 gamma power: {gamma_pat:.3e}")
print(f"ΔGamma power: {gamma_delta:.3e}")
print(f"ΔGamma (%): {gamma_pct:.2f}%")

peak_power = pat_mean[expected_mask].max()
band_power = pat_mean[gamma_mask].mean()

print(f"Peak / Band ratio: {peak_power / band_power:.2f}")

band_mask = (freqs >= EXPECTED_BAND[0]) & (freqs <= EXPECTED_BAND[1])
f_peak = freqs[band_mask][np.argmax(pat_mean[band_mask])]

print(f"\nLOCKED f_peak = {f_peak:.2f} Hz")
print(f"Baseline SNR: {local_snr(base_mean, freqs, f_peak):.3f}")
print(f"Pattern  SNR: {local_snr(pat_mean,  freqs, f_peak):.3f}")

plv_base, _ = compute_plv_roi(epochs_base, f_peak, ROI_CHS)
plv_pat,  _ = compute_plv_roi(epochs_pat,  f_peak, ROI_CHS)

print("\n===== ROI PLV =====")
print(f"Baseline PLV: {plv_base:.3f}")
print(f"Pattern  PLV: {plv_pat:.3f}")
print(f"ΔPLV: {plv_pat - plv_base:.3f}")
# #==================================
# # PLOT — SAME STYLE AS 240 Hz
# #==================================
# plt.figure(figsize=(7, 4))
# plt.plot(freqs_gamma, delta_gamma, label="Pattern 3 − Baseline")

# plt.axhline(0, color="black", linewidth=0.8)
# plt.axvspan(*EXPECTED_BAND, color="gray", alpha=0.2, label="Expected 40 Hz band")
# plt.axvline(f_peak, linestyle="--", label=f"{f_peak:.2f} Hz")

# plt.xlabel("Frequency (Hz)")
# plt.ylabel("Δ PSD (V²/Hz)")
# plt.title("Gamma-Band Power Difference (Pattern 3 − Baseline)")
# plt.legend()
# plt.tight_layout()
# plt.savefig("psd_delta_gamma_pattern3_clean.png", dpi=300)
# plt.show()
