import mne
import numpy as np
from scipy.stats import ttest_rel, wilcoxon, binomtest

# ===============================
# CONFIG
# ===============================

# VHDR_PATH = "../fri_ptn6/flicker - p6_60Hz vs baseline.vhdr"
# VHDR_PATH = "../fri_ptn6_20hz/flicker - p6_20Hz flicker vs baseline.vhdr"
# VHDR_PATH = "../fri_cny/pattern1_60hz_vs_baseline-20feb.vhdr"
# VHDR_PATH = "../ptn6_015/flicker - p6-40hz-015alpha015 vs baseline.vhdr"
VHDR_PATH = "../fri_ptn4/flicker_p4_60Hz vs baseline.vhdr"
ROI = ["O1", "Oz", "O2", "PO3", "PO8"]

TMIN = 2
TMAX = 19

FMIN = 1
FMAX = 80

BB_LOW = 35
BB_HIGH = 80

N_BOOT = 5000

# ===============================
# LOAD DATA
# ===============================

print("\n[INFO] Loading EEG...")
raw = mne.io.read_raw_brainvision(VHDR_PATH, preload=True, verbose=False)

misc = [ch for ch in raw.ch_names if raw.get_channel_types(picks=ch)[0] == "misc"]
if misc:
    raw.set_channel_types({ch: "eeg" for ch in misc})

raw.set_montage("standard_1020", on_missing="ignore")
raw.set_eeg_reference("average")

raw.notch_filter(50, verbose=False)
raw.filter(1, 80, verbose=False)

events, event_id = mne.events_from_annotations(raw)

pattern_label = None
baseline_label = None

for k in event_id:
    name = k.lower()
    if "baseline" in name:
        baseline_label = k
    elif "stop" not in name and "lostsamples" not in name:
        pattern_label = k

if pattern_label is None or baseline_label is None:
    raise ValueError("Missing markers")

print("Pattern :", pattern_label)
print("Baseline:", baseline_label)

# ===============================
# EPOCH
# ===============================

pattern_epochs = mne.Epochs(
    raw, events,
    event_id={pattern_label: event_id[pattern_label]},
    tmin=TMIN, tmax=TMAX,
    baseline=None,
    preload=True,
    verbose=False
).pick(ROI)

baseline_epochs = mne.Epochs(
    raw, events,
    event_id={baseline_label: event_id[baseline_label]},
    tmin=TMIN, tmax=TMAX,
    baseline=None,
    preload=True,
    verbose=False
).pick(ROI)

print("Pattern trials :", len(pattern_epochs))
print("Baseline trials:", len(baseline_epochs))

# ===============================
# PSD
# ===============================

psd_p = pattern_epochs.compute_psd(method="welch", fmin=FMIN, fmax=FMAX, verbose=False)
psd_b = baseline_epochs.compute_psd(method="welch", fmin=FMIN, fmax=FMAX, verbose=False)

psds_p, freqs = psd_p.get_data(return_freqs=True)
psds_b, _ = psd_b.get_data(return_freqs=True)

# Average across ROI channels
psds_p = psds_p.mean(axis=1)
psds_b = psds_b.mean(axis=1)

# ===============================
# DESCRIPTIVE TOP PEAKS
# ===============================

mean_p = psds_p.mean(axis=0)
mean_b = psds_b.mean(axis=0)

mean_p_db = 10 * np.log10(mean_p + 1e-30)
mean_b_db = 10 * np.log10(mean_b + 1e-30)

diff_db = mean_p_db - mean_b_db

mask = ~((freqs >= 8) & (freqs <= 12))

freqs_valid = freqs[mask]
diff_valid = diff_db[mask]

sorted_indices = np.argsort(diff_valid)[::-1]


print("\n==============================")
print("TOP FREQUENCY INCREASES")
print("==============================")

for i in range(10):
    idx = sorted_indices[i]
    print(f"{i+1}. {freqs_valid[idx]:.2f} Hz  |  +{diff_valid[idx]:.3f} dB")

# ===============================
# TRIAL-LEVEL BROADBAND GAMMA
# ===============================

print("\n==============================")
print("TRIAL-LEVEL BROADBAND (35–80 Hz)")
print("==============================")

# Convert EACH trial to dB first
psds_p_db = 10 * np.log10(psds_p + 1e-30)
psds_b_db = 10 * np.log10(psds_b + 1e-30)

bb_mask = (freqs >= BB_LOW) & (freqs <= BB_HIGH)

# Mean broadband power per trial
bb_p_trials = psds_p_db[:, bb_mask].mean(axis=1)
bb_b_trials = psds_b_db[:, bb_mask].mean(axis=1)

# Trial-wise difference
bb_diff_trials = bb_p_trials - bb_b_trials

# Basic stats
bb_mean = bb_diff_trials.mean()
bb_std = bb_diff_trials.std()

print("Mean broadband increase:", bb_mean)
print("Std across trials:", bb_std)
# ===============================
# 40 Hz SNR
# ===============================

TARGET = 40
WIDTH = 1.0

target_mask = (freqs >= TARGET - WIDTH) & (freqs <= TARGET + WIDTH)
noise_mask = (freqs >= TARGET - 5) & (freqs <= TARGET + 5)
noise_mask &= ~target_mask

snr_trials = []

for i in range(len(psds_p_db)):
    signal = psds_p_db[i, target_mask].mean()
    noise = psds_p_db[i, noise_mask].mean()
    snr_trials.append(signal - noise)

snr_trials = np.array(snr_trials)

print("\n40 Hz SNR mean (dB):", snr_trials.mean())
print("40 Hz SNR std:", snr_trials.std())
# ===============================
# STATISTICAL TESTS
# ===============================

# Paired t-test
t_stat, p_t = ttest_rel(bb_p_trials, bb_b_trials)

# Wilcoxon (non-parametric)
w_stat, p_w = wilcoxon(bb_p_trials, bb_b_trials)

# Sign test
n_pos = np.sum(bb_diff_trials > 0)
sign_test = binomtest(n_pos, len(bb_diff_trials), 0.5)

# Cohen's d (paired)
cohens_d = bb_mean / bb_std

print("\nPaired t-test p:", p_t)
print("Wilcoxon p:", p_w)
print("Sign test p:", sign_test.pvalue)
print("Cohen's d:", cohens_d)

# ===============================
# BOOTSTRAP CI
# ===============================

boot_means = []

for _ in range(N_BOOT):
    resample = np.random.choice(bb_diff_trials, size=len(bb_diff_trials), replace=True)
    boot_means.append(resample.mean())

ci_low = np.percentile(boot_means, 2.5)
ci_high = np.percentile(boot_means, 97.5)

print("\nBootstrap 95% CI:", (ci_low, ci_high))
percent_positive = np.mean(bb_diff_trials > 0) * 100
print("Percent positive trials:", percent_positive)