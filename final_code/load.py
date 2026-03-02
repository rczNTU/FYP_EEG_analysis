import mne
import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import ttest_rel, binomtest, wilcoxon

# VHDR_PATH = "../tues_test_2/flicker_p2revised vs baseline - 60hz screen.vhdr"
# VHDR_PATH = "../fri_cny/pattern1_60hz_vs_baseline-20feb.vhdr"
# VHDR_PATH = "../fri_cny/pattern4_strong-vs-baseline_120Hzscreen_20feb.vhdr"
# VHDR_PATH = "../tues_test/flicker_p2 vs baseline - 60hz screen.vhdr"
# VHDR_PATH = "../fri_ptn2/flicker - p2_60hz vs baseline.vhdr"
VHDR_PATH = "../fri_ptn4/flicker_p4_60Hz vs baseline.vhdr"

# ROI = ["O1", "Oz", "O2", "PO3", "PO4"]
ROI = ["O1", "Oz", "O2", "PO3", "PO7", "PO8"]

TMIN = 2
TMAX = 19

FMIN = 1
FMAX = 80

FLICKER = 40
SIGNAL_WIDTH = 3
NOISE_GAP = 1.0
NOISE_WIDTH = 5.0

def compute_snr(psds, signal_mask, noise_mask):
    snr_vals = []
    for trial in psds:
        signal = trial[signal_mask].mean()
        noise = trial[noise_mask].mean()
        snr_vals.append(signal / (noise + 1e-30))
    return np.array(snr_vals)
def compute_snr_gamma_log(psds, sig_mask, n1_mask, n2_mask):
    snr_vals = []
    for trial in psds:
        signal = np.trapezoid(trial[sig_mask])
        noise = 0.5 * (
            np.trapezoid(trial[n1_mask]) +
            np.trapezoid(trial[n2_mask])
        )
        snr_vals.append(10 * np.log10(signal / (noise + 1e-30)))
    return np.array(snr_vals)

    return np.array(snr_vals)
def compute_gamma(psds, freqs):
    band = (freqs >= 35) & (freqs <= 45)
    return np.trapezoid(psds[:, band], freqs[band], axis=1)

def compute_peak_freq(psds, freqs):
    peaks = []
    band = (freqs >= 30) & (freqs <= 50)
    for trial in psds:
        sub = trial[band]
        f_sub = freqs[band]
        peaks.append(f_sub[np.argmax(sub)])
    return np.array(peaks)

def paired_effect_size(diff):
    sd = np.std(diff, ddof=1)
    return np.mean(diff) / (sd + 1e-12)
def mean_ci_95(diff):
    # 95% CI for mean difference (normal approx)
    n = len(diff)
    if n < 2:
        return (np.nan, np.nan)
    m = np.mean(diff)
    se = np.std(diff, ddof=1) / np.sqrt(n)
    return (m - 1.96 * se, m + 1.96 * se)

def bootstrap_ci(diff, B=5000, seed=0):
    # nonparametric bootstrap CI for mean difference
    rng = np.random.default_rng(seed)
    n = len(diff)
    if n < 2:
        return (np.nan, np.nan)
    means = np.empty(B)
    for b in range(B):
        sample = rng.choice(diff, size=n, replace=True)
        means[b] = sample.mean()
    return (np.percentile(means, 2.5), np.percentile(means, 97.5))

print("\n[INFO] Loading...")
raw = mne.io.read_raw_brainvision(VHDR_PATH, preload=True, verbose=False)

misc = [ch for ch in raw.ch_names if raw.get_channel_types(picks=ch)[0] == "misc"]
if misc:
    raw.set_channel_types({ch: "eeg" for ch in misc})

raw.set_montage("standard_1020", on_missing="ignore")
raw.set_eeg_reference("average")

print("Sampling rate:", raw.info["sfreq"])

raw.notch_filter(50, verbose=False)
raw.filter(1, 80, verbose=False)

events, event_id = mne.events_from_annotations(raw)

print("\nEvent IDs:")
for k, v in event_id.items():
    print(f"{k} -> {v}")

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

print("\nDetected:")
print("Pattern :", pattern_label)
print("Baseline:", baseline_label)

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

print("\nTrials:")
print("Pattern :", len(pattern_epochs))
print("Baseline:", len(baseline_epochs))

psd_p = pattern_epochs.compute_psd(method="welch", fmin=FMIN, fmax=FMAX, verbose=False)
psd_b = baseline_epochs.compute_psd(method="welch", fmin=FMIN, fmax=FMAX, verbose=False)

psds_p, freqs = psd_p.get_data(return_freqs=True)
psds_b, _ = psd_b.get_data(return_freqs=True)
# ===============================
# Frequency Masks (DEFINE ONCE)
# ===============================

signal_mask = (
    (freqs >= FLICKER - SIGNAL_WIDTH) &
    (freqs <= FLICKER + SIGNAL_WIDTH)
)

noise_mask = (
    (freqs >= FLICKER - NOISE_WIDTH) &
    (freqs <= FLICKER + NOISE_WIDTH) &
    ~((freqs >= FLICKER - NOISE_GAP) &
      (freqs <= FLICKER + NOISE_GAP))
)

# Wide gamma masks
sig_mask = (freqs >= 35) & (freqs <= 45)
n1_mask  = (freqs >= 30) & (freqs < 35)
n2_mask  = (freqs > 45) & (freqs <= 50)

psds_p = psds_p.mean(axis=1)
psds_b = psds_b.mean(axis=1)

snr_p = compute_snr(psds_p, signal_mask, noise_mask)
snr_b = compute_snr(psds_b, signal_mask, noise_mask)

snr_wide_p = compute_snr_gamma_log(psds_p, sig_mask, n1_mask, n2_mask)
snr_wide_b = compute_snr_gamma_log(psds_b, sig_mask, n1_mask, n2_mask)

gamma_p = compute_gamma(psds_p, freqs)
gamma_b = compute_gamma(psds_b, freqs)

peak_p = compute_peak_freq(psds_p, freqs)
peak_b = compute_peak_freq(psds_b, freqs)

# ---- Ensure pairing ----
n = min(len(gamma_p), len(gamma_b))

gamma_p = gamma_p[:n]
gamma_b = gamma_b[:n]

snr_p = snr_p[:n]
snr_b = snr_b[:n]

snr_wide_p = snr_wide_p[:n]
snr_wide_b = snr_wide_b[:n]

peak_p = peak_p[:n]
peak_b = peak_b[:n]

# ---- Log transform gamma ----
gamma_p_db = 10 * np.log10(gamma_p + 1e-30)
gamma_b_db = 10 * np.log10(gamma_b + 1e-30)

# ---- Differences ----
diff_gamma = gamma_p_db - gamma_b_db
diff_snr = snr_p - snr_b
diff_wide = snr_wide_p - snr_wide_b

# ---- Paired tests ----
t_snr, p_snr = ttest_rel(snr_p, snr_b)
t_wide, p_wide = ttest_rel(snr_wide_p, snr_wide_b)
t_gam, p_gam = ttest_rel(gamma_p_db, gamma_b_db)

# ---- Robust paired tests (nonparametric) ----
w_snr, p_snr_w = wilcoxon(snr_p, snr_b, zero_method="wilcox", alternative="two-sided")
w_wide, p_wide_w = wilcoxon(snr_wide_p, snr_wide_b, zero_method="wilcox", alternative="two-sided")
w_gam, p_gam_w = wilcoxon(gamma_p_db, gamma_b_db, zero_method="wilcox", alternative="two-sided")

# ---- Peak shift test (descriptive support) ----
t_peak, p_peak = ttest_rel(peak_p, peak_b)
diff_peak = peak_p - peak_b

# ---- Confidence intervals (gamma diff is main) ----
ci_gam = mean_ci_95(diff_gamma)
boot_ci_gam = bootstrap_ci(diff_gamma, B=5000, seed=1)

#binom test
n_pos = (diff_gamma > 0).sum()
n_total = len(diff_gamma)

p_binom = binomtest(n_pos, n_total, p=0.5, alternative='greater').pvalue
print("BINOMIAL Sign test p:", p_binom)

print(f"t_snr = {t_snr}, p_snr = {p_snr}")

print("\n==============================")
print("Peak Frequency")
print("==============================")
print("Pattern :", peak_p.mean())
print("Baseline:", peak_b.mean())
print("Peak shift mean (Pattern - Baseline):", diff_peak.mean())
print("Peak shift p =", p_peak)

print("\n==============================")
print("40 Hz SNR (paired)")
print("==============================")
print("Pattern :", snr_p.mean())
print("Baseline:", snr_b.mean())
print("Mean diff:", diff_snr.mean())
print("Effect size:", paired_effect_size(diff_snr))
print("p =", p_snr)
print("Wilcoxon p =", p_snr_w)

print("\n==============================")
print("Wide SNR (35–45 Hz, paired)")
print("==============================")
print("Pattern :", snr_wide_p.mean())
print("Baseline:", snr_wide_b.mean())
print("Mean diff:", diff_wide.mean())
print("Effect size:", paired_effect_size(diff_wide))
print("p =", p_wide)
print("Wilcoxon p =", p_wide_w)

print("\n==============================")
print("==============================")
print("Pattern :", gamma_p_db.mean())
print("Baseline:", gamma_b_db.mean())
print("Mean diff:", diff_gamma.mean())
print("Effect size:", paired_effect_size(diff_gamma))
print("p =", p_gam)
print("Wilcoxon p =", p_gam_w)
print("95% CI (mean diff):", ci_gam)
print("Bootstrap CI (mean diff):", boot_ci_gam)
print("\nPositive trials (gamma):", (diff_gamma > 0).sum(), "/", len(diff_gamma))
print("Std diff:", diff_gamma.std())

mean_p_db = 10 * np.log10(psds_p.mean(axis=0) + 1e-30)
mean_b_db = 10 * np.log10(psds_b.mean(axis=0) + 1e-30)

plt.figure()
plt.plot(freqs, mean_p_db, label="Pattern")
plt.plot(freqs, mean_b_db, label="Baseline")
plt.xlim(0, 60)
plt.title("PSD (dB)")
plt.legend()
plt.show()