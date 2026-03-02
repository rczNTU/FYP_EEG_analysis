import numpy as np
from preprocessing import load_raw, extract_events, create_epochs
from metrics import compute_psd, gamma_mean_db, snr_40hz_db, wide_gamma_snr_db, peak_frequency, gamma_integral_db
from stats import paired_stats

# ---------------------------------------------------------------
# FIXES:
# 1. reject_artifacts_by_power: now also prints which trials
#    were removed (indices) when debug=True so you can inspect
# 2. run_gamma_analysis: added freq_res diagnostic print,
#    absolute SNR values printed (not just diff) so you can tell
#    whether the stimulus is driving 40 Hz at all
# 3. Paired stats now called after artifact rejection (was
#    already correct, kept)
# 4. Added sanity check: warns if n_pattern != n_baseline
# ---------------------------------------------------------------


def reject_artifacts_by_power(x, y, z_thresh=2.5, debug=False):
    """
    Reject paired trials where broadband gamma power (either condition)
    is a robust outlier (MAD z-score > z_thresh).

    Uses combined distribution of both conditions so threshold is
    consistent across conditions.
    """
    n = min(len(x), len(y))
    x = x[:n]
    y = y[:n]

    combined = np.concatenate([x, y])

    median = np.median(combined)
    mad    = np.median(np.abs(combined - median)) + 1e-12

    zx = 0.6745 * (x - median) / mad
    zy = 0.6745 * (y - median) / mad

    mask = (np.abs(zx) < z_thresh) & (np.abs(zy) < z_thresh)

    if debug:
        removed_idx = np.where(~mask)[0]
        print(f"\n[Artifact Rejection] z_thresh={z_thresh}")
        print(f"  Removed {(~mask).sum()} / {n} trials")
        if len(removed_idx):
            print(f"  Removed trial indices: {removed_idx.tolist()}")
            print(f"  Their zx values: {np.round(zx[removed_idx], 2)}")
            print(f"  Their zy values: {np.round(zy[removed_idx], 2)}")

    return x[mask], y[mask], mask


def run_gamma_analysis(
    vhdr_path,
    roi,
    tmin,
    tmax,
    reject_artifacts=True,
    z_thresh=3.0,
    debug=False,
):
    raw = load_raw(vhdr_path)

    events, event_id, pattern_label, baseline_label = extract_events(raw)

    pattern_epochs  = create_epochs(raw, events, event_id, pattern_label,  roi, tmin, tmax)
    baseline_epochs = create_epochs(raw, events, event_id, baseline_label, roi, tmin, tmax)

    # Sanity check — unequal trial counts will silently truncate
    n_p = len(pattern_epochs)
    n_b = len(baseline_epochs)
    if n_p != n_b:
        print(f"\n[WARNING] Unequal trial counts: pattern={n_p}, baseline={n_b}. "
              f"Will use first {min(n_p, n_b)} of each.")

    psds_p, freqs = compute_psd(pattern_epochs)
    psds_b, _     = compute_psd(baseline_epochs)

    if debug:
        print(f"\n[PSD] freq_res = {freqs[1] - freqs[0]:.4f} Hz")
        print(f"[PSD] freq range = {freqs[0]:.1f} – {freqs[-1]:.1f} Hz")
        print(f"[PSD] n_trials: pattern={len(psds_p)}, baseline={len(psds_b)}")

    # ---- Compute all metrics in linear PSD space ----
    gamma_p     = gamma_mean_db(psds_p, freqs)
    gamma_b     = gamma_mean_db(psds_b, freqs)

    gamma_int_p = gamma_integral_db(psds_p, freqs)
    gamma_int_b = gamma_integral_db(psds_b, freqs)

    snr40_p     = snr_40hz_db(psds_p, freqs)
    snr40_b     = snr_40hz_db(psds_b, freqs)

    wide_snr_p  = wide_gamma_snr_db(psds_p, freqs)
    wide_snr_b  = wide_gamma_snr_db(psds_b, freqs)

    peak_p      = peak_frequency(psds_p, freqs)
    peak_b      = peak_frequency(psds_b, freqs)

    # ---- Artifact rejection ----
    if reject_artifacts:
        gamma_p, gamma_b, mask = reject_artifacts_by_power(
            gamma_p, gamma_b, z_thresh=z_thresh, debug=debug
        )

        gamma_int_p = gamma_int_p[mask]
        gamma_int_b = gamma_int_b[mask]
        snr40_p     = snr40_p[mask]
        snr40_b     = snr40_b[mask]
        wide_snr_p  = wide_snr_p[mask]
        wide_snr_b  = wide_snr_b[mask]
        peak_p      = peak_p[mask]
        peak_b      = peak_b[mask]

    # ---- Debug output ----
    if debug:
        print("\n=== Metric Evidence (per trial) ===")
        print("Gamma Mean dB Diff:     ", np.round(gamma_p     - gamma_b,     3))
        print("Gamma Integral dB Diff: ", np.round(gamma_int_p - gamma_int_b, 3))
        print("40Hz SNR dB Diff:       ", np.round(snr40_p     - snr40_b,     3))
        print("Wide Gamma SNR dB Diff: ", np.round(wide_snr_p  - wide_snr_b,  3))
        print("Peak Frequency Shift:   ", np.round(peak_p      - peak_b,      3))

        # CRITICAL: absolute SNR tells you whether stimulus drives 40 Hz at all
        # If both conditions show ~0 dB SNR, the stimulus is not working.
        # If pattern > baseline > 0, it's working but effect may be small.
        print("\n=== Absolute SNR (pattern vs baseline) ===")
        print(f"40Hz SNR pattern  mean = {snr40_p.mean():.3f} dB  (std={snr40_p.std():.3f})")
        print(f"40Hz SNR baseline mean = {snr40_b.mean():.3f} dB  (std={snr40_b.std():.3f})")
        print(f"Wide SNR pattern  mean = {wide_snr_p.mean():.3f} dB")
        print(f"Wide SNR baseline mean = {wide_snr_b.mean():.3f} dB")

    # ---- Stats ----
    results = {
        "gamma_mean_db":      paired_stats(gamma_p,     gamma_b),
        "gamma_integral_db":  paired_stats(gamma_int_p, gamma_int_b),
        "snr_40hz_db":        paired_stats(snr40_p,     snr40_b),
        "wide_gamma_snr_db":  paired_stats(wide_snr_p,  wide_snr_b),
        "peak_frequency_shift": paired_stats(peak_p,    peak_b),
    }

    return {
        "pattern_label":  pattern_label,
        "baseline_label": baseline_label,

        "freqs":        freqs,
        "psd_pattern":  psds_p,
        "psd_baseline": psds_b,

        "gamma_mean_pattern":      gamma_p,
        "gamma_mean_baseline":     gamma_b,
        "gamma_integral_pattern":  gamma_int_p,
        "gamma_integral_baseline": gamma_int_b,
        "snr40_pattern":           snr40_p,
        "snr40_baseline":          snr40_b,
        "wide_snr_pattern":        wide_snr_p,
        "wide_snr_baseline":       wide_snr_b,
        "peak_pattern":            peak_p,
        "peak_baseline":           peak_b,

        "stats": results,
    }