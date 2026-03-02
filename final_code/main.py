from verify import run_gamma_analysis
from plotting import plot_psd_overlay, plot_distribution
trials_paths = [
    # "../tues_test_2/flicker_p2revised vs baseline - 60hz screen.vhdr",
    # "../fri_cny/pattern1_60hz_vs_baseline-20feb.vhdr",
    # "../tues_test/flicker_p2 vs baseline - 60hz screen.vhdr"
    # "../fri_ptn2/flicker - p2_60hz vs baseline.vhdr"
    # "../fri_ptn4/flicker_p4_60Hz vs baseline.vhdr"
    # "../fri_ptn6_20hz/flicker - p6_20Hz flicker vs baseline.vhdr",
    "../fri_ptn6/flicker - p6_60Hz vs baseline.vhdr"
]
# ROI = ["O1", "Oz", "O2", "PO3", "PO4"]
# ROI = ["O1", "Oz", "O2", "PO3", "PO7","PO8"]
ROI = ["O1", "Oz", "O2", "PO3","PO8"]
TMIN = 2
TMAX = 19
def print_clean_stats(stats_dict):
    print("\n=== METRIC SUMMARY ===\n")

    for metric, vals in stats_dict.items():

        n = vals["n"]
        mean = float(vals["mean_diff"])
        std = float(vals["std_diff"])
        d = float(vals["effect_size"])
        t_p = float(vals["t_p"])
        w_p = float(vals["wilcoxon_p"])
        b_p = float(vals["binom_p"])

        print(f"{metric.upper()}")
        print(f"  n = {n}")
        print(f"  Mean diff = {mean:.3f} dB")
        print(f"  Std diff  = {std:.3f}")
        print(f"  Effect size (d) = {d:.3f}")
        print(f"  p (t-test)   = {t_p:.3f}")
        print(f"  p (Wilcoxon) = {w_p:.3f}")
        print(f"  p (Binomial) = {b_p:.3f}")
        print("-" * 40)
print("\n=== RUNNING ANALYSIS ===")
for VHDR_PATH in trials_paths:
    trial_name = VHDR_PATH.split("/")[-1].split(".")[0]
    print(f"\n=== RUNNING ANALYSIS FOR {trial_name} ===")
    result = run_gamma_analysis(
        VHDR_PATH,
        ROI,
        TMIN,
        TMAX,
        reject_artifacts=True,
        z_thresh=3.0,
        debug=True
    )

    print_clean_stats(result["stats"])

    # --- PLOTTING ---
    plot_psd_overlay(
        freqs=result["freqs"],
        psd_stim=result["psd_pattern"],
        psd_base=result["psd_baseline"],
        title=f"PSD Overlay - {trial_name}",
        save=True
    )