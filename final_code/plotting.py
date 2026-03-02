import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import numpy as np
import os

# ---------------------------------------------------------------
# FIXES:
# 1. plot_psd_overlay: SEM shading was computing log of
#    (mean ± sem) — this can produce log(negative) when sem > mean.
#    Fixed: compute SEM in dB space after log transform.
# 2. Added 40 Hz marker label and gamma band shading.
# 3. Added plot_snr_distributions: shows per-trial SNR for both
#    conditions side by side — most informative diagnostic plot.
# 4. Added plot_trial_heatmap: shows per-trial gamma power as
#    heatmap — reveals session drift and outlier structure.
# ---------------------------------------------------------------

FIGURES_DIR = "figures"


def _ensure_figures_dir():
    os.makedirs(FIGURES_DIR, exist_ok=True)


def plot_psd_overlay(freqs, psd_stim, psd_base, title="PSD Overlay", save=False):
    """
    Mean ± SEM PSD overlay for stimulus vs baseline.

    SEM computed in dB space (after log transform) to avoid
    log(negative) errors when power is small.
    """
    # Convert trials to dB, then compute mean and SEM
    stim_db = 10 * np.log10(psd_stim + 1e-30)   # (n_trials, n_freqs)
    base_db = 10 * np.log10(psd_base + 1e-30)

    mean_stim = stim_db.mean(axis=0)
    mean_base = base_db.mean(axis=0)

    sem_stim = stim_db.std(axis=0) / np.sqrt(len(stim_db))
    sem_base = base_db.std(axis=0) / np.sqrt(len(base_db))

    fig, ax = plt.subplots(figsize=(10, 5))

    ax.plot(freqs, mean_base, label="Baseline", color="steelblue", linewidth=1.5)
    ax.plot(freqs, mean_stim, label="Stimulus",  color="tomato",    linewidth=1.5)

    ax.fill_between(freqs, mean_base - sem_base, mean_base + sem_base,
                    alpha=0.25, color="steelblue")
    ax.fill_between(freqs, mean_stim - sem_stim, mean_stim + sem_stim,
                    alpha=0.25, color="tomato")

    # Gamma band shading
    ax.axvspan(35, 45, alpha=0.07, color="gold", label="35–45 Hz")

    # 40 Hz marker
    ax.axvline(40, linestyle="--", color="gray", alpha=0.7, linewidth=1, label="40 Hz")

    ax.set_xlim(1, 80)
    ax.set_xlabel("Frequency (Hz)")
    ax.set_ylabel("Power (dB)")
    ax.set_title(title)
    ax.legend()
    fig.tight_layout()

    if save:
        _ensure_figures_dir()
        path = os.path.join(FIGURES_DIR, f"{title}_psd.png")
        fig.savefig(path, dpi=300)
        plt.close(fig)
        print(f"[PLOT] Saved: {path}")
    else:
        plt.show()


def plot_snr_distributions(snr_stim, snr_base, title="40Hz SNR", save=False):
    """
    Side-by-side histogram + paired line plot of per-trial 40Hz SNR.

    This is the most important diagnostic:
    - If both conditions cluster near 0 dB → stimulus not driving 40 Hz
    - If stimulus is right-shifted → entrainment is working
    """
    diffs = snr_stim - snr_base

    fig, axes = plt.subplots(1, 2, figsize=(12, 5))

    # --- Left: paired dot + line plot ---
    ax = axes[0]
    n = len(snr_stim)
    jitter = np.random.default_rng(0).uniform(-0.05, 0.05, n)

    for i in range(n):
        ax.plot([0 + jitter[i], 1 + jitter[i]], [snr_base[i], snr_stim[i]],
                color="gray", alpha=0.5, linewidth=0.8)

    ax.scatter(np.zeros(n) + jitter, snr_base,  color="steelblue", zorder=3, label="Baseline")
    ax.scatter(np.ones(n)  + jitter, snr_stim,  color="tomato",    zorder=3, label="Stimulus")

    ax.axhline(0, linestyle="--", color="black", alpha=0.4, linewidth=0.8)
    ax.set_xticks([0, 1])
    ax.set_xticklabels(["Baseline", "Stimulus"])
    ax.set_ylabel("40 Hz SNR (dB)")
    ax.set_title(f"{title} — Per-Trial")
    ax.legend()

    # --- Right: histogram of diffs ---
    ax2 = axes[1]
    ax2.hist(diffs, bins=max(5, n // 2), color="mediumpurple", edgecolor="white")
    ax2.axvline(0, linestyle="--", color="black", alpha=0.6)
    ax2.axvline(diffs.mean(), linestyle="-", color="red",
                alpha=0.8, label=f"mean={diffs.mean():.3f} dB")
    ax2.set_xlabel("Stimulus − Baseline (dB)")
    ax2.set_ylabel("Count")
    ax2.set_title(f"{title} — Difference Distribution")
    ax2.legend()

    fig.tight_layout()

    if save:
        _ensure_figures_dir()
        path = os.path.join(FIGURES_DIR, f"{title}_snr_dist.png")
        fig.savefig(path, dpi=300)
        plt.close(fig)
        print(f"[PLOT] Saved: {path}")
    else:
        plt.show()


def plot_trial_heatmap(freqs, psd_stim, psd_base, title="Trial Heatmap", save=False):
    """
    Heatmap of per-trial PSD (dB) for stimulus and baseline.

    Reveals session drift (power changing across trials) and
    structural outliers that survive MAD rejection.
    """
    stim_db = 10 * np.log10(psd_stim + 1e-30)
    base_db = 10 * np.log10(psd_base + 1e-30)

    freq_mask = (freqs >= 1) & (freqs <= 80)
    f_plot    = freqs[freq_mask]

    vmin = min(stim_db[:, freq_mask].min(), base_db[:, freq_mask].min())
    vmax = max(stim_db[:, freq_mask].max(), base_db[:, freq_mask].max())

    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    for ax, data, label in zip(axes,
                                [base_db[:, freq_mask], stim_db[:, freq_mask]],
                                ["Baseline", "Stimulus"]):
        im = ax.imshow(
            data,
            aspect="auto",
            origin="upper",
            extent=[f_plot[0], f_plot[-1], len(data), 0],
            vmin=vmin,
            vmax=vmax,
            cmap="viridis",
        )
        ax.axvline(40, color="red", linewidth=0.8, linestyle="--", alpha=0.7)
        ax.set_xlabel("Frequency (Hz)")
        ax.set_ylabel("Trial")
        ax.set_title(f"{title} — {label}")
        plt.colorbar(im, ax=ax, label="Power (dB)")

    fig.tight_layout()

    if save:
        _ensure_figures_dir()
        path = os.path.join(FIGURES_DIR, f"{title}_heatmap.png")
        fig.savefig(path, dpi=300)
        plt.close(fig)
        print(f"[PLOT] Saved: {path}")
    else:
        plt.show()


def plot_distribution(metric_name, pattern_vals, baseline_vals, save=False):
    """Simple histogram of per-trial diffs (kept for backward compatibility)."""
    diffs = pattern_vals - baseline_vals

    fig, ax = plt.subplots(figsize=(6, 4))
    ax.hist(diffs, bins=10, edgecolor="white")
    ax.axvline(0, linestyle="--", color="black", alpha=0.6)
    ax.axvline(diffs.mean(), linestyle="-", color="red",
               alpha=0.8, label=f"mean={diffs.mean():.3f}")
    ax.set_title(f"{metric_name}\nMean diff = {diffs.mean():.3f} dB")
    ax.set_xlabel("Pattern − Baseline (dB)")
    ax.set_ylabel("Count")
    ax.legend()
    fig.tight_layout()

    if save:
        _ensure_figures_dir()
        path = os.path.join(FIGURES_DIR, f"{metric_name}_distribution.png")
        fig.savefig(path, dpi=300)
        plt.close(fig)
        print(f"[PLOT] Saved: {path}")
    else:
        plt.show()