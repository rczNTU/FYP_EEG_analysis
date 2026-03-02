import numpy as np

# ---------------------------------------------------------------
# FIXES:
# 1. compute_psd: explicit n_fft for controlled freq resolution
# 2. snr_40hz_db: gap now == signal_width (was smaller → noise
#    contaminated by signal bins)
# 3. All SNR functions: single log ratio instead of log-subtract
#    (equivalent but cleaner)
# ---------------------------------------------------------------

FREQ_RES_HZ = 0.25   # frequency resolution in Hz (window = 1/res seconds)
                      # 0.25 Hz → 4-second windows → fine enough to resolve
                      # 40 Hz cleanly with ±1.5 Hz signal window


def compute_psd(epochs, fmin=1, fmax=80, freq_res=FREQ_RES_HZ):
    """
    Welch PSD with explicit frequency resolution.

    freq_res controls n_fft:
        n_fft = sfreq / freq_res
        e.g. sfreq=500, freq_res=0.25 → n_fft=2000 (4-second windows)

    Overlap is set to 50% (n_overlap = n_fft // 2).
    Channels are averaged in LINEAR power space (correct — log is nonlinear).
    """
    sfreq = epochs.info["sfreq"]
    n_fft = int(round(sfreq / freq_res))

    psd = epochs.compute_psd(
        method="welch",
        fmin=fmin,
        fmax=fmax,
        n_fft=n_fft,
        n_per_seg=n_fft,
        n_overlap=n_fft // 2,
        verbose=False,
    )

    psds, freqs = psd.get_data(return_freqs=True)

    # psds shape: (n_epochs, n_channels, n_freqs)
    # Average across channels in linear space — correct because log is nonlinear
    psds = psds.mean(axis=1)   # → (n_epochs, n_freqs)

    # Sanity print (silent in production — enable via debug flag if needed)
    # print(f"[PSD] freq_res={freqs[1]-freqs[0]:.4f} Hz  n_fft={n_fft}  shape={psds.shape}")

    return psds, freqs


def gamma_mean_db(psds, freqs, band=(35, 45)):
    """
    Mean power in gamma band, in dB.
    Converts to dB AFTER selecting band (correct — avoids averaging in log space).
    """
    mask = (freqs >= band[0]) & (freqs <= band[1])
    band_power = psds[:, mask].mean(axis=1)          # mean in linear space
    return 10 * np.log10(band_power + 1e-30)          # then convert


def gamma_integral_db(psds, freqs, band=(35, 45)):
    """
    Integral (trapezoid) of power over gamma band, in dB.
    Integrates in linear space, converts at end.
    """
    mask = (freqs >= band[0]) & (freqs <= band[1])
    gamma_lin = np.trapezoid(psds[:, mask], freqs[mask], axis=1)
    return 10 * np.log10(gamma_lin + 1e-30)


def peak_frequency(psds, freqs, band=(30, 50)):
    """Frequency of maximum power within band, per trial."""
    mask = (freqs >= band[0]) & (freqs <= band[1])
    f_sub = freqs[mask]
    peaks = []
    for trial in psds:
        peaks.append(f_sub[np.argmax(trial[mask])])
    return np.array(peaks)


def snr_40hz_db(psds, freqs, center=40, signal_width=1.5, noise_width=6):
    """
    Narrow-band SNR at target frequency (default 40 Hz).

    Signal window : [center - signal_width,  center + signal_width]
    Noise window  : [center - noise_width,   center + noise_width]
                    EXCLUDING the signal window entirely.

    FIX vs original: gap was 1 Hz but signal_width was 1.5 Hz,
    so noise mask included signal bins [1.0, 1.5] Hz from center.
    Now noise exclusion = signal window exactly.
    """
    snr_vals = []

    for trial in psds:
        signal_mask = (freqs >= center - signal_width) & (freqs <= center + signal_width)

        noise_mask = (
            (freqs >= center - noise_width)
            & (freqs <= center + noise_width)
            & ~signal_mask          # exclude EXACTLY the signal window
        )

        signal_power = trial[signal_mask].mean()
        noise_power  = trial[noise_mask].mean()

        snr_db = 10 * np.log10((signal_power + 1e-30) / (noise_power + 1e-30))
        snr_vals.append(snr_db)

    return np.array(snr_vals)


def wide_gamma_snr_db(psds, freqs, band=(35, 45), noise_band=(25, 55)):
    """
    Broadband gamma SNR: mean power in gamma band vs flanking noise.
    Signal and noise windows do not overlap.
    """
    snr_vals = []

    for trial in psds:
        signal_mask = (freqs >= band[0]) & (freqs <= band[1])

        noise_mask = (
            (freqs >= noise_band[0])
            & (freqs <= noise_band[1])
            & ~signal_mask
        )

        signal_power = trial[signal_mask].mean()
        noise_power  = trial[noise_mask].mean()

        snr_db = 10 * np.log10((signal_power + 1e-30) / (noise_power + 1e-30))
        snr_vals.append(snr_db)

    return np.array(snr_vals)