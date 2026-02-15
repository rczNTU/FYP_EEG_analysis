import mne
import numpy as np
from scipy.stats import ttest_ind


# =====================================================
# CONFIG
# =====================================================

# VHDR_PATH = "../fyp-test-bv/flicker-fyp(p3-10trials).vhdr"
# VHDR_PATH = "../fyp-test-bv/flicker-fyp(p5-stronger-10trials).vhdr"
VHDR_PATH  = "../fyp-test-bv/flicker-fyp(p1-10trials-120hz).vhdr"

ROI = ["O1", "Oz", "O2", "PO3", "PO4"]

TMIN = 4
TMAX = 25

FLICKER_FREQ = 40
SIGNAL_WIDTH = 0.5
NOISE_GAP = 1.0
NOISE_WIDTH = 3.0

NOTCH_FREQ = 50
BANDPASS = (1, 80)


# =====================================================
# LOAD + PREP
# =====================================================

def load_raw(path):
    print("\nLoading file...")
    raw = mne.io.read_raw_brainvision(path, preload=True, verbose=False)

    # Convert misc → eeg if needed
    misc = [
        ch for ch in raw.info["ch_names"]
        if raw.get_channel_types(picks=ch)[0] == "misc"
    ]
    if misc:
        raw.set_channel_types({ch: "eeg" for ch in misc})

    raw.set_montage("standard_1020", on_missing="ignore")
    raw.set_eeg_reference("average")

    print("Sampling rate:", raw.info["sfreq"])
    return raw


# =====================================================
# FILTER
# =====================================================

def apply_filters(raw):
    raw.notch_filter(NOTCH_FREQ, verbose=False)
    raw.filter(BANDPASS[0], BANDPASS[1], verbose=False)
    return raw


# =====================================================
# EVENTS
# =====================================================

def get_conditions(raw):
    events, event_id = mne.events_from_annotations(raw)

    # find any pattern condition automatically
    pattern_label = None
    for k in event_id:
        if "stop" not in k.lower() and "lostsamples" not in k.lower():
            pattern_label = k

    stop_label = [k for k in event_id if "stop" in k.lower()][0]

    if pattern_label is None:
        raise ValueError("No pattern condition found.")

    print("Pattern:", pattern_label)
    print("Stop:", stop_label)

    return events, event_id, pattern_label, stop_label



# =====================================================
# EPOCHS
# =====================================================

def make_epochs(raw, events, event_id, label):
    epochs = mne.Epochs(
        raw,
        events,
        event_id={label: event_id[label]},
        tmin=TMIN,
        tmax=TMAX,
        baseline=None,
        preload=True,
        verbose=False
    )
    return epochs.pick(ROI)


# =====================================================
# 40 Hz SNR
# =====================================================

def compute_40hz_snr(epochs):

    psd = epochs.compute_psd(
        method="multitaper",
        fmin=FLICKER_FREQ - NOISE_WIDTH,
        fmax=FLICKER_FREQ + NOISE_WIDTH,
        verbose=False
    )

    psds, freqs = psd.get_data(return_freqs=True)

    # Average channels
    psds = psds.mean(axis=1)

    snr_values = []

    for trial_psd in psds:

        signal_mask = (
            (freqs >= FLICKER_FREQ - SIGNAL_WIDTH) &
            (freqs <= FLICKER_FREQ + SIGNAL_WIDTH)
        )

        noise_mask = (
            (freqs >= FLICKER_FREQ - NOISE_WIDTH) &
            (freqs <= FLICKER_FREQ + NOISE_WIDTH) &
            (~(
                (freqs >= FLICKER_FREQ - NOISE_GAP) &
                (freqs <= FLICKER_FREQ + NOISE_GAP)
            ))
        )

        signal_power = trial_psd[signal_mask].mean()
        noise_power = trial_psd[noise_mask].mean()

        snr_values.append(signal_power / noise_power)

    return np.array(snr_values)

def compute_induced_gamma(epochs, fmin=30, fmax=50):

    psd = epochs.compute_psd(
        method="multitaper",
        fmin=fmin,
        fmax=fmax,
        verbose=False
    )

    psds, freqs = psd.get_data(return_freqs=True)

    # Average channels first
    psds = psds.mean(axis=1)

    # Integrate broadband power
    gamma_power = np.trapz(psds, freqs, axis=1)

    return gamma_power
# =====================================================
# REPORT
# =====================================================

def report_results(pattern_snr, stop_snr):

    tval, pval = ttest_ind(pattern_snr, stop_snr, equal_var=False)


    print("\n==============================")
    print("40 Hz Narrowband SNR")
    print("==============================")

    print("Pattern SNR:", pattern_snr)
    print("Stop SNR:", stop_snr)

    print("\nMean Pattern:", pattern_snr.mean())
    print("Mean Stop:", stop_snr.mean())

    print("\nPaired t-test")
    print("t =", round(tval, 4))
    print("p =", round(pval, 6))

    if pval < 0.05 and pattern_snr.mean() > stop_snr.mean():
        print("\nSignificant 40 Hz entrainment detected")
    else:
        print("\nNo significant entrainment")


# =====================================================
# MAIN
# =====================================================

def main():

    raw = load_raw(VHDR_PATH)
    raw = apply_filters(raw)

    events, event_id, pattern_lbl, stop_lbl = get_conditions(raw)

    pattern_epochs = make_epochs(raw, events, event_id, pattern_lbl)
    stop_epochs = make_epochs(raw, events, event_id, stop_lbl)

    print("Pattern trials:", len(pattern_epochs))
    print("Stop trials:", len(stop_epochs))

        # ----------------------------
    # Induced Broadband Gamma
    # ----------------------------

    pattern_gamma = compute_induced_gamma(pattern_epochs)
    stop_gamma = compute_induced_gamma(stop_epochs)

    tval, pval = ttest_ind(pattern_gamma, stop_gamma, equal_var=False)

    print("\n==============================")
    print("Induced Broadband Gamma (30–50 Hz)")
    print("==============================")

    print("Pattern gamma:", pattern_gamma)
    print("Stop gamma:", stop_gamma)

    print("\nMean Pattern:", pattern_gamma.mean())
    print("Mean Stop:", stop_gamma.mean())

    print("\nWelch t-test")
    print("t =", round(tval, 4))
    print("p =", round(pval, 6))

    if pval < 0.05 and pattern_gamma.mean() > stop_gamma.mean():
        print("\nSignificant induced gamma increase detected")
    else:
        print("\nNo significant induced gamma increase")



if __name__ == "__main__":
    main()
