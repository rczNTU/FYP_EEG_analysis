EEG Analysis Pipeline Documentation
Overview

This repository implements a modular EEG analysis pipeline designed to evaluate 40 Hz gamma entrainment effects under different visual stimulation patterns.

The pipeline is structured to:

Ensure reproducibility

Separate signal processing from statistical analysis

Enable session-level and cross-session comparisons

Provide artifact-robust gamma band analysis

The architecture follows a clear separation of concerns:

preprocessing.py  → raw loading + epoching
metrics.py        → PSD + gamma computation
stats.py          → paired statistical tests
verify.py         → full pipeline orchestration
main.py           → experiment runner
1. Preprocessing Stage (preprocessing.py)
Purpose

Prepare raw BrainVision EEG data for frequency-domain analysis.

Steps Performed
1.1 Raw Loading

Reads .vhdr BrainVision file

Loads data into memory

raw = mne.io.read_raw_brainvision(...)
1.2 Channel Type Enforcement

All channels are explicitly set to EEG before applying montage:

raw.set_channel_types({ch: "eeg" for ch in raw.ch_names})

This prevents MNE from assigning channels as misc.

1.3 Montage

Standard 10–20 montage applied:

raw.set_montage("standard_1020", on_missing="ignore")
1.4 Re-referencing

Common average reference:

raw.set_eeg_reference("average")
1.5 Filtering

50 Hz notch filter (line noise removal)

1–80 Hz bandpass filter

raw.notch_filter(50)
raw.filter(1, 80)
1.6 Event Extraction

Automatically detects:

Pattern condition

Baseline condition

Based on annotation names containing:

"baseline"

Any non-stop non-lostsamples marker

1.7 Epoching

Creates epochs:

Time window: TMIN to TMAX

Baseline correction: none

ROI channels only

ROI:

["O1", "Oz", "O2", "PO3", "PO4"]
2. Frequency-Domain Analysis (metrics.py)
Purpose

Extract trial-level frequency features.

2.1 PSD Computation

Method:

Welch PSD

Frequency range: 1–80 Hz

Channel averaging is performed before gamma extraction.

2.2 Gamma Band Definition

Gamma band:

35–45 Hz

Gamma power is computed as:

Convert PSD to dB

Average across gamma band frequencies

gamma_db = mean(10 * log10(psd))

Important:

Log transform is performed BEFORE band averaging.

This avoids nonlinear bias.

3. Artifact Rejection (verify.py)
Purpose

Remove EMG/motion contamination.

Why Rejection Is Necessary

Artifacts appear as:

Broadband power spikes

Gamma values around -70 to -80 dB

Normal EEG gamma is ~ -128 to -131 dB

Difference-based rejection fails when artifacts inflate variance symmetrically.

Method Used

Robust median-based rejection using MAD:

median absolute deviation (MAD)

Procedure:

Combine pattern and baseline gamma values

Compute median

Compute MAD

Compute robust z-score

Reject trials exceeding threshold (default z = 2.5)

This method:

Is robust to extreme values

Does not assume normality

Removes EMG without removing strong real effects

4. Statistical Analysis (stats.py)

Paired statistics are computed after artifact rejection.

Tests Performed
4.1 Paired t-test

Tests mean gamma difference.

4.2 Wilcoxon Signed-Rank Test

Non-parametric paired test.

4.3 Binomial Sign Test

Tests whether proportion of positive trials > 50%.

Effect Size

Cohen’s d (paired):

mean(diff) / std(diff)
5. Execution Flow (verify.py)

The pipeline execution order:

Load + preprocess raw

Extract events

Create epochs

Compute PSD

Extract gamma (dB)

Artifact rejection

Paired statistical tests

Return structured result dictionary