import mne


def load_raw(vhdr_path, notch=50, l_freq=1, h_freq=80):
    raw = mne.io.read_raw_brainvision(vhdr_path, preload=True, verbose=False)

    # Fix misc → EEG
    misc = [ch for ch in raw.ch_names if raw.get_channel_types(picks=ch)[0] == "misc"]
    if misc:
        raw.set_channel_types({ch: "eeg" for ch in misc})

    raw.set_montage("standard_1020", on_missing="ignore")
    #For every time sample:compute mean of all EEG channels,subtract that mean from each channel
    raw.set_eeg_reference("average")

    raw.notch_filter(notch, verbose=False)
    raw.filter(l_freq, h_freq, verbose=False)

    return raw


def extract_events(raw):
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
        raise ValueError("Missing pattern or baseline markers")

    return events, event_id, pattern_label, baseline_label


def create_epochs(raw, events, event_id, label, roi, tmin, tmax):
    epochs = mne.Epochs(
        raw,
        events,
        event_id={label: event_id[label]},
        tmin=tmin,
        tmax=tmax,
        baseline=None,
        preload=True,
        verbose=False,
    )

    return epochs.pick(roi)