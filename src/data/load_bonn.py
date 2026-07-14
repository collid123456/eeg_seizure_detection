"""Load raw Bonn EEG .txt files into a single labeled dataset."""

import numpy as np
from pathlib import Path


def load_bonn_set(folder_path, label):
    """Load all .txt files in a folder into a list of (signal, label) pairs."""
    folder = Path(folder_path)
    signals = []
    for file in sorted(folder.glob("*.txt")):
        signal = np.loadtxt(file)
        signals.append(signal)
    labels = [label] * len(signals)
    return signals, labels


def load_bonn_dataset(raw_dir="data/raw/bonn"):
    """Load Set Z (non-seizure, label 0) and Set S (seizure, label 1)."""
    z_signals, z_labels = load_bonn_set(f"{raw_dir}/Z", label=0)
    s_signals, s_labels = load_bonn_set(f"{raw_dir}/S", label=1)

    all_signals = z_signals + s_signals
    all_labels = z_labels + s_labels

    print(f"Loaded {len(z_signals)} non-seizure signals (Set Z)")
    print(f"Loaded {len(s_signals)} seizure signals (Set S)")
    print(f"Each signal has {len(all_signals[0])} samples")

    return all_signals, all_labels


if __name__ == "__main__":
    signals, labels = load_bonn_dataset()
    print(f"\nTotal signals: {len(signals)}")
    print(f"Total labels: {len(labels)}")
