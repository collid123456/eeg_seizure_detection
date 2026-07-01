# eeg_seizure_detection
End-to-end EEG seizure-detection pipeline in Python. Filters and segments scalp EEG, extracts time/frequency/wavelet features, and classifies ictal vs interictal windows. Uses patient-wise cross-validation and reports AUROC/sensitivity/false-positives-per-hour over accuracy. Built on the CHB-MIT dataset.
