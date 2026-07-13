# eeg_seizure_detection
An end-to-end machine learning pipeline for detecting epileptic seizures from scalp EEG recordings. The system filters and segments raw EEG, extracts time-, frequency-, and wavelet-domain features, and classifies short signal windows as ictal (seizure) or interictal (non-seizure).

The project is built around a principle that is often overlooked in EEG-ML work: rigorous, leakage-free evaluation matters more than a headline accuracy number. Models are validated with patient-wise cross-validation, and performance is reported using sensitivity, specificity, AUROC/AUPRC, and false positives per hour — metrics that are actually meaningful under the severe class imbalance inherent to seizure data.


Disclaimer: This is an educational and research project only. It is not a medical device and must not be used for diagnosis, monitoring, or any clinical decision-making.




Overview

Epileptic seizures occupy a small fraction of any EEG recording — often minutes out of many hours — which makes naive accuracy a misleading metric and makes correct experimental design the hardest part of the problem. This project treats seizure detection as a realistic clinical machine learning task rather than a standard classification exercise:


No data leakage. Splits are performed by patient (leave-one-patient-out / grouped cross-validation), not by window, so no recording contributes to both training and test data.
Clinically meaningful metrics. Results are reported as sensitivity, specificity, AUROC, AUPRC, and false positives per hour — not accuracy alone.
Interpretable by default. The primary model operates on hand-crafted, physiologically grounded features (band power, line length, spectral entropy, Hjorth parameters), with SHAP used to explain predictions.
Reproducible. Config-driven runs, modular code, and automated tests replace notebook-only analysis.


Dataset

Built primarily on the CHB-MIT Scalp EEG Database (PhysioNet) — 23 pediatric subjects with continuous scalp EEG and clinician-annotated seizure onset/offset times. Initial development and pipeline validation use the smaller Bonn University EEG dataset for rapid iteration.

Pipeline

Raw EDF → Filtering → Artifact Handling → Windowing → Feature Extraction → Classification → Evaluation


1. Preprocessing — band-pass filtering (0.5–40 Hz), 50 Hz notch filtering, artifact rejection
2. Segmentation — fixed-length overlapping windows, labelled against clinical annotations
3. Feature extraction — time domain (line length, Hjorth parameters, RMS), frequency domain (band power via Welch's method, spectral entropy), and wavelet domain (DWT energies)
4. Classification — gradient-boosted trees (XGBoost) as the primary model, with class-weighted handling of imbalance
5. Evaluation — patient-wise cross-validation, reporting sensitivity, AUROC/AUPRC, and false positives per hour

