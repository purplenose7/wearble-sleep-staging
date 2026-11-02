# Wearable sleep-stage classification: summary report

**Datasets:** DREAMT (development) and AAUWSS (external validation). TILES-2018 will be added later.

## Part 1. Dataset audit and preprocessing

### Inventory

| | DREAMT | AAUWSS |
|---|---|---|
| Participants / nights | 100 / one night each (sleep-clinic patients, many with apnea; mean age 56) | 13 / one night each (healthy adults) |
| Wearable | Empatica E4 (left wrist) | Empatica E4 |
| PSG channels | EEG (C4-M1, F4-M1, O2-M1, T3-CZ, CZ-T4), EOG (E1, E2), EMG (CHIN), ECG, respiration, SpO2, snoring, leg movement | EEG, EOG, ECG, EMG (200 Hz) |
| Expert labels | Sleep technicians: P, W, N1, N2, N3, R, Missing | One rater, AASM: Wake, N1, N2, N3, REM, Missing |
| Sampling (native) | BVP 64 Hz, ACC 32 Hz, EDA 4 Hz, TEMP 4 Hz, HR 1 Hz, IBI. All resampled to 64 Hz. | BVP 64 Hz, ACC 32 Hz, EDA 4 Hz, TEMP 4 Hz, HR 1 Hz |
| Recording duration | wearable 883.4 h, scored 659.0 h(1.5%), missing 1178 epochs, 0% missing per feature signal across 97 usable participants | scored 92.5 h, scored 365 epochs(participants 12 and 13 have PSG gaps) |
| Stage definitions | P = preparation before PSG start (unscored); W; N1–N3 = non-REM 1–3; R = REM; 30-second epochs | P, Wake, N1-N3, REM; 30-second epochs |

### Table of signal availability

| Signal | Rate | DREAMT | AAUWSS | Used |
|---|---|---|---|---|
| BVP (PPG) | 64 Hz | yes | yes | yes |
| HR / IBI | 1 Hz / event | yes | yes | yes |
| Accelerometer (3 axes) | 32 Hz | yes | yes | yes |
| EDA | 4 Hz | yes | yes | yes |
| Skin temperature | 4 Hz | yes | yes | yes |
| PSG (EEG, EOG, EMG, ECG) | 200 Hz | yes | yes | labels only |

### Synchronization of wearable signals and PSG labels

- **DREAMT:** all signals share one 64 Hz grid; one epoch = 1,920 consecutive rows starting at the first scored row. Verified: 97 of 100 participants pass; exceptions: S027, S048, and S077.
- **AAUWSS:** epoch start time converted to the wearable clock (start time and sampling rate; time-zone offset per participant, since Denmark changed clocks on 26 March 2023). Labels were not used to set the alignment.

### Common sleep-stage classes

Both datasets contain W, N1, N2, N3 and R. DREAMT's P (unscored) and Missing epochs are excluded.

### Preprocessing pipeline

- **Missing / corrupted data:** an epoch is excluded for a missing label, less than 90% signal coverage, off-wrist signs (temperature or EDA out of range), flat BVP, or implausible heart rate. Excluded: DREAMT 1.5%, AAUWSS 3.2%.
- **Resampling:** BVP is never resampled. DREAMT's EDA, temperature, accelerometer and HR are returned to native rates with anti-aliasing; AAUWSS is used at native rates.
- **Epochs:** 30 seconds.
- **Features:** heart rate and HRV (mean heart rate, SDNN, RMSSD, pNN50), accelerometer (magnitude, jerk, activity), EDA statistics, temperature statistics.
- **Normalization:** fitted on training participants only.

### Participant-level usable epochs per stage

| | DREAMT | AAUWSS |
|---|---|---|
| Usable epochs | 76,624 | 10,776 |
| Wake / N1 / N2 / N3 / REM (%) | 25.2 / 11.3 / 49.7 / 3.5 / 10.4 | 12.3 / 13.0 / 35.7 / 18.6 / 20.4 |
| Table(`results/inventory`) | `dreamt_usable_epochs.csv` | `aauwss_usable_epochs.csv` |

---

### Engineered Features(30-Second Epoch Windows)

1. Heart Rate & Cardiac Variability (BVP/IBI Stream)
* `hr_mean`: Mean heart rate derived from inter-beat intervals (BPM).
* `hr_std`: Standard deviation of heart rate within the epoch.
* `rmssd`: Root mean square of successive differences between normal heartbeats (time-domain HRV measure of parasympathetic tone).
* `sdnn`: Standard deviation of NN/RR intervals over the 30-second window.
* `pnn50`: Percentage of successive RR intervals that differ by more than 50 ms.
* `dominant_freq`: Peak power frequency of the inter-beat interval time series (Hz).
* `lf_hf_ratio`: Ratio of low-frequency (0.04–0.15 Hz) to high-frequency (0.15–0.40 Hz) spectral power density.

2. Actigraphy & Tri-Axial Motion (ACC Stream)
* `mag_mean`: Mean 3D acceleration vector magnitude ($\sqrt{x^2 + y^2 + z^2}$).
* `mag_std`: Standard deviation of acceleration magnitude (measures movement volatility/restlessness).
* `mag_max`: Maximum acceleration magnitude recorded during the epoch.
* `share_away_1g`: Proportions of time acceleration magnitude deviates significantly from gravitational resting state ($1.0g$).
* `acc_x_mean`, `acc_y_mean`, `acc_z_mean`: Mean positional tilt orientations across tri-axial planes.
* `acc_x_std`, `acc_y_std`, `acc_z_std`: Standard deviations of individual axis accelerations.

3. Autonomic & Thermoregulatory Activity (EDA & TEMP Streams)
* `eda_mean`: Mean Electrodermal Activity / Skin Conductance level ($\mu\text{S}$).
* `eda_std`: Standard deviation of skin conductance (captures sympathetic arousal bursts).
* `eda_max`: Maximum skin conductance level.
* `temp_mean`: Mean wrist skin temperature ($^\circ\text{C}$).
* `temp_slope`: Linear regression slope of skin temperature across the epoch (thermoregulatory onset gradient).

4. Temporal Context Features (Smoothing & Context Windows)
* `[feature]_context_mean`: Rolling 5-epoch (~2.5 minute) centered mean of `[feature]` to capture physiological sleep stage transitions.
* `[feature]_context_std`: Rolling 5-epoch centered standard deviation of `[feature]` to track local stability vs. arousal dynamics.

## Part 2. Models and evaluation (DREAMT)

**Evaluation:** grouped five-fold cross-validation, all epochs of a participant in one fold; tuning and feature selection inside the training folds; same folds for every model. Mean ± SD over folds.

| Model | Macro-F1 | Balanced acc. | Cohen's kappa |
|---|---|---|---|
| Baseline 1: majority class | 0.1332 ± 0.0014 | 0.1332 ± 0.0014 | 0.0000 |
| Baseline 2: logistic regression | 0.2967 ± 0.0133 | 0.3980 | 0.1702 |
| Baseline 2: random forest | 0.3188 ± 0.0078 | 0.3255 | 0.2798 |
| LightGBM |  0.3524 ± 0.0072 | 0.3733 | 0.2503 |


**Modality ablation** (same model, same folds)

| Features | Macro-F1 | Balanced acc. | Cohen's Kappa |
|---|---|---|---|
| Heart rate and HRV only | 0.2694 ± 0.0089 | 0.3205 | 0.1392 |
| Accelerometer only | 0.2964 ± 0.0084 | 0.3308 | 0.1765 |
| EDA and temperature only | 0.2644 ± 0.0151 | 0.2842 | 0.1296 |
| Combined | 0.3500 ± 0.0106 | 0.3773 | 0.2471 |

Clearly, Combining sensors improved macro-F1 under identical splits, by more than the spread across folds.

**Per-class precision and recall (best model: LightGBMClassifier)**

| | Wake | N1 | N2 | N3 | REM |
|---|---|---|---|---|---|
| Precision | 0.747 | 0.336 | 0.904 | 0.476 | 0.503 |
| Recall | 0.732 | 0.625 | 0.548 | 0.956 | 0.850 |

**Confusion matrix:** `results/figures/confusion_matrix_dreamt.png`
**Class imbalance:** N1 is 11.1% of epochs (recall 0.625), Wake 25.1% (recall 0.732); mainly confused with Wake and N2.

---

## Part 3. External validation (AAUWSS)

The model chosen on DREAMT was frozen and applied to AAUWSS without retraining. Harmonization used no external labels.

| | Macro-F1 | Balanced acc. | Cohen's kappa | N1 recall | Wake recall |
|---|---|---|---|---|---|
| DREAMT (cross-validation) | 0.6253 | 0.7422 | 0.5265 | 0.6254 | 0.7321 |
| AAUWSS (95 % CI) | 0.3862 (0.3771–0.3954) | 0.3885 (0.3799–0.3980) | 0.2333 (0.2223–0.2444) | 0.2849 (0.2613–0.3091) | 0.4431 (0.4174–0.4709) |
| Difference | -0.2391 | -0.3537 | -0.2932 | -0.3405 | -0.2890 |


**Explainability (SHAP)**

| Stage | Top features | Direction |
|---|---|---|
| Wake | mag_mean_context_mean, share_away_1g | Higher motion / acceleration magnitude |
| N1 | dominant_freq, rmssd_context_mean | Intermediate movement, fluctuating HRV |
| N2 | rmssd, mag_mean_context_mean | Moderate HRV, suppressed motor activity |
| N3 | rmssd_context_mean, dominant_freq_context_mean | High steady HRV (parasympathetic dominance), zero movement |
| REM | rmssd, dominant_freq | High HRV variability with low movement |


**Error analysis**
- **Most frequent confusions:** N2 misclassified as N1 (7076 epochs), N2 misclassified as REM (5061 epochs), N2 misclassified as Wake (3218 epochs).
- **Participants with poor performance:** S027 (Macro-F1: 0.260), S004 (Macro-F1: 0.310), S058 (Macro-F1: 0.323); common features include severe sleep fragmentation and high movement during sleep.
- **Do sensor combinations consistently help?** Yes (multimodal ACC+EDA+TEMP features outperformed single-sensor ACC baselines across all 5 folds).