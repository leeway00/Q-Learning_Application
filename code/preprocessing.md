# AI Clinician MIMIC-III preprocessing (AIClinician_mimic3_dataset_160219.m)

This document summarizes the data processing performed by `refrepo/AI_Clinician/AIClinician_mimic3_dataset_160219.m` and the meaning of the variables it outputs. The goal is to make it easier to port the pipeline to Python and to understand what the core model consumes.

## Inputs assumed to exist
The script assumes the following variables/tables are already loaded (created by earlier steps such as `AIClinician_sepsis3_def_160219.m` and related preprocessing scripts):

- `sepsis`: cohort table with `icustayid` and `sepsis_time`.
- `ce010 ... ce90100`: chunked chartevents tables (by `icustayid` ranges).
- `labU`: labevents table (already unified / cleaned).
- `MV`: mechanical ventilation table (time‑stamped).
- `inputMV`, `inputCV`: fluid input tables (MetaVision and CareVue).
- `inputpreadm`: pre‑admission fluid volume.
- `vasoMV`, `vasoCV`: vasopressor infusion tables (MetaVision and CareVue).
- `UO`: urine output table.
- `UOpreadm`: pre‑admission urine output.
- `demog`: demographics/outcomes table.
- `sample_and_hold`: metadata for which variables use sample‑and‑hold.

From the repo README, these mappings and SAH metadata are stored in `reference_matrices.mat`, which includes:
- `Refvitals` and `Reflabs` (itemid‑to‑column mappings)
- `sample_and_hold` (SAH configuration)

## High‑level flow

1. **Create time‑stamped raw table (`reformat`)**
   - For each `icustayid` in `sepsis`, pull chart, lab, and mech‑vent rows in a wide time window around sepsis time.
   - Build a row per unique timestamp; store all available measurements from the three sources in fixed columns.
   - Save per‑stay sepsis time and first/last timestamps in `qstime`.

2. **Outlier filtering**
   - Apply `deloutabove` / `deloutbelow` for physiologic and lab ranges (weight, HR, BP, RR, SpO2, temp, FiO2, electrolytes, ABGs, lactate, etc).

3. **Value fixes and derivations (raw)**
   - Derive GCS from RASS if missing.
   - Convert / infer FiO2 from device and O2 flow (uses `SAH` with `sample_and_hold`).
   - Infer missing BP components (sys/mean/dia) from other components.
   - Fix temperature unit errors (C/F conversions).
   - Infer Hb/Hct and total/indirect bilirubin from each other.

4. **Sample‑and‑hold**
   - Apply `SAH` to `reformat` so sparse clinical measurements are carried forward between observations.

5. **Aggregate into 4‑hour bins (`reformat2`)**
   - For each stay, build 4‑hour windows starting from the first record in the per‑stay window (80 hours total).
   - Within each bin, compute mean values of chart + lab measurements.
   - Add vasopressor dose summaries (median and max) in the bin.
   - Add fluid inputs (cumulative and 4‑hourly) and urine outputs (cumulative and 4‑hourly).
   - Compute cumulative fluid balance.

6. **Select variables and clean**
   - Convert to table and keep curated columns (`dataheaders5`).
   - Correct gender coding, cap extreme age, binarize mech‑vent, fill missing Elixhauser, and make vasopressor columns non‑NaN.

7. **Handle missingness**
   - Linear interpolation for variables with <5% missingness.
   - kNN imputation (chunks of 10k rows) for remaining variables.

8. **Compute derived scores**
   - `PaO2_FiO2` and `Shock_Index` (with outlier handling).
   - SOFA and SIRS scores from the imputed values.

9. **Output**
   - Final table: `MIMICtable`.
   - Written to `mimictable.csv` and `step_4_start.mat`.

## How the sepsis‑3 cohort is created (AIClinician_sepsis3_def_160219.m)
This file prepares the sepsis cohort and creates some of the upstream structures used by the dataset builder.

1. **Load extracted CSVs** (abx, culture, microbio, demog, chart, labs, mechvent, fluids, vasopressors, urine output).
2. **Clean and align microbio/culture** tables and demographics.
3. **Compute normalized infusion rate** in `inputMV`.
4. **Fill missing icustay_id** in microbiology and antibiotics by matching admission windows.
5. **Presumed infection onset (`onset`)**:
   - If antibiotics precede cultures within 24h, onset = antibiotic time.
   - If cultures precede antibiotics within 72h, onset = culture time.
6. **Replace item_id with reference column indices** using `Refvitals` and `Reflabs` (from `reference_matrices.mat`).
7. **Build a raw, time‑stamped table (`reformat`)** for sepsis‑windowed data:
   - Window is around presumed infection time: **−48h to +24h** (plus a ±4h buffer in code).
8. **Outlier filtering, data fixes, SAH, and 4‑hour aggregation** follow (similar structure to the dataset builder).

## Time window details
- The script uses a wide time window around sepsis time:
  - `winb4 = 25`, `winaft = 49` (in hours), but it includes an extra `±4` hours in selection.
  - Effective filter is: `qst − (winb4+4)h` to `qst + (winaft+4)h`.
- Aggregation step uses **4‑hour bins** for 80 total hours per stay (from the first observed time in the selected window).

## Final output variables
These are the columns kept in the final table (from `dataheaders5`), grouped by category.

**Indexing / outcomes**
- `bloc`: 4‑hour block index (1,2,3...)
- `icustayid`: ICU stay id (offset removed)
- `charttime`: left boundary of the bin (epoch time)
- `gender`
- `age` (in days)
- `elixhauser`
- `re_admission`
- `died_in_hosp`
- `died_within_48h_of_out_time`
- `mortality_90d`
- `delay_end_of_record_and_discharge_or_death`

**Physiology / labs**
- `SOFA`, `SIRS`
- `Weight_kg`
- `GCS`
- `HR`
- `SysBP`, `MeanBP`, `DiaBP`
- `RR`
- `SpO2`
- `Temp_C`
- `FiO2_1` (fraction, not percent)
- `Potassium`
- `Sodium`
- `Chloride`
- `Glucose`
- `BUN`
- `Creatinine`
- `Magnesium`
- `Calcium`
- `Ionised_Ca`
- `CO2_mEqL` (bicarbonate)
- `SGOT`, `SGPT`
- `Total_bili`
- `Albumin`
- `Hb`
- `WBC_count`
- `Platelets_count`
- `PTT`
- `PT`
- `INR`
- `Arterial_pH`
- `paO2`
- `paCO2`
- `Arterial_BE`
- `HCO3`
- `Arterial_lactate`
- `mechvent` (0/1)

**Derived**
- `Shock_Index` (`HR / SysBP` with outlier handling)
- `PaO2_FiO2` (`paO2 / FiO2_1`)

**Treatments / outputs**
- `median_dose_vaso`
- `max_dose_vaso`
- `input_total` (cumulative fluid input)
- `input_4hourly` (fluid given in current 4‑hour bin)
- `output_total` (cumulative urine output)
- `output_4hourly` (urine output in current 4‑hour bin)
- `cumulated_balance` (`input_total − output_total`)

## Notes for Python port
- The data flow is **per‑stay**, then aggregated into fixed 4‑hour windows.
- The script assumes some derived columns are filled via **sample‑and‑hold** before aggregation.
- Imputation is performed **after** selecting the final columns and **before** SOFA/SIRS derivation.
- Shock Index and P/F are recomputed after imputation to remove NaNs and infinities.
