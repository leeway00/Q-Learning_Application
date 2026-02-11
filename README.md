# Q-Learning Application

Current focus: replicating the AI Clinician core pipeline from MATLAB in Python.

## Project Status

- `Implemented`: Core AI Clinician model pipeline (MIMIC-only) in `code/AIClinician_core.py`
- `Implemented`: Core helper modules (`ai_utils.py`, `core_utils.py`, `evaluation.py`, `offpolicy.py`)
- `Implemented`: Run outputs, best-model recovery helpers, and analysis scaffolding in `analysis/`
- `In progress`: Statistical inference and full post-hoc statistical analysis over trained models
- `Not included`: eICU-dependent parts of the original MATLAB script (private data dependency)

## What Is Working Now

The Python core pipeline reproduces the main MATLAB training/evaluation flow:

1. Feature preprocessing/z-scoring
2. Train/test split by ICU stay
3. State clustering (750 states)
4. Action discretization (25 actions)
5. Transition/reward matrix construction
6. Policy iteration and Q reconstruction
7. Off-policy evaluation on MIMIC train/test
8. Saving run outputs and per-model policy payloads

## Main Entry Points

- Core run:
  - `code/AIClinician_core.py`
- Debug/plain script variant:
  - `code/temp_run_core.py`
- Analysis helpers:
  - `analysis/core_analysis.py` (best model selection and recovery)
  - `analysis/core_analysis_fig2.py` (MIMIC-only analysis figures)
  - `analysis/run_core_analysis.ipynb` (analysis notebook runner)

## Running

Install dependencies:

```bash
pip install -r requirements.txt
```

Run the core model:

```bash
python code/AIClinician_core.py --mimic-csv data/mimictable.csv --nr-reps 500
```

Run analysis notebook:

```bash
jupyter notebook analysis/run_core_analysis.ipynb
```

## Notes

- This repository contains legacy LD2Z experimentation code as well; the active replication work is under `code/` and `analysis/`.
- Statistical inference is not finalized yet. Current analysis modules are for model selection and figure-level diagnostics, not final inference claims.

## Reference

Komorowski et al. (2018), *Nature Medicine*:  
The Artificial Intelligence Clinician learns optimal treatment strategies for sepsis in intensive care.  
https://doi.org/10.1038/s41591-018-0213-5
