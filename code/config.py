from pathlib import Path

curr_dir = Path(__file__).parent
base_dir = curr_dir.parent
data_dir = base_dir / 'data'
temp_dir = data_dir / 'temp'
export_dir = data_dir / 'export_dir'
python_dir = data_dir / "python_output"
mdp_output_dir = data_dir / "mdp_output"

# MIMIC feature groups used in AIClinician_core.py
COLBIN = ["gender", "mechvent", "max_dose_vaso", "re_admission"]
COLNORM = [
    "age",
    "Weight_kg",
    "GCS",
    "HR",
    "SysBP",
    "MeanBP",
    "DiaBP",
    "RR",
    "Temp_C",
    "FiO2_1",
    "Potassium",
    "Sodium",
    "Chloride",
    "Glucose",
    "Magnesium",
    "Calcium",
    "Hb",
    "WBC_count",
    "Platelets_count",
    "PTT",
    "PT",
    "Arterial_pH",
    "paO2",
    "paCO2",
    "Arterial_BE",
    "HCO3",
    "Arterial_lactate",
    "SOFA",
    "SIRS",
    "Shock_Index",
    "PaO2_FiO2",
    "cumulated_balance",
]
COLLOG = [
    "SpO2",
    "BUN",
    "Creatinine",
    "SGOT",
    "SGPT",
    "Total_bili",
    "INR",
    "input_total",
    "input_4hourly",
    "output_total",
    "output_4hourly",
]
