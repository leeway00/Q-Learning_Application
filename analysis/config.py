from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]

DATA_DIR = ROOT / "data"
MDP_OUTPUT_DIR = DATA_DIR / "mdp_output"
MDP_LOG_DIR = DATA_DIR / "mdp_log"

