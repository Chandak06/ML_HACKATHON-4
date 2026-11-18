from pathlib import Path
import json, os
ROOT = Path(__file__).resolve().parents[1]
DATA = Path(os.getenv("PATHSNAP_DATA", ROOT / "data")).resolve()
SEED = 42
K = 8
FEATURES = ["distance_m","start_distance_m","end_distance_m","length_m","heading_difference","heading_valid","dt_s","speed_mps","in_degree","out_degree","distance_rank","projection_fraction","distance_margin_m","gps_outlier"]
def save_json(path, obj):
    path=Path(path); path.parent.mkdir(parents=True,exist_ok=True)
    path.write_text(json.dumps(obj,indent=2,default=lambda x:x.item() if hasattr(x,"item") else str(x)),encoding="utf8")

