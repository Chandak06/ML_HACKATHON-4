"""Real socket requests through the exact HTTP client used by the UI."""
import argparse,json,time,sys
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from ui.client import ApiClient
from src.config import ROOT,save_json
p=argparse.ArgumentParser(); p.add_argument("--url",default="http://127.0.0.1:8000"); a=p.parse_args()
client=ApiClient(a.url); assert client.health()["model_loaded"]
trips=client.trips(); assert trips["mode"]=="student_test" and len(trips["trips"])==10
rows=[]
for trip_id in trips["trips"]:
    start=time.perf_counter(); body=client.predict(trip_id)
    assert body["trip_id"]==trip_id and body["predicted_geometry"]
    assert len(body["predicted_arc_ids"])==len(body["gps"])
    assert not any("reference" in k or "label" in k for k in body)
    rows.append(dict(trip_id=trip_id,top1_accuracy=body["metrics"]["top1_accuracy"],http_seconds=time.perf_counter()-start,processing_seconds=body["processing_seconds"]))
    print(json.dumps(rows[-1]),flush=True)
save_json(ROOT/"results/logs/http_smoke.json",{"url":a.url,"trips":rows,"cross_laptop_verified":False})

