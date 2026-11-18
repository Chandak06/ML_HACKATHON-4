from pathlib import Path
import json,sys
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
import numpy as np,pandas as pd
from src.config import ROOT,save_json
from src.data_loader import list_trips,load_trip
from src.preprocessing import project,haversine
rows=[]
for partition in ("train","student_test"):
    for trip_id in list_trips(partition):
        trip=load_trip(partition,trip_id); xy,_=project(trip)
        projected=np.linalg.norm(xy[trip.arcs[:,1]]-xy[trip.arcs[:,0]],axis=1)
        spherical=haversine(trip.nodes[trip.arcs[:,0]],trip.nodes[trip.arcs[:,1]])
        good=spherical>.01
        err=np.abs(projected[good]/spherical[good]-1)
        rows.append(dict(partition=partition,trip_id=trip_id,median_relative_error=float(np.median(err)),max_relative_error=float(err.max())))
pd.DataFrame(rows).to_csv(ROOT/"results/eda/projection_audit.csv",index=False)
cfg=json.loads((ROOT/"configs/features.json").read_text()); cfg["projection"]="local equirectangular WGS84 spherical radius 6371008.8 m; origin median road nodes"; save_json(ROOT/"configs/features.json",cfg)
print(pd.DataFrame(rows).groupby("partition")[["median_relative_error","max_relative_error"]].max())

