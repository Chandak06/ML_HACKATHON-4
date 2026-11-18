import argparse,time
import numpy as np
import pandas as pd
from .config import ROOT,FEATURES,K,save_json
from .data_loader import load_trip,load_reference,list_trips
from .preprocessing import project
from .road_graph import RoadGraph
from .candidate_generation import candidates
from .feature_engineering import features
from .reference_alignment import align_reference
def run():
    records=[]; summaries=[]
    train_ids=list_trips("train"); test_ids=list_trips("student_test")
    assert len(train_ids)==80 and len(test_ids)==10 and not set(train_ids)&set(test_ids)
    for partition in ("train","student_test"):
        for trip_id in list_trips(partition):
            started=time.perf_counter()
            trip=load_trip(partition,trip_id); reference=load_reference(trip)
            nodes,gps=project(trip); graph=RoadGraph(trip,nodes)
            ids,d,f=candidates(graph,gps,trip.valid); x=features(trip,graph,gps,ids,d,f)
            labels,residual=align_reference(trip,graph,gps,reference)
            path=ROOT/"artifacts/features"/partition/f"{trip_id}.npz"; path.parent.mkdir(parents=True,exist_ok=True)
            np.savez_compressed(path,x=x,ids=ids,fractions=f,labels=labels,residual=residual,valid=trip.valid)
            dt=np.diff(trip.track[:,2]); steps=np.linalg.norm(np.diff(gps,axis=0),axis=1)
            r=dict(partition=partition,trip_id=trip_id,observations=len(gps),nodes=len(nodes),arcs=len(trip.arcs),reference_arcs=len(reference),
                invalid_gps=int((~trip.valid).sum()),missing_values=int((~np.isfinite(trip.track)).sum()),invalid_nodes=0,invalid_arc_endpoints=0,
                duplicate_timestamps=int((dt==0).sum()),reversed_timestamps=int((dt<0).sum()),gaps_over_30s=int((dt>30).sum()),
                median_interval_s=float(np.nanmedian(dt)),max_interval_s=float(np.nanmax(dt)),one_second_steps=int((dt==1).sum()),
                trajectory_length_m=float(np.nansum(steps)),weak_components=int(graph.components),strong_components=int(graph.strong_components),
                isolated_nodes=int(((graph.indegree+graph.outdegree)==0).sum()),zero_length_arcs=int((graph.length<1e-6).sum()),
                longitude_min=float(np.nanmin(trip.track[:,0])),longitude_max=float(np.nanmax(trip.track[:,0])),
                latitude_min=float(np.nanmin(trip.track[:,1])),latitude_max=float(np.nanmax(trip.track[:,1])),
                reference_breaks=int((trip.arcs[reference[:-1],1]!=trip.arcs[reference[1:],0]).sum()),
                alignment_median_m=float(np.median(residual[trip.valid])),alignment_p95_m=float(np.percentile(residual[trip.valid],95)),
                alignment_over_80m=int((residual[trip.valid]>80).sum()),candidate_miss_rate=float(np.mean(~(ids[trip.valid]==labels[trip.valid,None]).any(axis=1))),
                build_seconds=time.perf_counter()-started)
            records.append(r)
            for j,name in enumerate(FEATURES):
                values=x[:,:,j][ids>=0]
                summaries.append(dict(partition=partition,trip_id=trip_id,feature=name,mean=float(values.mean()),std=float(values.std()),min=float(values.min()),max=float(values.max())))
            print(partition,trip_id,"observations",len(gps),"miss",round(r["candidate_miss_rate"],4),"seconds",round(r["build_seconds"],2),flush=True)
    pd.DataFrame(records).to_csv(ROOT/"results/eda/trips.csv",index=False)
    pd.DataFrame(summaries).to_csv(ROOT/"results/eda/features.csv",index=False)
    save_json(ROOT/"configs/features.json",dict(features=FEATURES,candidates=K,projection="local equirectangular spherical projection, metres; see projection_audit.csv",alignment="monotone route-occurrence DP; geometry and heading; proxy labels",training_max_alignment_m=80))
    save_json(ROOT/"results/eda/summary.json",{"partitions":pd.DataFrame(records).groupby("partition")[["observations","invalid_gps","missing_values","duplicate_timestamps","reversed_timestamps","gaps_over_30s","alignment_over_80m"]].sum().to_dict("index"),"trip_counts":{"train":80,"student_test":10},"evaluator_used":False})
if __name__=="__main__": run()


