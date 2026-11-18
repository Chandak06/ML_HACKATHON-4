import numpy as np,pandas as pd
from .config import ROOT,save_json
from .data_loader import load_trip,load_reference,list_trips
from .inference import Predictor
from .reference_alignment import align_reference
from .evaluate_models import route_metrics
from .visualizations import route_map,geometry
def run():
    predictor=Predictor(); rows=[]
    for trip_id in list_trips("student_test"):
        trip=load_trip("student_test",trip_id); result=predictor.predict(trip)
        reference=load_reference(trip)
        labels,residual=align_reference(trip,result["graph"],result["gps"],reference)
        for kind,pred in [("classifier",result["classifier_pred"]),("hmm",result["pred"])]:
            metrics=route_metrics(pred,labels,result["ids"],trip,reference,route=result["route"] if kind=="hmm" else None)
            rows.append(dict(trip_id=trip_id,method=kind,**metrics,classifier_seconds=result["classifier_seconds"],hmm_seconds=result["hmm_seconds"],processing_seconds=result["processing_seconds"],breaks=len(result["breaks"]),route_breaks=len(result["boundaries"])))
        save_json(ROOT/"results/hmm"/f"{trip_id}.json",dict(predictions=result["pred"].tolist(),classifier_predictions=result["classifier_pred"].tolist(),route=result["route"],breaks=result["breaks"],route_boundaries=result["boundaries"]))
        if trip_id==list_trips("student_test")[0]:
            route_map(trip.track[trip.valid,:2][:,::-1].tolist(),geometry(trip,result["route"]),
                geometry(trip,np.arange(len(trip.arcs))),geometry(trip,reference)).save(str(ROOT/"results/maps"/f"{trip_id}_offline.html"))
        pd.DataFrame(rows).to_csv(ROOT/"results/hmm/per_trip.csv",index=False)
        print(trip_id,rows[-2]["top1_accuracy"],rows[-1]["top1_accuracy"],"seconds",result["processing_seconds"],flush=True)
    frame=pd.DataFrame(rows); summary={}
    for method,part in frame.groupby("method"):
        summary[method]=dict(top1_accuracy=float(np.average(part.top1_accuracy,weights=part.observations)),
            route_continuity=float(np.average(part.route_continuity,weights=part.valid_transition_pairs)),
            sequence_edit_similarity=float(part.sequence_edit_similarity.mean()),processing_seconds=float(part.processing_seconds.sum()))
    save_json(ROOT/"results/hmm/summary.json",summary); print(summary,flush=True)
    import matplotlib.pyplot as plt
    fig,axes=plt.subplots(1,3,figsize=(12,4))
    for ax,key in zip(axes,["top1_accuracy","route_continuity","sequence_edit_similarity"]):
        ax.bar(list(summary),[summary[m][key] for m in summary]); ax.set_title(key.replace("_"," ")); ax.set_ylim(0,1)
    fig.tight_layout(); fig.savefig(ROOT/"results/hmm/comparison.png",dpi=150); plt.close(fig)
if __name__=="__main__": run()

