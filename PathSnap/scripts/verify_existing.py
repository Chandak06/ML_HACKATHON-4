"""Audit existing evidence and models; never fit or overwrite experiment outputs."""
from pathlib import Path
import sys,json,hashlib,time
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
import joblib,numpy as np,pandas as pd
from src.config import ROOT,FEATURES,save_json
from src.train_models import SPECS,evaluate_student
from src.data_loader import list_trips,load_trip
from src.preprocessing import project
from src.road_graph import RoadGraph
from src.candidate_generation import candidates
from src.feature_engineering import features
OUT=ROOT/"results/completion_audit"; OUT.mkdir(parents=True,exist_ok=True)
def evidence_hashes():
    paths=list((ROOT/"artifacts").glob("*.joblib"))
    for directory in ["results/models","results/tuning","results/hmm","results/eda"]:
        paths.extend(p for p in (ROOT/directory).rglob("*") if p.is_file())
    return {p.relative_to(ROOT).as_posix():hashlib.sha256(p.read_bytes()).hexdigest() for p in paths}
if __name__=="__main__":
    before=evidence_hashes()
    assert len(list_trips("train"))==80 and len(list_trips("student_test"))==10
    assert not set(list_trips("train"))&set(list_trips("student_test"))
    history=json.loads((ROOT/"results/tuning/trial_history.json").read_text())
    summary=json.loads((ROOT/"results/tuning/summary.json").read_text())
    assert len(history)==summary["actual_trials"]==10
    best=max(history,key=lambda t:t["value"])
    assert best["params"]==summary["best_parameters"] and np.isclose(best["value"],summary["best_objective"])
    for trial in history:
        assert trial["state"]=="COMPLETE"
        folds=trial["attributes"]["fold_metrics"]
        for fold in folds:
            assert not set(fold["training_trips"])&set(fold["validation_trips"])
            assert set(fold["training_trips"])|set(fold["validation_trips"])==set(list_trips("train"))
        assert np.isclose(trial["value"],np.mean([f["top1_accuracy"]-.02*f["log_loss"] for f in folds]))
    rows=[]
    for name in list(SPECS)+["tuned_model","best_model"]:
        path=ROOT/"artifacts"/f"{name}.joblib"; model=joblib.load(path)
        expected=json.loads((ROOT/"results/models"/f"{name}.json").read_text()) if name in SPECS else summary
        estimator=model.steps[-1][1] if hasattr(model,"steps") else model
        params=expected.get("parameters",expected.get("best_parameters"))
        assert all(estimator.get_params()[k]==v for k,v in params.items())
        assert model.n_features_in_==len(FEATURES)
        metrics,_=evaluate_student(model,name)
        for key,value in expected["student_test"].items():
            if key=="inference_seconds": continue
            np.testing.assert_allclose(metrics[key],value,rtol=1e-6,atol=1e-8,err_msg=name+" "+key)
        rows.append(dict(model=name,top1=metrics["top1_accuracy"],macro_f1=metrics["macro_f1"],log_loss=metrics["log_loss"],matches_saved=True))
        print(name,"saved metrics reproduced",flush=True)
    trip=load_trip("student_test",list_trips("student_test")[0])
    xy,gps=project(trip); graph=RoadGraph(trip,xy)
    ids,d,f=candidates(graph,gps,trip.valid); x=features(trip,graph,gps,ids,d,f)
    with np.load(ROOT/"artifacts/features/student_test"/f"{trip.trip_id}.npz") as cache:
        np.testing.assert_array_equal(ids,cache["ids"]); np.testing.assert_array_equal(x,cache["x"])
    assert evidence_hashes()==before
    save_json(OUT/"existing_evidence.json",dict(models=rows,feature_cache_exact_match=True,feature_names=FEATURES,tuning_trials=len(history),best_trial=best["number"],trip_counts={"train":80,"student_test":10},evidence_hashes=before,no_retraining=True))
    print("All metrics reproduced; feature cache matches inference; original evidence hashes unchanged.")

