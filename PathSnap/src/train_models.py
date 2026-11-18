import argparse,time,platform,importlib.metadata
from pathlib import Path
import joblib,numpy as np,pandas as pd
from threadpoolctl import threadpool_limits
from sklearn.model_selection import GroupKFold
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import GaussianNB
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier,HistGradientBoostingClassifier
from .config import ROOT,SEED,FEATURES,save_json
from .data_loader import load_trip,load_reference,list_trips
from .evaluate_models import candidate_metrics,score,top1,route_metrics
SPECS={
 "logistic":("Linear logistic regression","binary log loss + L2",dict(C=1.,max_iter=500,random_state=SEED)),
 "naive_bayes":("Gaussian naive Bayes","Gaussian class conditional likelihood",dict(var_smoothing=1e-5)),
 "decision_tree":("Decision tree","Gini impurity",dict(max_depth=12,min_samples_leaf=20,random_state=SEED)),
 "random_forest":("Bagged random forest","averaged Gini decision trees",dict(n_estimators=100,max_depth=18,min_samples_leaf=5,n_jobs=2,random_state=SEED)),
 "hist_gradient_boosting":("Histogram gradient boosting","binary log loss",dict(max_iter=100,max_leaf_nodes=31,learning_rate=.1,l2_regularization=1.,early_stopping=False,random_state=SEED))}
def model_for(name,params=None):
    settings=SPECS[name][2]|(params or {})
    cls={"logistic":LogisticRegression,"naive_bayes":GaussianNB,"decision_tree":DecisionTreeClassifier,"random_forest":RandomForestClassifier,"hist_gradient_boosting":HistGradientBoostingClassifier}[name]
    model=cls(**settings)
    return make_pipeline(StandardScaler(),model) if name in ("logistic","naive_bayes") else model
def training_data(max_observations=400):
    xs=[]; ys=[]; groups=[]; ids_all=[]; labels_all=[]; sample_manifest={}
    for trip_id in list_trips("train"):
        with np.load(ROOT/"artifacts/features/train"/f"{trip_id}.npz") as z:
            eligible=np.flatnonzero(z["valid"]&(z["residual"]<=80)&(z["ids"]>=0).all(axis=1))
            idx=eligible[np.unique(np.linspace(0,len(eligible)-1,min(len(eligible),max_observations)).astype(int))]
            if not len(idx): raise ValueError(f"No usable observations in {trip_id}")
            ids=z["ids"][idx]; labels=z["labels"][idx]
            xs.append(z["x"][idx]); ys.append(ids==labels[:,None]); groups.extend([trip_id]*len(idx))
            ids_all.append(ids); labels_all.append(labels); sample_manifest[trip_id]=idx.tolist()
    return np.concatenate(xs),np.concatenate(ys),np.asarray(groups),np.concatenate(ids_all),np.concatenate(labels_all),sample_manifest
def cv_model(name,params,data,folds=3):
    x,y,groups,ids,labels,_=data; rows=[]
    for fold,(train,val) in enumerate(GroupKFold(folds).split(x,y,groups)):
        assert not set(groups[train])&set(groups[val])
        model=model_for(name,params); start=time.perf_counter()
        with threadpool_limits(limits=2):
            model.fit(x[train].reshape(-1,x.shape[-1]),y[train].ravel())
            train_seconds=time.perf_counter()-start
            p,inference=score(model,x[val],ids[val])
        row=candidate_metrics(y[val],p); row.pop("confusion_matrix")
        row.update(fold=fold,top1_accuracy=float(np.mean(top1(ids[val],p)==labels[val])),train_seconds=train_seconds,inference_seconds=inference,
                   training_trips=sorted(set(groups[train])),validation_trips=sorted(set(groups[val])))
        rows.append(row)
    return rows
def evaluate_student(model,name):
    rows=[]; all_y=[]; all_p=[]; total_time=0
    for trip_id in list_trips("student_test"):
        trip=load_trip("student_test",trip_id); reference=load_reference(trip)
        with np.load(ROOT/"artifacts/features/student_test"/f"{trip_id}.npz") as z:
            with threadpool_limits(limits=2): p,seconds=score(model,z["x"],z["ids"])
            pred=top1(z["ids"],p); valid=z["valid"]
            mask=(z["ids"]>=0)&valid[:,None]
            all_y.extend((z["ids"]==z["labels"][:,None])[mask]); all_p.extend(p[mask])
            rows.append(dict(model=name,trip_id=trip_id,inference_seconds=seconds,**route_metrics(pred,z["labels"],z["ids"],trip,reference)))
            total_time+=seconds
    metrics=candidate_metrics(all_y,all_p)
    frame=pd.DataFrame(rows)
    metrics.update(top1_accuracy=float(np.average(frame.top1_accuracy,weights=frame.observations)),
        candidate_miss_rate=float(np.average(frame.candidate_miss_rate,weights=frame.observations)),
        route_continuity=float(np.average(frame.route_continuity,weights=frame.valid_transition_pairs)),
        sequence_edit_similarity=float(frame.sequence_edit_similarity.mean()),inference_seconds=total_time)
    return metrics,rows
def run(folds=3,max_observations=400):
    data=training_data(max_observations)
    save_json(ROOT/"configs/training_samples.json",data[-1])
    save_json(ROOT/"configs/experiment.json",dict(seed=SEED,folds=folds,max_observations_per_training_trip=max_observations,training_observations=len(data[0]),training_candidate_rows=int(data[1].size),training_trips=80,test_trips=10,features=FEATURES,python=platform.python_version(),packages={p:importlib.metadata.version(p) for p in ["numpy","pandas","scipy","scikit-learn","optuna"]}))
    comparison=[]; per_trip=[]; cv_all=[]
    for name,(family,objective,params) in SPECS.items():
        print("Training",name,flush=True); cv=cv_model(name,None,data,folds); cv_all.extend(dict(model=name,**r) for r in cv)
        model=model_for(name); start=time.perf_counter()
        with threadpool_limits(limits=2): model.fit(data[0].reshape(-1,len(FEATURES)),data[1].ravel())
        train_time=time.perf_counter()-start
        joblib.dump(model,ROOT/"artifacts"/f"{name}.joblib",compress=3)
        metrics,trips=evaluate_student(model,name); per_trip.extend(trips)
        save_json(ROOT/"results/models"/f"{name}.json",dict(family=family,objective=objective,parameters=params,cv=cv,student_test=metrics,train_seconds=train_time))
        metrics.pop("confusion_matrix")
        comparison.append(dict(model=name,family=family,cv_macro_f1=np.mean([r["macro_f1"] for r in cv]),cv_macro_f1_std=np.std([r["macro_f1"] for r in cv]),
            cv_top1_accuracy=np.mean([r["top1_accuracy"] for r in cv]),cv_log_loss=np.mean([r["log_loss"] for r in cv]),train_seconds=train_time,**metrics))
        pd.DataFrame(comparison).to_csv(ROOT/"results/models/comparison.csv",index=False)
        pd.DataFrame(per_trip).to_csv(ROOT/"results/models/per_trip.csv",index=False)
        print(comparison[-1],flush=True)
    save_json(ROOT/"results/models/cross_validation.json",cv_all)
    frame=pd.DataFrame(comparison)
    # Match quality primary; among models within 0.5 percentage points use
    # CV probability quality then measured student inference time.
    pool=frame[frame.cv_top1_accuracy>=frame.cv_top1_accuracy.max()-.005]
    chosen=pool.sort_values(["cv_log_loss","inference_seconds"]).iloc[0].model
    save_json(ROOT/"configs/selection.json",dict(selected_model=chosen,rule="CV Top-1 within 0.005 of maximum, then lowest CV log loss, then inference time",eligible_models=pool.model.tolist()))
    print("Selected:",chosen,flush=True)
    from .visualizations import model_plots
    model_plots()
if __name__=="__main__":
    p=argparse.ArgumentParser(); p.add_argument("--folds",type=int,default=3); p.add_argument("--max-observations",type=int,default=400)
    a=p.parse_args(); run(a.folds,a.max_observations)

