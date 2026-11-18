import argparse,json,time
import numpy as np,pandas as pd,joblib,optuna
from threadpoolctl import threadpool_limits
from .config import ROOT,FEATURES,SEED,save_json
from .train_models import training_data,cv_model,model_for,evaluate_student,SPECS
def search(trial,name):
    if name=="hist_gradient_boosting":
        return dict(max_iter=trial.suggest_int("max_iter",70,180),max_leaf_nodes=trial.suggest_int("max_leaf_nodes",10,45),
            learning_rate=trial.suggest_float("learning_rate",.025,.2,log=True),l2_regularization=trial.suggest_float("l2_regularization",.01,20,log=True),
            min_samples_leaf=trial.suggest_int("min_samples_leaf",15,80))
    if name=="random_forest": return dict(n_estimators=trial.suggest_int("n_estimators",60,160),max_depth=trial.suggest_int("max_depth",8,24),min_samples_leaf=trial.suggest_int("min_samples_leaf",2,30),max_features=trial.suggest_float("max_features",.4,1.))
    if name=="decision_tree": return dict(max_depth=trial.suggest_int("max_depth",4,24),min_samples_leaf=trial.suggest_int("min_samples_leaf",5,100))
    if name=="logistic": return dict(C=trial.suggest_float("C",.001,100,log=True))
    return dict(var_smoothing=trial.suggest_float("var_smoothing",1e-10,.01,log=True))
def run(trials):
    cfg=json.loads((ROOT/"configs/experiment.json").read_text()); name=json.loads((ROOT/"configs/selection.json").read_text())["selected_model"]
    data=training_data(cfg["max_observations_per_training_trip"])
    study=optuna.create_study(direction="maximize",sampler=optuna.samplers.TPESampler(seed=SEED,n_startup_trials=3),
        study_name="pathsnap")
    def objective(trial):
        start=time.perf_counter(); params=search(trial,name); cv=cv_model(name,params,data,cfg["folds"])
        trial.set_user_attr("seconds",time.perf_counter()-start); trial.set_user_attr("fold_metrics",cv)
        # Primary task quality with small probability-quality penalty.
        return float(np.mean([r["top1_accuracy"]-.02*r["log_loss"] for r in cv]))
    start=time.perf_counter()
    study.optimize(objective,n_trials=trials,callbacks=[lambda study,trial: save_json(ROOT/"results/tuning/trial_history.json",[{"number":t.number,"value":t.value,"params":t.params,"state":t.state.name,"attributes":t.user_attrs} for t in study.trials])])
    elapsed=time.perf_counter()-start
    study.trials_dataframe().to_csv(ROOT/"results/tuning/trials.csv",index=False)
    best=study.best_trial; params=best.params; model=model_for(name,params)
    start=time.perf_counter()
    with threadpool_limits(limits=2): model.fit(data[0].reshape(-1,len(FEATURES)),data[1].ravel())
    train_seconds=time.perf_counter()-start
    joblib.dump(model,ROOT/"artifacts/tuned_model.joblib",compress=3)
    metrics,rows=evaluate_student(model,name+"_tuned"); pd.DataFrame(rows).to_csv(ROOT/"results/tuning/per_trip.csv",index=False)
    cv=best.user_attrs["fold_metrics"]
    baseline=json.loads((ROOT/"results/models"/f"{name}.json").read_text())
    base_objective=np.mean([r["top1_accuracy"]-.02*r["log_loss"] for r in baseline["cv"]])
    deploy_tuned=best.value>=base_objective
    chosen=model if deploy_tuned else joblib.load(ROOT/"artifacts"/f"{name}.joblib")
    joblib.dump(chosen,ROOT/"artifacts/best_model.joblib",compress=3)
    summary=dict(model=name,actual_trials=len(study.trials),tpe_startup_trials=3,bayesian_trials=max(len(study.trials)-3,0),
        best_parameters=params,best_objective=best.value,baseline_objective=base_objective,
        objective="mean group-CV (Top-1 accuracy - 0.02 * candidate log loss)",cv=cv,student_test=metrics,
        search_seconds=elapsed,refit_seconds=train_seconds,deployed="tuned" if deploy_tuned else "baseline")
    save_json(ROOT/"results/tuning/summary.json",summary)
    save_json(ROOT/"configs/deployment.json",dict(model=name,variant=summary["deployed"],parameters=SPECS[name][2]|(params if deploy_tuned else {}),features=FEATURES,
        alignment_metrics_are_proxies=True,hmm=dict(beta_m=30.,max_gap_s=30.,max_path_m=3000.)))
    comparison=[]
    for variant,result,cross in [("baseline",baseline["student_test"],baseline["cv"]),("tuned",metrics,cv)]:
        comparison.append(dict(variant=variant,cv_macro_f1=np.mean([r["macro_f1"] for r in cross]),cv_top1_accuracy=np.mean([r["top1_accuracy"] for r in cross]),
            **{k:v for k,v in result.items() if k!="confusion_matrix"}))
    pd.DataFrame(comparison).to_csv(ROOT/"results/tuning/comparison.csv",index=False)
    print(json.dumps(summary,indent=2),flush=True)
if __name__=="__main__":
    p=argparse.ArgumentParser(); p.add_argument("--trials",type=int,default=10); a=p.parse_args()
    if a.trials<4: p.error("Use at least four trials so TPE runs beyond startup.")
    run(a.trials)


