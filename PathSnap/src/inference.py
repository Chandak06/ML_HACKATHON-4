import json,time,joblib
from pathlib import Path
import numpy as np
from threadpoolctl import threadpool_limits
from .config import ROOT,FEATURES
from .preprocessing import project
from .road_graph import RoadGraph
from .candidate_generation import candidates
from .feature_engineering import features
from .evaluate_models import score,top1
from .hmm_viterbi import decode,reconstruct
class ModelArtifactError(RuntimeError):
    """A missing, unreadable or incompatible inference artifact."""

class Predictor:
    def __init__(self,model_path=None,config_path=None):
        model_path=Path(model_path or ROOT/"artifacts/best_model.joblib")
        config_path=Path(config_path or ROOT/"configs/deployment.json")
        for path in (model_path,config_path):
            if not path.is_file():
                raise ModelArtifactError(f"Missing inference artifact: {path.name}. Restore the saved artifacts and configuration; retraining is not required.")
        try:
            self.config=json.loads(config_path.read_text(encoding="utf-8-sig"))
            if self.config.get("features")!=FEATURES:
                raise ValueError("Saved feature names/order differ from the current feature pipeline")
            hmm=self.config["hmm"]
            if set(hmm)!={"beta_m","max_gap_s","max_path_m"} or any(not np.isfinite(v) or v<=0 for v in hmm.values()):
                raise ValueError("Invalid HMM configuration")
            self.model=joblib.load(model_path)
            if not callable(getattr(self.model,"predict_proba",None)) or self.model.n_features_in_!=len(FEATURES):
                raise ValueError("Model feature count or probability interface is incompatible")
            if not np.array_equal(self.model.classes_,[0,1]):
                raise ValueError("Model must expose binary candidate classes [0, 1]")
            estimator=self.model.steps[-1][1] if hasattr(self.model,"steps") else self.model
            actual=estimator.get_params()
            if any(actual.get(k)!=v for k,v in self.config.get("parameters",{}).items()):
                raise ValueError("Model parameters differ from the saved deployment configuration")
        except Exception as exc:
            raise ModelArtifactError(f"Cannot load compatible model/configuration: {exc}") from exc
    def predict(self,trip):
        start=time.perf_counter()
        nodes,gps=project(trip); graph=RoadGraph(trip,nodes)
        ids,d,f=candidates(graph,gps,trip.valid); x=features(trip,graph,gps,ids,d,f)
        with threadpool_limits(limits=2): p,classifier_seconds=score(self.model,x,ids)
        raw=top1(ids,p)
        hmm_start=time.perf_counter(); matched,breaks=decode(trip,graph,gps,ids,f,p,**self.config["hmm"])
        route,boundaries=reconstruct(trip,graph,matched,breaks,self.config["hmm"]["max_path_m"])
        return dict(pred=matched,classifier_pred=raw,route=route,boundaries=boundaries,breaks=breaks,
            ids=ids,graph=graph,gps=gps,classifier_seconds=classifier_seconds,
            hmm_seconds=time.perf_counter()-hmm_start,processing_seconds=time.perf_counter()-start)

