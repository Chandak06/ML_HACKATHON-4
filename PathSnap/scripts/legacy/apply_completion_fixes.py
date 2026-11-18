from pathlib import Path
ROOT=Path(__file__).resolve().parents[1]
def edit(name,old,new):
    p=ROOT/name; s=p.read_text(encoding="utf-8-sig")
    if old not in s: raise RuntimeError(f"Expected source not found: {name}")
    p.write_text(s.replace(old,new),encoding="utf8")
edit("src/inference.py","import json,time,joblib","import json,time,joblib\nfrom pathlib import Path\nimport numpy as np")
edit("src/inference.py","from .config import ROOT","from .config import ROOT,FEATURES")
edit("src/inference.py",'class Predictor:\n    def __init__(self,model_path=None):\n        self.model=joblib.load(model_path or ROOT/"artifacts/best_model.joblib")\n        self.config=json.loads((ROOT/"configs/deployment.json").read_text())', '''
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
'''.strip())
edit("app/api.py","from fastapi import FastAPI,HTTPException,Request","from fastapi import FastAPI,HTTPException,Request\nfrom fastapi.responses import JSONResponse\nimport numpy as np")
edit("app/api.py","from src.inference import Predictor","from src.inference import Predictor,ModelArtifactError")
edit("app/api.py",'    app.state.predictor=Predictor(); app.state.lock=Lock()\n    logger.info("PathSnap model loaded; mode=%s; available_trips=%d",MODE,len(list_trips(MODE)))','''    app.state.lock=Lock(); app.state.predictor=None; app.state.artifact_error=None
    try:
        app.state.predictor=Predictor()
        logger.info("PathSnap model loaded; mode=%s; available_trips=%d",MODE,len(list_trips(MODE)))
    except ModelArtifactError as exc:
        app.state.artifact_error=str(exc)
        logger.error("PathSnap inference unavailable: %s",exc)''')
edit("app/api.py",'    return {"status":"ok","mode":MODE,"model_loaded":hasattr(app.state,"predictor"),"available_trips":len(list_trips(MODE))}', '''    ready=getattr(app.state,"predictor",None) is not None
    payload={"status":"ok" if ready else "degraded","mode":MODE,"model_loaded":ready,"available_trips":len(list_trips(MODE))}
    if not ready: payload["error"]=getattr(app.state,"artifact_error",None) or "Model has not loaded"
    return JSONResponse(payload,status_code=200 if ready else 503)''')
edit("app/api.py",'    if body.trip_id not in list_trips(MODE): raise HTTPException(404,"Unknown trip ID")', '''    if getattr(app.state,"predictor",None) is None:
        raise HTTPException(503,"Model unavailable. Check /health and restore the saved inference artifacts.")
    if body.trip_id not in list_trips(MODE):
        logger.warning("PREDICT rejected trip=%s reason=unknown_trip",body.trip_id)
        raise HTTPException(404,"Unknown trip ID")
    logger.info("PREDICT start trip=%s mode=%s",body.trip_id,MODE)''')
edit("app/api.py",'    elapsed=time.perf_counter()-started','''    # Candidate-road corridor is derived solely from observations and the road
    # network. It contains no reference membership or labels.
    network_ids=np.unique(result["ids"][result["ids"]>=0])
    road_geometry=geometry(trip,network_ids)
    elapsed=time.perf_counter()-started''')
edit("app/api.py",'        "metrics":metrics,"metric_status":metric_status,','        "network_geometry":road_geometry,"network_segment_count":len(network_ids),\n        "metrics":metrics,"metric_status":metric_status,')
edit("ui/app.py",'    client=ApiClient(url); available=client.trips()', '''    client=ApiClient(url); health=client.health()
    if not health.get("model_loaded") or health.get("status")!="ok":
        st.error("Backend is reachable but the model is not ready. Check Laptop 1's /health endpoint and terminal.")
        st.stop()
    available=client.trips()
    st.sidebar.success("Backend connected - model ready")''')
edit("ui/app.py",'    st.stop()\nmode=available["mode"]', '''    st.stop()
source=client.url
if st.session_state.get("result_source")!=source:
    st.session_state.pop("result",None)
mode=available["mode"]''')
edit("ui/app.py",'        with st.spinner("Laptop 1 is reconstructing the route…"): st.session_state.result=client.predict(selected)', '''        with st.spinner("Laptop 1 is reconstructing the route…"):
            st.session_state.result=client.predict(selected)
            st.session_state.result_source=source''')
edit("ui/app.py",'route_map(result["gps"],result["predicted_geometry"])','route_map(result["gps"],result["predicted_geometry"],network=result.get("network_geometry"))')
edit("src/visualizations.py",'zoom_start=13,tiles="OpenStreetMap"','zoom_start=13,tiles="OpenStreetMap",prefer_canvas=True')
edit("src/visualizations.py",'name="Road network",show=False','name="Road network",show=True')
# Keep old audit evidence intact: allow current tests to choose new destinations.
edit("scripts/check_ui.py","import sys","import sys,os")
edit("scripts/check_ui.py",'save_json(ROOT/"results/logs/ui_smoke.json",','save_json(Path(os.getenv("PATHSNAP_UI_EVIDENCE",str(ROOT/"results/logs/ui_smoke.json"))),')
edit("scripts/check_ui.py",'"real_http":"http://127.0.0.1:8000"','"real_http":os.getenv("PATHSNAP_API_URL","http://127.0.0.1:8000")')
edit("scripts/check_ui.py",'assert app.session_state["result"]["predicted_geometry"]','assert app.session_state["result"]["predicted_geometry"]\nassert app.session_state["result"]["network_geometry"]\nassert app.sidebar.success')
edit("scripts/smoke_http.py",'p.add_argument("--url",default="http://127.0.0.1:8000"); a=p.parse_args()','p.add_argument("--url",default="http://127.0.0.1:8000"); p.add_argument("--output",default=str(ROOT/"results/logs/http_smoke.json")); a=p.parse_args()')
edit("scripts/smoke_http.py",'assert body["trip_id"]==trip_id and body["predicted_geometry"]','assert body["trip_id"]==trip_id and body["predicted_geometry"] and body["network_geometry"]')
edit("scripts/smoke_http.py",'save_json(ROOT/"results/logs/http_smoke.json",','save_json(Path(a.output),')
print("Applied audited fixes; models, training, features, decoder and experiment outputs untouched.")

