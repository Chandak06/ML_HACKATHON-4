import logging,os,time,uuid
from contextlib import asynccontextmanager
from threading import Lock
from fastapi import FastAPI,HTTPException,Request
from fastapi.responses import JSONResponse
import numpy as np
from pydantic import BaseModel,Field
from src.config import ROOT
from src.data_loader import list_trips,load_trip,load_reference
from src.inference import Predictor,ModelArtifactError
from src.reference_alignment import align_reference
from src.evaluate_models import route_metrics
from src.visualizations import geometry
logger=logging.getLogger("uvicorn.error")
MODE=os.getenv("PATHSNAP_MODE","student_test")
if MODE not in ("student_test","evaluator_test"): raise RuntimeError("PATHSNAP_MODE must be student_test or evaluator_test")
@asynccontextmanager
async def lifespan(app):
    app.state.lock=Lock(); app.state.predictor=None; app.state.artifact_error=None
    try:
        app.state.predictor=Predictor()
        logger.info("PathSnap model loaded; mode=%s; available_trips=%d",MODE,len(list_trips(MODE)))
    except ModelArtifactError as exc:
        app.state.artifact_error=str(exc)
        logger.error("PathSnap inference unavailable: %s",exc)
    yield
app=FastAPI(title="PathSnap",version="1.0",lifespan=lifespan)
class PredictionRequest(BaseModel):
    trip_id: str=Field(pattern=r"^[0-9]{8}$",min_length=8,max_length=8)
@app.middleware("http")
async def request_log(request:Request,call_next):
    request_id=uuid.uuid4().hex[:10]; started=time.perf_counter()
    try:
        response=await call_next(request)
        logger.info("request=%s method=%s path=%s status=%s seconds=%.3f",request_id,request.method,request.url.path,response.status_code,time.perf_counter()-started)
        return response
    except Exception:
        logger.exception("request=%s failed path=%s",request_id,request.url.path); raise
@app.get("/health")
def health():
    ready=getattr(app.state,"predictor",None) is not None
    payload={"status":"ok" if ready else "degraded","mode":MODE,"model_loaded":ready,"available_trips":len(list_trips(MODE))}
    if not ready: payload["error"]=getattr(app.state,"artifact_error",None) or "Model has not loaded"
    return JSONResponse(payload,status_code=200 if ready else 503)
@app.get("/trips")
def trips():
    return {"mode":MODE,"trips":list_trips(MODE),"reserved_evaluator_data_present":MODE=="evaluator_test" and bool(list_trips(MODE))}
@app.post("/predict")
def predict(body:PredictionRequest):
    if getattr(app.state,"predictor",None) is None:
        raise HTTPException(503,"Model unavailable. Check /health and restore the saved inference artifacts.")
    if body.trip_id not in list_trips(MODE):
        logger.warning("PREDICT rejected trip=%s reason=unknown_trip",body.trip_id)
        raise HTTPException(404,"Unknown trip ID")
    logger.info("PREDICT start trip=%s mode=%s",body.trip_id,MODE)
    started=time.perf_counter()
    try:
        trip=load_trip(MODE,body.trip_id)
        with app.state.lock: result=app.state.predictor.predict(trip)
    except (ValueError,FileNotFoundError) as exc:
        logger.exception("Invalid trip data %s",body.trip_id)
        raise HTTPException(422,"Trip data are missing or invalid; see server log") from exc
    metrics=None; metric_status="Reference data unavailable"
    # References enter only after prediction is complete. Never serialize them.
    try:
        reference=load_reference(trip)
        labels,_=align_reference(trip,result["graph"],result["gps"],reference)
        metrics=route_metrics(result["pred"],labels,result["ids"],trip,reference,route=result["route"])
        metric_status="Server-side monotone reference-alignment proxy; not independently verified point labels"
    except (FileNotFoundError,ValueError):
        logger.warning("Reference unavailable/invalid for trip %s; prediction retained",body.trip_id)
    # Candidate-road corridor is derived solely from observations and the road
    # network. It contains no reference membership or labels.
    network_ids=np.unique(result["ids"][result["ids"]>=0])
    road_geometry=geometry(trip,network_ids)
    elapsed=time.perf_counter()-started
    logger.info("PREDICT trip=%s mode=%s observations=%d route_arcs=%d top1=%s inference_seconds=%.3f total_seconds=%.3f",
        body.trip_id,MODE,len(trip.track),len(result["route"]),metrics.get("top1_accuracy") if metrics else None,result["processing_seconds"],elapsed)
    return {"trip_id":body.trip_id,"mode":MODE,"predicted_arc_ids":result["pred"].tolist(),"predicted_route_arc_ids":result["route"],
        "predicted_geometry":geometry(trip,result["route"]),"gps":[trip.track[i,:2][::-1].tolist() if trip.valid[i] else None for i in range(len(trip.track))],
        "network_geometry":road_geometry,"network_segment_count":len(network_ids),
        "metrics":metrics,"metric_status":metric_status,"processing_seconds":result["processing_seconds"],"total_seconds":elapsed,
        "break_observation_indices":result["breaks"],"route_boundaries":result["boundaries"]}
if __name__=="__main__":
    import uvicorn
    uvicorn.run("app.api:app",host=os.getenv("PATHSNAP_HOST","0.0.0.0"),port=int(os.getenv("PATHSNAP_PORT","8000")),log_level="info",access_log=True)


