import dataclasses,json,os
from pathlib import Path
import numpy as np
import pytest
from fastapi.testclient import TestClient
from src.config import ROOT
from src.data_loader import load_trip,list_trips
from src.inference import Predictor
from src.train_models import SPECS
from app.api import app
def test_artifacts_and_group_leakage():
    for name in SPECS: assert (ROOT/"artifacts"/f"{name}.joblib").exists()
    splits=json.loads((ROOT/"results/models/cross_validation.json").read_text())
    for split in splits:
        assert not set(split["training_trips"])&set(split["validation_trips"])
        assert len(set(split["training_trips"])|set(split["validation_trips"]))==80
    samples=json.loads((ROOT/"configs/training_samples.json").read_text())
    assert set(samples)==set(list_trips("train"))
    assert not set(samples)&set(list_trips("student_test"))
def test_inference_without_reference_access():
    trip=load_trip("student_test",list_trips("student_test")[0])
    trip=dataclasses.replace(trip,directory=ROOT/"no_reference_directory")
    result=Predictor().predict(trip)
    assert len(result["pred"])==len(trip.track)
    assert ((result["pred"]>=0)&(result["pred"]<len(trip.arcs))).all()
    assert all(0<=a<len(trip.arcs) for a in result["route"])
def test_fastapi_validation_and_prediction():
    with TestClient(app) as client:
        assert client.get("/health").json()["model_loaded"]
        ids=client.get("/trips").json()["trips"]; assert len(ids)==10
        assert client.post("/predict",json={"trip_id":"../../etc"}).status_code==422
        assert client.post("/predict",json={"trip_id":"99999999"}).status_code==404
        result=client.post("/predict",json={"trip_id":ids[0]})
        assert result.status_code==200
        body=result.json()
        assert body["predicted_geometry"] and body["metrics"]
        assert not any("reference" in key or "label" in key for key in body)
        assert 0<=body["metrics"]["top1_accuracy"]<=1
def test_reserved_mode_empty(monkeypatch):
    import app.api as api
    monkeypatch.setattr(api,"MODE","evaluator_test")
    with TestClient(app) as client:
        assert client.get("/trips").json()["trips"]==[]
        assert client.post("/predict",json={"trip_id":"00000000"}).status_code==404

