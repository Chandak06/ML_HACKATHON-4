import os,sys
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
import requests
import streamlit as st

from ui.client import ApiClient
from src.visualizations import route_map
st.set_page_config(page_title="PathSnap evaluator",layout="wide")
st.title("PathSnap")
st.caption("Distributed map matching and route reconstruction")
url=st.sidebar.text_input("Laptop 1 API address",os.getenv("PATHSNAP_API_URL","http://127.0.0.1:8000"))
try:
    client=ApiClient(url); available=client.trips()
except (requests.RequestException,ValueError) as exc:
    st.error(f"Cannot connect to Laptop 1: {exc}")
    st.info("Start the backend, check its Wi-Fi IPv4 address and port, then check the firewall.")
    st.stop()
mode=available["mode"]
if mode=="student_test": st.warning("STUDENT-TESTING DEVELOPMENT MODE — these are not reserved evaluator trips.")
else: st.info("Reserved evaluator mode — reference labels stay on Laptop 1.")
if not available["trips"]:
    st.info("No trips supplied for this mode. Place the authorised evaluator trip folders on Laptop 1 when provided.")
    st.stop()
if mode=="evaluator_test" and len(available["trips"])!=10: st.warning(f"{len(available['trips'])} evaluator trips available; expected ten when the full set is supplied.")
selected=st.selectbox("Choose a trip",available["trips"])
if st.button("Predict selected trip",type="primary"):
    try:
        with st.spinner("Laptop 1 is reconstructing the route…"): st.session_state.result=client.predict(selected)
    except (requests.RequestException,ValueError) as exc:
        st.error(f"Prediction request failed: {exc}")
        st.session_state.pop("result",None)
result=st.session_state.get("result")
if result and result["trip_id"]==selected and result["mode"]==mode:
    metrics=result["metrics"]
    a,b,c=st.columns(3)
    a.metric("Observation Top-1 accuracy",f"{metrics['top1_accuracy']:.2%}" if metrics and metrics["top1_accuracy"] is not None else "Unavailable")
    b.metric("Route continuity",f"{metrics['route_continuity']:.2%}" if metrics and metrics["route_continuity"] is not None else "Unavailable")
    c.metric("Processing time",f"{result['processing_seconds']:.2f} s")
    st.caption(result["metric_status"])
    st.write(f"Route breaks: {len(result['route_boundaries'])}")
    st.iframe(route_map(result["gps"],result["predicted_geometry"]).get_root().render(),height=620,alt="Predicted route and GPS observations")
    if metrics: st.json(metrics)


