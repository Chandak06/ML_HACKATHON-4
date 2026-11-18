"""Strict whitespace loaders; row positions are immutable identifiers."""
from dataclasses import dataclass
from pathlib import Path
import re
import numpy as np
from .config import DATA
@dataclass
class Trip:
    trip_id: str
    track: np.ndarray
    nodes: np.ndarray
    arcs: np.ndarray
    valid: np.ndarray
    directory: Path
def table(path, columns):
    a=np.loadtxt(path, ndmin=2)
    if a.shape[1]!=columns or not len(a): raise ValueError(f"Invalid shape: {path}")
    return a
def coordinates_ok(a):
    return np.isfinite(a).all(axis=1)&(np.abs(a[:,0])<=180)&(np.abs(a[:,1])<=90)
def list_trips(partition, data=DATA):
    if partition not in ("train","student_test","evaluator_test"): raise ValueError("Invalid partition")
    root=Path(data)/partition
    if not root.exists(): return []
    return sorted(p.name for p in root.iterdir() if p.is_dir() and re.fullmatch(r"[0-9]{8}",p.name))
def trip_directory(partition, trip_id, data=DATA):
    if not re.fullmatch(r"[0-9]{8}",trip_id) or trip_id not in list_trips(partition,data):
        raise ValueError("Unknown trip ID")
    root=(Path(data)/partition).resolve()
    p=(root/trip_id).resolve()
    if not p.is_relative_to(root): raise ValueError("Trip path outside partition")
    return p
def load_trip(partition, trip_id, data=DATA):
    p=trip_directory(partition,trip_id,data)
    track=table(p/f"{trip_id}.track",3); nodes=table(p/f"{trip_id}.nodes",2)
    arcs=table(p/f"{trip_id}.arcs",2)
    if not coordinates_ok(nodes).all(): raise ValueError("Invalid road node coordinates")
    if not np.isfinite(arcs).all() or not np.equal(arcs,np.floor(arcs)).all(): raise ValueError("Noninteger arc endpoints")
    arcs=arcs.astype(np.int64)
    if arcs.min()<0 or arcs.max()>=len(nodes): raise ValueError("Arc node ID out of bounds")
    valid=coordinates_ok(track[:,:2])&np.isfinite(track[:,2])
    return Trip(trip_id,track,nodes,arcs,valid,p)
def load_reference(trip):
    a=table(trip.directory/f"{trip.trip_id}.route",1).ravel()
    if not np.isfinite(a).all() or not np.equal(a,np.floor(a)).all(): raise ValueError("Invalid reference IDs")
    a=a.astype(np.int64)
    if a.min()<0 or a.max()>=len(trip.arcs): raise ValueError("Reference arc out of bounds")
    return a

