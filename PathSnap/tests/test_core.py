from pathlib import Path
import numpy as np
import pytest
from src.data_loader import Trip,load_trip,load_reference,list_trips,table
from src.preprocessing import project,projections
from src.road_graph import RoadGraph
from src.candidate_generation import candidates
from src.feature_engineering import features
from src.reference_alignment import align_reference
def fixture():
    nodes=np.array([[8.,50.],[8.001,50.],[8.002,50.],[8.003,50.]])
    arcs=np.array([[0,1],[1,2],[2,3],[1,0]])
    track=np.array([[8.0002,50.,1],[8.0008,50.,2],[8.0015,50.,3],[8.0028,50.,4]])
    t=Trip("00000000",track,nodes,arcs,np.ones(4,dtype=bool),Path("."))
    xy,gps=project(t); return t,RoadGraph(t,xy),gps
def test_loading_real_data():
    assert len(list_trips("train"))==80 and len(list_trips("student_test"))==10
    assert not set(list_trips("train"))&set(list_trips("student_test"))
    trip=load_trip("student_test",list_trips("student_test")[0])
    assert len(load_reference(trip))!=len(trip.track)
def test_id_security():
    with pytest.raises(ValueError): load_trip("student_test","../../data")
    with pytest.raises(ValueError): list_trips("invalid")
def test_graph_and_metres():
    t,g,gps=fixture()
    assert 70<g.length[0]<73
    assert g.shortest(0,3)[1]==[0,1,2]
    assert np.isinf(g.shortest(3,0)[0])
def test_exact_candidates_and_features():
    t,g,gps=fixture(); ids,d,f=candidates(g,gps,t.valid,k=4)
    brute=np.array([projections(p,g.start,g.end)[0] for p in gps])
    np.testing.assert_allclose(d,np.sort(brute,axis=1))
    x=features(t,g,gps,ids,d,f); assert np.isfinite(x).all() and x.shape==(4,4,14)
    np.testing.assert_allclose(x[:,:,0],d,atol=1e-4)
def test_reference_alignment_chronology():
    t,g,gps=fixture()
    labels,residual=align_reference(t,g,gps,np.array([0,1,2]))
    assert labels.tolist()==[0,0,1,2]
    assert np.max(residual)<1e-6
def test_missing_candidate_handling():
    t,g,gps=fixture(); t.valid[1]=False
    ids,d,f=candidates(g,gps,t.valid,k=8)
    assert (ids[1]==-1).all() and (ids[:,4:]==-1).all()
def test_projection_endpoints():
    d,f=projections(np.array([3,4]),np.array([0,0]),np.array([2,0]))
    assert f==1 and abs(d-np.sqrt(17))<1e-9

