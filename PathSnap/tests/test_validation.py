import numpy as np
import pytest
from src.data_loader import table,coordinates_ok
from src.candidate_generation import candidates
from src.preprocessing import projections
from test_core import fixture
def test_malformed_missing_columns(tmp_path):
    path=tmp_path/"bad.track"; path.write_text("8 50\n9 51\n")
    with pytest.raises(ValueError): table(path,3)
    assert coordinates_ok(np.array([[8.,50.],[181.,50.],[8.,91.],[np.nan,50.]])).tolist()==[True,False,False,False]
def test_long_segment_not_missed_by_midpoint_index():
    class Network:
        arcs=np.array([[0,1],[2,3],[4,5]])
        start=np.array([[-10000.,0.],[0.,10.],[0.,20.]])
        end=np.array([[100.,0.],[1.,10.],[1.,20.]])
        length=np.linalg.norm(end-start,axis=1)
    g=Network(); gps=np.array([[0.,1.]])
    ids,d,f=candidates(g,gps,np.array([True]),k=1)
    assert ids[0,0]==0 and d[0,0]==1
def test_feature_inputs_have_no_reference_dependency():
    import inspect
    from src.feature_engineering import features
    from src.inference import Predictor
    assert "load_reference" not in inspect.getsource(Predictor)
    assert "labels" not in inspect.signature(features).parameters

