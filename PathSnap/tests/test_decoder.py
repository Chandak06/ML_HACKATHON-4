import numpy as np
from test_core import fixture
from src.candidate_generation import candidates
from src.hmm_viterbi import decode,reconstruct
def test_viterbi_and_backtracking():
    t,g,gps=fixture(); ids,d,f=candidates(g,gps,t.valid,k=4)
    p=np.exp(-d/10); p[0]=np.where(ids[0]==0,0.99,0.001)
    pred,breaks=decode(t,g,gps,ids,f,p)
    assert pred.tolist()==[0,0,1,2]
    route,boundaries=reconstruct(t,g,pred,breaks)
    assert route==[0,1,2] and not boundaries
def test_gaps_and_missing():
    t,g,gps=fixture(); ids,d,f=candidates(g,gps,t.valid,k=4)
    ids[1]=-1; t.track[3,2]=100
    pred,breaks=decode(t,g,gps,ids,f,np.ones_like(d))
    assert pred[1]==-1 and 1 in breaks and 3 in breaks
    assert all(a==-1 or 0<=a<len(g.arcs) for a in pred)
def test_disconnected_fallback():
    t,g,gps=fixture()
    ids=np.array([[2],[0],[0],[1]]); fractions=np.full_like(ids,.5,dtype=float)
    pred,breaks=decode(t,g,gps,ids,fractions,np.ones_like(ids))
    assert pred.tolist()==[2,0,0,1] and 1 in breaks


