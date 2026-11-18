"""Exact nearest-segment refinement using midpoint cKDTree bounding spheres."""
import numpy as np
from scipy.spatial import cKDTree
from .config import K
from .preprocessing import projections
def candidates(graph,gps,valid,k=K):
    tree=cKDTree((graph.start+graph.end)/2)
    maxhalf=float(graph.length.max()/2)
    ids=np.full((len(gps),k),-1,dtype=np.int64)
    distances=np.full((len(gps),k),np.inf); fractions=np.zeros((len(gps),k))
    for i in np.flatnonzero(valid):
        p=gps[i]; _,seed=tree.query(p,k=min(k,len(graph.arcs)))
        seed=np.atleast_1d(seed)
        d,_=projections(p,graph.start[seed],graph.end[seed])
        # Any segment closer than the kth seed distance has midpoint within
        # kth distance + maximum half-length. Thus no closer road is omitted.
        pool=np.asarray(tree.query_ball_point(p,float(d.max()+maxhalf+1e-6)),dtype=int)
        d,f=projections(p,graph.start[pool],graph.end[pool])
        order=np.lexsort((pool,d))[:k]; n=len(order)
        ids[i,:n]=pool[order]; distances[i,:n]=d[order]; fractions[i,:n]=f[order]
    return ids,distances,fractions

