import numpy as np
from .preprocessing import movement
from .config import FEATURES
def features(trip,graph,gps,ids,distance,fraction):
    h,good,dt,speed,outlier=movement(trip,gps)
    safe=np.maximum(ids,0); a=graph.start[safe]; b=graph.end[safe]
    difference=np.abs(np.arctan2(np.sin(h[:,None]-graph.heading[safe]),np.cos(h[:,None]-graph.heading[safe])))
    difference=np.where(good[:,None],difference,0)
    shape=ids.shape
    broadcast=lambda x:np.broadcast_to(x[:,None],shape)
    x=np.stack([distance,np.linalg.norm(gps[:,None]-a,axis=2),np.linalg.norm(gps[:,None]-b,axis=2),
        graph.length[safe],difference,broadcast(good),broadcast(dt),broadcast(speed),
        graph.indegree[graph.arcs[safe,0]],graph.outdegree[graph.arcs[safe,1]],
        np.broadcast_to(np.arange(shape[1]),shape),fraction,distance-distance[:,0:1],broadcast(outlier)],axis=-1)
    return np.nan_to_num(x,nan=0,posinf=100000,neginf=-100000).astype(np.float32)

