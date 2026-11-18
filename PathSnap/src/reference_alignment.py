"""Offline monotone geometry alignment, independent of candidate generation/model."""
import numpy as np
from .preprocessing import projections,movement
def align_reference(trip,graph,gps,route):
    indices=np.flatnonzero(trip.valid)
    labels=np.full(len(gps),-1,dtype=int); residual=np.full(len(gps),np.inf)
    if not len(indices): return labels,residual
    start=graph.start[route]; end=graph.end[route]
    h,good,*_=movement(trip,gps)
    costs=np.zeros(len(route)); back=np.zeros((len(indices),len(route)),dtype=np.int32)
    for step,i in enumerate(indices):
        d,_=projections(gps[i],start,end)
        angle=np.abs(np.arctan2(np.sin(h[i]-graph.heading[route]),np.cos(h[i]-graph.heading[route])))
        emission=np.minimum(d/20,50)**2+np.where(good[i],angle/np.pi,0)
        if step:
            best=np.minimum.accumulate(costs)
            arg=np.maximum.accumulate(np.where(costs==best,np.arange(len(route)),-1))
            back[step]=arg; costs=best+emission
        else: costs=emission
        costs-=costs.min()
    state=int(np.argmin(costs))
    for step in range(len(indices)-1,-1,-1):
        i=indices[step]; labels[i]=route[state]
        residual[i]=projections(gps[i],start[state],end[state])[0]
        state=back[step,state]
    return labels,residual

