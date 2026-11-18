"""Directed-network HMM; inputs contain no reference labels."""
import numpy as np
from scipy.special import logsumexp
from .evaluate_models import top1
def decode(trip,graph,gps,ids,fractions,probabilities,beta_m=30.,max_gap_s=30.,max_path_m=3000.):
    n,k=ids.shape
    output=np.full(n,-1,dtype=int); breaks=[]; start=0
    back=np.full((n,k),-1,dtype=np.int16); history=[]
    def finish(end,score):
        if score is None: return
        state=int(np.argmax(score))
        for t in range(end,start-1,-1):
            output[t]=ids[t,state]
            if t>start: state=int(back[t,state])
    previous=None
    for t in range(n):
        valid=ids[t]>=0
        if not valid.any():
            finish(t-1,previous); previous=None; breaks.append(t); start=t+1; continue
        emission=np.full(k,-np.inf); p=np.clip(probabilities[t,valid],1e-7,1)
        emission[valid]=np.log(p/p.sum())
        dt=trip.track[t,2]-trip.track[t-1,2] if t else 0
        reset=previous is None or dt<=0 or dt>max_gap_s
        if not reset:
            distance=float(np.linalg.norm(gps[t]-gps[t-1]))
            transitions=np.full((k,k),-np.inf)
            for a in np.flatnonzero(np.isfinite(previous)):
                for b in np.flatnonzero(valid):
                    source,target=int(ids[t-1,a]),int(ids[t,b])
                    road=graph.transition_distance(source,target,fractions[t-1,a],fractions[t,b],max_path_m)
                    if np.isfinite(road) and road<=max(150.,60*dt):
                        transitions[a,b]=-abs(road-distance)/(beta_m+distance*.1)
            # Normalize available transitions for each preceding state.
            normal=logsumexp(transitions,axis=1)
            finite=np.isfinite(normal)
            transitions[finite]-=normal[finite,None]
            values=previous[:,None]+transitions
            back[t]=np.argmax(values,axis=0)
            current=emission+np.max(values,axis=0)
            if not np.isfinite(current).any(): reset=True
        if reset:
            if previous is not None: finish(t-1,previous); breaks.append(t)
            start=t; current=emission
        current-=np.max(current); previous=current
    finish(n-1,previous)
    return output,sorted(set(breaks))
def reconstruct(trip,graph,pred,breaks,max_path_m=3000.):
    route=[]; boundaries=[]; previous=-1; breakset=set(breaks)
    for i,arc in enumerate(pred):
        if arc<0: previous=-1; continue
        arc=int(arc)
        if previous<0 or i in breakset:
            if route: boundaries.append(len(route))
            route.append(arc)
        elif arc!=previous:
            distance,bridge=graph.shortest(graph.arcs[previous,1],graph.arcs[arc,0],max_path_m)
            if np.isfinite(distance): route.extend(bridge)
            else: boundaries.append(len(route))
            route.append(arc)
        previous=arc
    return route,boundaries

