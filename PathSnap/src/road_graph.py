import heapq
from collections import OrderedDict
import numpy as np
from scipy.sparse import csr_matrix
from scipy.sparse.csgraph import connected_components
class RoadGraph:
    def __init__(self,trip,xy):
        self.arcs=trip.arcs; self.xy=xy
        self.start=xy[self.arcs[:,0]]; self.end=xy[self.arcs[:,1]]
        self.length=np.linalg.norm(self.end-self.start,axis=1)
        self.heading=np.arctan2(*(self.end-self.start)[:,::-1].T)
        self.adj=[[] for _ in xy]
        for arc,(u,v) in enumerate(self.arcs):
            self.adj[u].append((int(v),float(self.length[arc]),arc))
        self.indegree=np.bincount(self.arcs[:,1],minlength=len(xy))
        self.outdegree=np.bincount(self.arcs[:,0],minlength=len(xy))
        sparse=csr_matrix((np.ones(len(self.arcs)),self.arcs.T),shape=(len(xy),len(xy)))
        self.components,self.component_labels=connected_components(sparse,directed=True,connection="weak")
        self.strong_components=connected_components(sparse,directed=True,connection="strong",return_labels=False)
        self.cache=OrderedDict()
    def shortest(self,source,target,limit=3000):
        if source==target: return 0.,[]
        key=(int(source),int(target),float(limit))
        if key in self.cache: return self.cache[key]
        if self.component_labels[source]!=self.component_labels[target]: return float("inf"),[]
        distances={int(source):0.}; parent={}; heap=[(0.,int(source))]
        while heap:
            d,u=heapq.heappop(heap)
            if d!=distances[u]: continue
            if u==target: break
            for v,w,arc in self.adj[u]:
                nd=d+w
                if nd<=limit and nd<distances.get(v,float("inf")):
                    distances[v]=nd; parent[v]=(u,arc); heapq.heappush(heap,(nd,v))
        if target not in distances: result=(float("inf"),[])
        else:
            route=[]; u=int(target)
            while u!=source: u,arc=parent[u]; route.append(arc)
            result=(distances[target],route[::-1])
        self.cache[key]=result
        if len(self.cache)>20000: self.cache.popitem(last=False)
        return result
    def transition_distance(self,a,b,fa,fb,limit=3000):
        if a==b and fb>=fa-0.03: return abs(fb-fa)*self.length[a]
        d,_=self.shortest(self.arcs[a,1],self.arcs[b,0],limit)
        return (1-fa)*self.length[a]+d+fb*self.length[b]

