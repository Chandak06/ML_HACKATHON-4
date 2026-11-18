import numpy as np
EARTH_RADIUS_M=6371008.8
def project(trip):
    # Local equirectangular metric projection. Audit arc-length distortion
    # against spherical haversine in the EDA; no native GIS runtime required.
    origin=np.median(trip.nodes,axis=0); lat0=np.deg2rad(origin[1])
    def convert(coords):
        delta=np.deg2rad(coords-origin)
        return EARTH_RADIUS_M*np.column_stack((delta[:,0]*np.cos(lat0),delta[:,1]))
    nodes=convert(trip.nodes); gps=np.full((len(trip.track),2),np.nan)
    gps[trip.valid]=convert(trip.track[trip.valid,:2])
    return nodes,gps
def haversine(a,b):
    a=np.deg2rad(a); b=np.deg2rad(b); delta=b-a
    value=np.sin(delta[:,1]/2)**2+np.cos(a[:,1])*np.cos(b[:,1])*np.sin(delta[:,0]/2)**2
    return 2*EARTH_RADIUS_M*np.arcsin(np.sqrt(np.clip(value,0,1)))
def movement(trip,gps):
    n=len(gps); delta=np.zeros((n,2)); dt=np.zeros(n)
    if n>1:
        delta[1:]=gps[1:]-gps[:-1]; delta[0]=delta[1]
        dt[1:]=np.diff(trip.track[:,2]); dt[0]=dt[1]
    dist=np.linalg.norm(delta,axis=1)
    speed=np.divide(dist,dt,out=np.zeros(n),where=dt>0)
    outlier=(speed>60)|~trip.valid|~np.isfinite(dist)
    left=np.maximum(np.arange(n)-2,0); right=np.minimum(np.arange(n)+2,n-1)
    smooth=gps[right]-gps[left]
    heading=np.arctan2(smooth[:,1],smooth[:,0])
    good=(np.linalg.norm(smooth,axis=1)>3)&~outlier&(dt>0)&(dt<=30)
    return np.nan_to_num(heading),good,np.nan_to_num(dt),np.clip(np.nan_to_num(speed),0,60),outlier
def projections(points,starts,ends):
    v=ends-starts
    t=np.sum((points-starts)*v,axis=-1)/np.maximum(np.sum(v*v,axis=-1),1e-12)
    t=np.clip(t,0,1)
    return np.linalg.norm(points-(starts+t[...,None]*v),axis=-1),t

