import json
import numpy as np,pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import folium
from .config import ROOT
def model_plots():
    table=pd.read_csv(ROOT/"results/models/comparison.csv")
    for metric in ["cv_macro_f1","top1_accuracy","macro_f1","log_loss"]:
        fig,ax=plt.subplots(figsize=(9,5)); ax.bar(table.model,table[metric],color="#217a91")
        ax.set_ylabel(metric.replace("_"," ")); ax.set_title("Trip-grouped CV" if metric.startswith("cv") else "Student testing (aligned reference proxy)")
        ax.tick_params(axis="x",rotation=20); fig.tight_layout(); fig.savefig(ROOT/"results/models"/f"{metric}.png",dpi=150); plt.close(fig)
    for name in table.model:
        result=json.loads((ROOT/"results/models"/f"{name}.json").read_text())
        cm=np.asarray(result["student_test"]["confusion_matrix"])
        fig,ax=plt.subplots(figsize=(5,4)); ax.imshow(cm,cmap="Blues")
        for i in range(2):
            for j in range(2): ax.text(j,i,str(cm[i,j]),ha="center",va="center",color="red")
        ax.set(xlabel="Predicted",ylabel="Reference proxy",xticks=[0,1],yticks=[0,1],title=name)
        fig.tight_layout(); fig.savefig(ROOT/"results/models"/f"{name}_confusion.png",dpi=150); plt.close(fig)
def route_map(gps,segments,network=None,reference=None):
    clean=np.asarray([p for p in gps if p is not None],dtype=float)
    m=folium.Map(location=clean.mean(axis=0).tolist() if len(clean) else [0,0],zoom_start=13,tiles="OpenStreetMap")
    if network is not None:
        g=folium.FeatureGroup(name="Road network",show=False)
        for s in network: folium.PolyLine(s,color="#777777",weight=1,opacity=.35).add_to(g)
        g.add_to(m)
    g=folium.FeatureGroup(name="GPS observations")
    for p in clean: folium.CircleMarker(p.tolist(),radius=2,color="#2479d4",fill=True).add_to(g)
    g.add_to(m)
    if reference is not None:
        g=folium.FeatureGroup(name="Reference route (offline only)")
        for s in reference: folium.PolyLine(s,color="#1c9e54",weight=6,opacity=.6).add_to(g)
        g.add_to(m)
    g=folium.FeatureGroup(name="Predicted route")
    for s in segments: folium.PolyLine(s,color="#e34b36",weight=4).add_to(g)
    g.add_to(m); folium.LayerControl().add_to(m)
    m.get_root().html.add_child(folium.Element('<div style="position:fixed;bottom:20px;left:20px;z-index:9999;background:white;padding:12px">Blue: GPS | Red: prediction | Grey: roads'+(' | Green: offline reference' if reference is not None else '')+'</div>'))
    if len(clean): m.fit_bounds([clean.min(axis=0).tolist(),clean.max(axis=0).tolist()])
    return m
def geometry(trip,route):
    return [trip.nodes[trip.arcs[int(a)]][:,::-1].tolist() for a in route if a>=0]
def eda_plots():
    table=pd.read_csv(ROOT/"results/eda/trips.csv")
    fig,axes=plt.subplots(1,3,figsize=(12,4))
    for part,group in table.groupby("partition"):
        axes[0].hist(group.observations,bins=15,alpha=.6,label=part)
        axes[1].scatter(group.longitude_min,group.latitude_min,s=12,label=part)
        axes[2].scatter(group.nodes,group.weak_components,s=12,label=part)
    for ax,title in zip(axes,["Trip observation counts","GPS extent origins (lon/lat)","Network nodes vs weak components"]): ax.set_title(title); ax.legend()
    fig.tight_layout(); fig.savefig(ROOT/"results/eda/exploration.png",dpi=150); plt.close(fig)
if __name__=="__main__": model_plots(); eda_plots()

