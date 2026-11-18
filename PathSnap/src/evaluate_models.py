import time
import numpy as np
from sklearn.metrics import accuracy_score,precision_score,recall_score,f1_score,log_loss,confusion_matrix
def candidate_metrics(y,p):
    y=np.asarray(y).ravel(); p=np.clip(np.asarray(p).ravel(),1e-7,1-1e-7); pred=p>=.5
    return dict(accuracy=accuracy_score(y,pred),precision=precision_score(y,pred,zero_division=0),
        recall=recall_score(y,pred,zero_division=0),macro_f1=f1_score(y,pred,average="macro",zero_division=0),
        log_loss=log_loss(y,p,labels=[0,1]),confusion_matrix=confusion_matrix(y,pred,labels=[0,1]).tolist())
def score(model,x,ids):
    started=time.perf_counter()
    p=model.predict_proba(x.reshape(-1,x.shape[-1]))[:,1].reshape(ids.shape)
    p=np.where(ids>=0,p,0)
    return p,time.perf_counter()-started
def top1(ids,p):
    answer=ids[np.arange(len(ids)),np.argmax(p,axis=1)]
    return np.where((ids>=0).any(axis=1),answer,-1)
def compress(values):
    return [int(v) for i,v in enumerate(values) if v>=0 and (i==0 or v!=values[i-1])]
def edit_similarity(a,b):
    if not a and not b: return 1.
    prev=list(range(len(b)+1))
    for i,x in enumerate(a):
        cur=[i+1]
        for j,y in enumerate(b): cur.append(min(cur[-1]+1,prev[j+1]+1,prev[j]+(x!=y)))
        prev=cur
    return 1-prev[-1]/max(len(a),len(b),1)
def route_metrics(pred,labels,ids,trip,reference,route=None):
    valid=trip.valid&(labels>=0)
    pairs=(pred[:-1]>=0)&(pred[1:]>=0)
    a=np.maximum(pred[:-1],0); b=np.maximum(pred[1:],0)
    continuous=(a==b)|(trip.arcs[a,1]==trip.arcs[b,0])
    sequence=compress(pred) if route is None else route
    return dict(top1_accuracy=float(np.mean(pred[valid]==labels[valid])) if valid.any() else None,
        candidate_miss_rate=float(np.mean(~(ids[valid]==labels[valid,None]).any(axis=1))) if valid.any() else None,
        route_continuity=float(np.mean(continuous[pairs])) if pairs.any() else None,
        valid_transition_pairs=int(pairs.sum()),sequence_edit_similarity=edit_similarity(sequence,reference.tolist()),
        sequence_exact_match=bool(sequence==reference.tolist()),observations=int(valid.sum()),unmatched=int((pred<0).sum()))

