#!/usr/bin/env python3
"""Evaluate the *deployed* 188 KiB INT8 TFLite artifact shipped with the paper."""
import os, json, time
os.environ.setdefault('TF_CPP_MIN_LOG_LEVEL','3'); os.environ['CUDA_VISIBLE_DEVICES']='-1'
import sys
import numpy as np, tensorflow as tf
from sklearn.metrics import roc_auc_score

HERE=os.path.dirname(os.path.abspath(__file__)); REPO=os.path.dirname(HERE)
OUT=os.path.join(HERE,'outputs')
TFL=os.path.join(REPO,'models','earthquake_detector.tflite')
sys.path.insert(0, HERE)
import dataset
_Xv,_yv,_Xt,_yt,_ = dataset.load()
d={'X_val':_Xv,'y_val':_yv,'X_test':_Xt,'y_test':_yt}

it=tf.lite.Interpreter(model_path=TFL); it.allocate_tensors()
inp,out=it.get_input_details()[0],it.get_output_details()[0]
print('input',inp['shape'],inp['dtype'],inp['quantization'])
print('output',out['shape'],out['dtype'],out['quantization'])

def predict(X):
    o=np.empty(len(X))
    qs,qz=inp['quantization']
    for i in range(len(X)):
        x=X[i:i+1]
        if inp['dtype']==np.int8:
            x=np.clip(np.round(x/qs+qz),-128,127).astype(np.int8)
        else: x=x.astype(np.float32)
        it.set_tensor(inp['index'],x); it.invoke()
        v=float(it.get_tensor(out['index'])[0][0])
        if out['dtype']==np.int8:
            s,z=out['quantization']; v=(v-z)*s
        o[i]=v
    return o

def met(y,p,thr=0.5):
    yh=(p>=thr).astype(int)
    tp=int(((yh==1)&(y==1)).sum());tn=int(((yh==0)&(y==0)).sum())
    fp=int(((yh==1)&(y==0)).sum());fn=int(((yh==0)&(y==1)).sum())
    pr=tp/max(tp+fp,1);rc=tp/max(tp+fn,1)
    return dict(tp=tp,tn=tn,fp=fp,fn=fn,precision=pr,recall=rc,
                f1=2*pr*rc/max(pr+rc,1e-9),fpr=fp/max(fp+tn,1),
                accuracy=(tp+tn)/len(y),auc=float(roc_auc_score(y,p)))

r={'tflite_bytes':os.path.getsize(TFL),'tflite_kib':os.path.getsize(TFL)/1024}
pv=predict(d['X_val']); pt=predict(d['X_test'])
r['val']=met(d['y_val'],pv); r['test_crossdomain']=met(d['y_test'],pt)

x1=d['X_val'][:1]
qs,qz=inp['quantization']
x1=np.clip(np.round(x1/qs+qz),-128,127).astype(np.int8) if inp['dtype']==np.int8 else x1.astype(np.float32)
for _ in range(50): it.set_tensor(inp['index'],x1); it.invoke()
ts=[]
for _ in range(500):
    t0=time.perf_counter(); it.set_tensor(inp['index'],x1); it.invoke(); ts.append((time.perf_counter()-t0)*1e3)
r['latency_ms']={'mean':float(np.mean(ts)),'p50':float(np.percentile(ts,50)),'p95':float(np.percentile(ts,95))}
np.save(os.path.join(OUT,'deployed_conf_val.npy'),pv)
np.save(os.path.join(OUT,'deployed_conf_test.npy'),pt)
json.dump(r,open(os.path.join(OUT,'deployed_tflite.json'),'w'),indent=2)
print(json.dumps(r,indent=2))
