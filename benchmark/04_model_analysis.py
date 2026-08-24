#!/usr/bin/env python3
"""
Revision R1 -- Reviewer #2, comment 7
=====================================
Reproduces the edge model's architecture table, training/validation curves,
overfitting diagnostics, INT8 quantisation footprint and inference latency
from the *actual* trained ModelV5 checkpoint.

All numbers written to outputs/model_analysis.json are measured, not quoted.
"""
import os, json, time, sys
os.environ.setdefault('TF_CPP_MIN_LOG_LEVEL', '3')
import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
OUT  = os.path.join(HERE, 'outputs'); os.makedirs(OUT, exist_ok=True)
sys.path.insert(0, HERE)
REPO = os.path.dirname(HERE)

import tensorflow as tf
from tensorflow import keras

res = {}

# ---------------------------------------------------------------- architecture
model = keras.models.load_model(os.path.join(REPO, 'models', 'edge_detector.keras'))
layers = []
for l in model.layers:
    cfg = l.get_config()
    out_shape = l.output.shape if hasattr(l, 'output') else None
    layers.append({
        'name': l.name,
        'type': l.__class__.__name__,
        'output_shape': [None if d is None else int(d) for d in out_shape],
        'params': int(l.count_params()),
        'filters': cfg.get('filters'),
        'kernel_size': cfg.get('kernel_size'),
        'units': cfg.get('units'),
        'rate': cfg.get('rate'),
        'pool_size': cfg.get('pool_size'),
        'padding': cfg.get('padding'),
    })
res['layers'] = layers
res['total_params'] = int(model.count_params())
res['trainable_params'] = int(sum(np.prod(w.shape) for w in model.trainable_weights))
res['keras_file_bytes'] = os.path.getsize(os.path.join(REPO, 'models',
                                                        'edge_detector.keras'))

# ---------------------------------------------------------------- training run
hist = json.load(open(os.path.join(REPO, 'models', 'training_history.json')))
n_ep = len(hist['loss'])
best_ep = int(np.argmin(hist['val_loss'])) + 1
res['training'] = {
    'epochs_run': n_ep,
    'epochs_budget': 50,
    'best_epoch_by_val_loss': best_ep,
    'early_stopping_patience': 10,
    'train_loss_at_best': hist['loss'][best_ep - 1],
    'val_loss_at_best':   hist['val_loss'][best_ep - 1],
    'train_acc_at_best':  hist['accuracy'][best_ep - 1],
    'val_acc_at_best':    hist['val_accuracy'][best_ep - 1],
    'final_train_loss':   hist['loss'][-1],
    'final_val_loss':     hist['val_loss'][-1],
    'generalisation_gap_acc': hist['accuracy'][best_ep-1] - hist['val_accuracy'][best_ep-1],
    'lr_schedule': hist.get('learning_rate', []),
    'history': hist,
}

# ---------------------------------------------------------------- data + repro
import dataset
Xv, yv, Xt, yt, Xtr = dataset.load()
res['dataset'] = {
    'train_windows': int(Xtr.shape[0]),
    'val_windows':   int(Xv.shape[0]),
    'test_windows':  int(Xt.shape[0]),
    'val_pos': int(yv.sum()), 'val_neg': int((yv == 0).sum()),
    'test_pos': int(yt.sum()), 'test_neg': int((yt == 0).sum()),
    'window_shape': list(map(int, Xv.shape[1:])),
}

def metrics(y, p, thr=0.5):
    yh = (p >= thr).astype(int)
    tp = int(((yh == 1) & (y == 1)).sum()); tn = int(((yh == 0) & (y == 0)).sum())
    fp = int(((yh == 1) & (y == 0)).sum()); fn = int(((yh == 0) & (y == 1)).sum())
    pr = tp / max(tp + fp, 1); rc = tp / max(tp + fn, 1)
    from sklearn.metrics import roc_auc_score
    return {'tp': tp, 'tn': tn, 'fp': fp, 'fn': fn,
            'precision': pr, 'recall': rc,
            'f1': 2 * pr * rc / max(pr + rc, 1e-9),
            'fpr': fp / max(fp + tn, 1),
            'accuracy': (tp + tn) / len(y),
            'auc': float(roc_auc_score(y, p))}

pv = model.predict(Xv, batch_size=512, verbose=0).ravel()
pt = model.predict(Xt, batch_size=512, verbose=0).ravel()
res['fp32'] = {'val': metrics(yv, pv), 'test_crossdomain': metrics(yt, pt)}

# ------------------------------------------------- bootstrap 95% CI (1000 reps)
def boot_ci(y, p, n=1000, seed=0):
    rng = np.random.default_rng(seed); keys = ['precision','recall','f1','fpr','auc','accuracy']
    acc = {k: [] for k in keys}
    for _ in range(n):
        idx = rng.integers(0, len(y), len(y))
        if len(np.unique(y[idx])) < 2: continue
        m = metrics(y[idx], p[idx])
        for k in keys: acc[k].append(m[k])
    return {k: {'mean': float(np.mean(v)),
                'lo': float(np.percentile(v, 2.5)),
                'hi': float(np.percentile(v, 97.5)),
                'halfwidth': float((np.percentile(v, 97.5) - np.percentile(v, 2.5)) / 2)}
            for k, v in acc.items()}
res['bootstrap_ci'] = {'val': boot_ci(yv, pv, seed=1), 'test_crossdomain': boot_ci(yt, pt, seed=2)}

# ---------------------------------------------------------------- INT8 export
rep = Xtr[:500].astype(np.float32)
def rep_gen():
    for i in range(rep.shape[0]):
        yield [rep[i:i+1]]
conv = tf.lite.TFLiteConverter.from_keras_model(model)
conv.optimizations = [tf.lite.Optimize.DEFAULT]
conv.representative_dataset = rep_gen
tfl_int8 = conv.convert()
p8 = os.path.join(OUT, 'edge_model_int8.tflite'); open(p8, 'wb').write(tfl_int8)

conv2 = tf.lite.TFLiteConverter.from_keras_model(model)
tfl_f32 = conv2.convert()
p32 = os.path.join(OUT, 'edge_model_fp32.tflite'); open(p32, 'wb').write(tfl_f32)

res['quantisation'] = {'fp32_tflite_bytes': len(tfl_f32),
                       'int8_tflite_bytes': len(tfl_int8),
                       'compression_ratio': len(tfl_f32) / len(tfl_int8)}

# INT8 accuracy on the same sets
def tflite_predict(path, X):
    it = tf.lite.Interpreter(model_path=path); it.allocate_tensors()
    inp, out = it.get_input_details()[0], it.get_output_details()[0]
    o = []
    for i in range(X.shape[0]):
        it.set_tensor(inp['index'], X[i:i+1].astype(inp['dtype'] if inp['dtype']!=np.int8 else np.float32)
                      if inp['dtype'] == np.float32 else
                      np.clip(np.round(X[i:i+1]/inp['quantization'][0] + inp['quantization'][1]), -128, 127).astype(np.int8))
        it.invoke()
        v = it.get_tensor(out['index'])[0][0]
        if out['dtype'] == np.int8:
            v = (float(v) - out['quantization'][1]) * out['quantization'][0]
        o.append(float(v))
    return np.array(o)

pv8 = tflite_predict(p8, Xv); pt8 = tflite_predict(p8, Xt)
res['int8'] = {'val': metrics(yv, pv8), 'test_crossdomain': metrics(yt, pt8)}
res['quantisation']['f1_drop_pp_val']  = (res['fp32']['val']['f1'] - res['int8']['val']['f1']) * 100
res['quantisation']['f1_drop_pp_test'] = (res['fp32']['test_crossdomain']['f1'] - res['int8']['test_crossdomain']['f1']) * 100

# ---------------------------------------------------------------- latency
it = tf.lite.Interpreter(model_path=p8); it.allocate_tensors()
inp = it.get_input_details()[0]; out = it.get_output_details()[0]
x1 = np.clip(np.round(Xv[:1]/inp['quantization'][0] + inp['quantization'][1]), -128, 127).astype(np.int8) \
     if inp['dtype'] == np.int8 else Xv[:1].astype(np.float32)
for _ in range(20):
    it.set_tensor(inp['index'], x1); it.invoke()
ts = []
for _ in range(300):
    t0 = time.perf_counter(); it.set_tensor(inp['index'], x1); it.invoke(); ts.append((time.perf_counter()-t0)*1e3)
res['latency_ms_int8_workstation_cpu'] = {'mean': float(np.mean(ts)), 'p50': float(np.percentile(ts,50)),
                                          'p95': float(np.percentile(ts,95)), 'std': float(np.std(ts))}

json.dump(res, open(os.path.join(OUT, 'model_analysis.json'), 'w'), indent=2)
print(json.dumps({k: v for k, v in res.items() if k not in ('layers', 'training', 'bootstrap_ci')}, indent=2))
print('\nLayers:')
for l in res['layers']:
    print(f"  {l['name']:<28} {l['type']:<20} {str(l['output_shape']):<20} {l['params']:>8,}")
print(f"\nTOTAL PARAMS {res['total_params']:,}")
print(f"best epoch (val_loss) = {res['training']['best_epoch_by_val_loss']} / {res['training']['epochs_run']} run")
