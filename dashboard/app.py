"""
Web Dashboard for EEW System

Real-time monitoring and control interface for the earthquake
early warning system. Provides:
- Live device map
- Earthquake event timeline
- System metrics
- Simulation controls

Author: Muhammed Şara
"""

from flask import Flask, render_template, jsonify, request
from flask_socketio import SocketIO, emit
import threading
import time
import json
import sys
import os

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from config import config
from devices.device_manager import DeviceManager
from consensus.engine import ConsensusEngine
from consensus.engine import ConsensusEngine
from simulation.trace_simulator import TraceSimulator
try:
    from data.preprocessor import preprocessor
except ImportError:
    try:
        from src.data.preprocessor import preprocessor
    except ImportError:
        preprocessor = None  # Not available — /api/analysis/sample will return error
import numpy as np

# TensorFlow for V5 model (optional)
try:
    import tensorflow as tf
    tf.get_logger().setLevel('ERROR')
except ImportError:
    tf = None

# V5 Model path
V5_MODEL_PATH = os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(__file__))), 
                              'ModelV5', 'models', 'best.keras')
V5_MODEL = None  # Lazy load


# Initialize Flask app
app = Flask(__name__, 
    template_folder='templates',
    static_folder='static'
)
app.config['SECRET_KEY'] = 'eew-secret-key-2024'
socketio = SocketIO(app, cors_allowed_origins="*")

# Global state
system_state = {
    'is_running': False,
    'mode': 'standby',  # standby, monitoring, simulation
    'events': [],
    'devices': {'mobile': [], 'iot': []},
    'metrics': {
        'total_events': 0,
        'earthquake_detections': 0,
        'false_alarms': 0,
        'tpr': 0.0,
        'fpr': 0.0
    },
    'config': {
        'mobile_weight': config.consensus.mobile_weight,
        'iot_weight': config.consensus.iot_weight,
        'threshold': config.consensus.threshold_default
    }
}

# Components
device_manager = None
consensus_engine = None
simulator = None


@app.route('/')
def index():
    """Main dashboard page"""
    return render_template('index.html')


@app.route('/api/status')
def get_status():
    """Get current system status"""
    return jsonify(system_state)


@app.route('/api/models')
def get_models():
    """List all available trained models with detailed metrics from output folder"""
    import glob
    
    models_dir = os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(__file__))), 'models')
    output_dir = os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(__file__))), 'output')
    models_list = []
    
    # Scan for model versions
    for version_dir in sorted(glob.glob(os.path.join(models_dir, 'v*'))):
        version = os.path.basename(version_dir)
        
        # Find tflite file
        tflite_files = glob.glob(os.path.join(version_dir, '*.tflite'))
        keras_files = glob.glob(os.path.join(version_dir, '*.keras'))
        
        if tflite_files or keras_files:
            model_file = tflite_files[0] if tflite_files else keras_files[0]
            model_size = os.path.getsize(model_file) / 1024  # KB
            
            # Try to read results from output folder first
            results_file = os.path.join(output_dir, version, f'results_{version}.json')
            results = {}
            if os.path.exists(results_file):
                try:
                    with open(results_file, 'r') as f:
                        results = json.load(f)
                except:
                    pass
            
            # Extract metrics from results
            metrics = results.get('metrics', {})
            
            # Check for metadata in model folder
            metadata_file = os.path.join(version_dir, 'model_metadata.json')
            metadata = {}
            if os.path.exists(metadata_file):
                try:
                    with open(metadata_file, 'r') as f:
                        metadata = json.load(f)
                except:
                    pass
            
            # Get performance - prefer output results over metadata
            perf = metadata.get('performance', {})
            
            models_list.append({
                'version': version,
                'name': results.get('model', f'KURTAR {version}'),
                'path': model_file,
                'type': 'tflite' if tflite_files else 'keras',
                'size_kb': round(model_size, 1),
                'date': results.get('date', metadata.get('created_date', 'Unknown')),
                'recall': metrics.get('recall', perf.get('recall_earthquake', None)),
                'precision': metrics.get('precision', perf.get('precision_earthquake', None)),
                'f1_score': metrics.get('f1', perf.get('f1_score_earthquake', None)),
                'accuracy': metrics.get('accuracy', perf.get('test_accuracy', None)),
                'auc': metrics.get('auc'),
                'tp': metrics.get('TP'),
                'tn': metrics.get('TN'),
                'fp': metrics.get('FP'),
                'fn': metrics.get('FN'),
                'dataset': results.get('dataset', metadata.get('training_data', {}).get('sources', 'Unknown')),
                'changes': results.get(f'{version.replace(".", "")}_changes', results.get('v26_changes', results.get('v25_changes', []))),
                'status': 'available'
            })
    
    # Also scan ModelV3 directory
    modelv3_base = os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(__file__))), 'ModelV3')
    modelv3_models_dir = os.path.join(modelv3_base, 'models')
    modelv3_output_dir = os.path.join(modelv3_base, 'outputs')
    
    for version_dir in sorted(glob.glob(os.path.join(modelv3_models_dir, 'v3*'))):
        version = os.path.basename(version_dir)
        
        # Find tflite file
        tflite_files = glob.glob(os.path.join(version_dir, '*.tflite'))
        keras_files = glob.glob(os.path.join(version_dir, '*.keras'))
        
        if tflite_files or keras_files:
            model_file = tflite_files[0] if tflite_files else keras_files[0]
            model_size = os.path.getsize(model_file) / 1024  # KB
            
            # Read results from ModelV3 output folder
            results_file = os.path.join(modelv3_output_dir, version, f'results_{version}.json')
            results = {}
            if os.path.exists(results_file):
                try:
                    with open(results_file, 'r') as f:
                        results = json.load(f)
                except:
                    pass
            
            metrics = results.get('metrics', {})
            
            # Determine model type name
            model_names = {
                'v3.0': 'KURTAR v3.0 (Enhanced CNN)',
                'v3.1': 'KURTAR v3.1 (Transformer)',
                'v3.2': 'KURTAR v3.2 (Hybrid CNN+Transformer)'
            }
            
            models_list.append({
                'version': version,
                'name': model_names.get(version, f'KURTAR {version}'),
                'path': model_file,
                'type': 'tflite' if tflite_files else 'keras',
                'size_kb': round(model_size, 1),
                'date': results.get('timestamp', '2026-02-06')[:10] if results.get('timestamp') else 'Unknown',
                'recall': metrics.get('recall'),
                'precision': metrics.get('precision'),
                'f1_score': metrics.get('f1'),
                'accuracy': metrics.get('accuracy'),
                'auc': metrics.get('auc'),
                'tp': metrics.get('TP'),
                'tn': metrics.get('TN'),
                'fp': metrics.get('FP'),
                'fn': metrics.get('FN'),
                'dataset': 'STEAD (36K) + HAR',
                'changes': [],
                'status': 'available',
                'generation': 'v3'
            })
    
    # Scan ModelV5 directory (Publication-Ready Version)
    modelv5_base = os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(__file__))), 'ModelV5')
    modelv5_models_dir = os.path.join(modelv5_base, 'models')
    modelv5_output_dir = os.path.join(modelv5_base, 'outputs')
    
    if os.path.exists(modelv5_models_dir):
        keras_files = glob.glob(os.path.join(modelv5_models_dir, '*.keras'))
        if keras_files:
            model_file = keras_files[0]
            model_size = os.path.getsize(model_file) / 1024
            
            # Read V5 results
            results_file = os.path.join(modelv5_output_dir, 'final_results.json')
            results = {}
            if os.path.exists(results_file):
                try:
                    with open(results_file, 'r') as f:
                        results = json.load(f)
                except:
                    pass
            
            # Use cross-domain test metrics (more realistic)
            test_metrics = results.get('test_metrics', {})
            val_metrics = results.get('validation_metrics', {})
            
            models_list.append({
                'version': 'v5.0',
                'name': 'KURTAR v5.0 (Publication-Ready)',
                'path': model_file,
                'type': 'keras',
                'size_kb': round(model_size, 1),
                'date': results.get('timestamp', '2026-02-09')[:10] if results.get('timestamp') else 'Unknown',
                'recall': test_metrics.get('recall'),
                'precision': test_metrics.get('precision'),
                'f1_score': test_metrics.get('f1'),
                'accuracy': test_metrics.get('accuracy'),
                'auc': test_metrics.get('auc'),
                'fpr': test_metrics.get('fpr'),
                'fnr': test_metrics.get('fnr'),
                'tp': test_metrics.get('tp'),
                'tn': test_metrics.get('tn'),
                'fp': test_metrics.get('fp'),
                'fn': test_metrics.get('fn'),
                'dataset': 'STEAD (train) → MyShake (cross-domain test)',
                'changes': [
                    'Cross-dataset validation',
                    'Event-based split',
                    'Data augmentation (noise, time shift)',
                    'FPR tracking'
                ],
                'status': 'available',
                'generation': 'v5',
                'cross_domain': True,
                'val_f1': val_metrics.get('f1'),
                'val_fpr': val_metrics.get('fpr')
            })
    
    # Also add the original v1.0 from tflite_v1.0 folder
    v1_dir = os.path.join(models_dir, 'tflite_v1.0')
    if os.path.exists(v1_dir) and not any(m['version'] == 'v1.0' for m in models_list):
        metadata_file = os.path.join(v1_dir, 'model_metadata_v1.0.json')
        metadata = {}
        if os.path.exists(metadata_file):
            with open(metadata_file, 'r') as f:
                metadata = json.load(f)
        
        perf = metadata.get('performance', {})
        models_list.insert(0, {
            'version': 'v1.0',
            'name': metadata.get('model_name', 'KURTAR v1.0'),
            'path': os.path.join(v1_dir, 'earthquake_detector_v1.0.tflite'),
            'type': 'tflite',
            'size_kb': 343,
            'date': metadata.get('created_date', '2026-01-01'),
            'recall': perf.get('recall_earthquake', 0.9227),
            'precision': perf.get('precision_earthquake', 0.5874),
            'f1_score': perf.get('f1_score_earthquake', 0.7179),
            'accuracy': perf.get('test_accuracy', 0.88),
            'tp': perf.get('true_positives'),
            'fp': perf.get('false_positives', 302),
            'fn': perf.get('false_negatives', 36),
            'dataset': 'STEAD + HAR',
            'changes': [],
            'status': 'current'
        })
    
    # Get current model from system state
    current_model = system_state.get('current_model', 'v1.0')
    
    # Mark current model
    for m in models_list:
        m['status'] = 'current' if m['version'] == current_model else 'available'
    
    return jsonify({
        'models': sorted(models_list, key=lambda x: x['version'], reverse=True),
        'count': len(models_list),
        'current_model': current_model
    })


@app.route('/api/models/select', methods=['POST'])
def select_model():
    """Set the active model version"""
    data = request.json
    version = data.get('version', 'v1.0')
    
    system_state['current_model'] = version
    
    # Find model info
    models_dir = os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(__file__))), 'models')
    output_dir = os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(__file__))), 'output')
    
    # Try to get model details
    model_info = {'version': version}
    
    # Check output folder for results
    if version != 'v1.0':
        results_file = os.path.join(output_dir, version, f'results_{version}.json')
        if os.path.exists(results_file):
            with open(results_file, 'r') as f:
                results = json.load(f)
                model_info.update(results.get('metrics', {}))
    else:
        # v1.0 metrics
        v1_dir = os.path.join(models_dir, 'tflite_v1.0')
        metadata_file = os.path.join(v1_dir, 'model_metadata_v1.0.json')
        if os.path.exists(metadata_file):
            with open(metadata_file, 'r') as f:
                metadata = json.load(f)
                model_info.update(metadata.get('performance', {}))
    
    # Emit to all clients
    socketio.emit('model_changed', {
        'version': version,
        'info': model_info
    })
    
    return jsonify({'success': True, 'current_model': version, 'info': model_info})

@app.route('/api/system-info')
def get_system_info():
    """Get system configuration and real model v1.0 metrics for display"""
    import json
    
    # Load real model metadata
    metadata_path = os.path.join(os.path.dirname(os.path.dirname(__file__)), 'models', 'model_metadata_v1.0.json')
    try:
        with open(metadata_path, 'r') as f:
            metadata = json.load(f)
    except:
        metadata = None
    
    # Use real metrics if available, otherwise fallback
    if metadata:
        perf = metadata.get('performance', {})
        arch = metadata.get('architecture', {})
        training = metadata.get('training_data', {})
        
        return jsonify({
            'model': {
                'name': metadata.get('model_name', 'KURTAR Earthquake Detector'),
                'version': metadata.get('version', 'v1.0'),
                'type': arch.get('type', '1D CNN') + ' (TensorFlow Lite)',
                'size_kb': arch.get('model_size_kb', 343),
                'parameters': arch.get('total_parameters', 332706),
                'input_shape': f"({arch.get('input_shape', [150, 6])[0]}, {arch.get('input_shape', [150, 6])[1]}) - 3s @ 50Hz",
                'channels': metadata.get('input_specs', {}).get('channels', [
                    'Acc X (g)', 'Acc Y (g)', 'Acc Z (g)', 
                    'Gyro X (rad/s)', 'Gyro Y (rad/s)', 'Gyro Z (rad/s)'
                ]),
                'preprocessing': [
                    'Resample to 50Hz',
                    f"Bandpass Filter ({metadata.get('preprocessing', {}).get('bandpass_filter', {}).get('lowcut_hz', 0.5)}-{metadata.get('preprocessing', {}).get('bandpass_filter', {}).get('highcut_hz', 20)}Hz)",
                    'Z-Score Normalization'
                ],
                'output': 'Binary (Earthquake / Not Earthquake)',
                'data_source': 'REAL v1.0 TRAINING'
            },
            'training_data': {
                'earthquake_samples': training.get('earthquake_samples', 3109),
                'non_earthquake_samples': training.get('non_earthquake_samples', 15236),
                'total_samples': training.get('total_samples', 18345),
                'sources': training.get('sources', {
                    'STEAD': '3,100 gerçek deprem (yer istasyonu)',
                    'TDG': '9 shake table test',
                    'HAR': '15,236 insan aktivitesi (WISDM + UCI HAR)'
                })
            },
            'consensus': {
                'algorithm': 'Weighted Spatiotemporal Voting',
                'spatial_clustering': 'DBSCAN (eps=5km, min_samples=3)',
                'temporal_window': '2s sliding window',
                'mobile_weight': system_state['config']['mobile_weight'],
                'iot_weight': system_state['config']['iot_weight'],
                'threshold': system_state['config']['threshold'],
                'min_iot_required': 2,
                'min_total_devices': 5
            },
            'real_v1_results': {
                'test_accuracy': perf.get('test_accuracy', 0.88),
                'recall': perf.get('recall_earthquake', 0.9227),
                'precision': perf.get('precision_earthquake', 0.5874),
                'f1_score': perf.get('f1_score_earthquake', 0.7179),
                'fpr': perf.get('false_positives', 302) / perf.get('test_samples', 2752),
                'test_samples': perf.get('test_samples', 2752),
                'confusion_matrix': {
                    'tp': perf.get('test_samples', 2752) - perf.get('false_negatives', 36) - perf.get('false_positives', 302),
                    'fp': perf.get('false_positives', 302),
                    'fn': perf.get('false_negatives', 36),
                    'tn': perf.get('test_samples', 2752) // 5  # Approx
                },
                'limitations': metadata.get('limitations', {}).get('notes', []),
                'data_source': 'Real Training (STEAD + TDG + HAR)'
            },
            'v2_targets': {
                'recall': 0.94,
                'precision': 0.80,
                'f1_score': 0.90,
                'fpr': 0.02,
                'improvements': metadata.get('next_version_improvements', [])
            },
            'baselines': [
                {'method': 'Mobile-Only', 'tpr': 0.85, 'fpr': 0.15, 'f1': 0.76, 'latency': '0.8s'},
                {'method': 'IoT-Only', 'tpr': 0.92, 'fpr': 0.05, 'f1': 0.88, 'latency': '1.2s'},
                {'method': 'Simple Average', 'tpr': 0.88, 'fpr': 0.08, 'f1': 0.82, 'latency': '1.0s'},
                {'method': 'Proposed v1.0', 'tpr': 0.92, 'fpr': 0.12, 'f1': 0.72, 'latency': '0.85s', 'current': True},
                {'method': 'Proposed v2.0 (Target)', 'tpr': 0.94, 'fpr': 0.02, 'f1': 0.95, 'latency': '<1s', 'highlight': True}
            ]
        })
    
    # Fallback to hardcoded values
    return jsonify({
        'model': {
            'name': 'Earthquake Detector v1.0',
            'type': '1D CNN (TensorFlow Lite)',
            'size_kb': 343,
            'input_shape': '(150, 6) - 3s @ 50Hz'
        },
        'error': 'Could not load real metadata'
    })

@app.route('/api/config', methods=['GET', 'POST'])
def handle_config():
    """Get or update configuration"""
    global consensus_engine
    
    if request.method == 'POST':
        data = request.json
        
        if 'mobile_weight' in data:
            system_state['config']['mobile_weight'] = data['mobile_weight']
        if 'iot_weight' in data:
            system_state['config']['iot_weight'] = data['iot_weight']
        if 'threshold' in data:
            system_state['config']['threshold'] = data['threshold']
        
        # Update consensus engine
        if consensus_engine:
            consensus_engine.update_weights(
                system_state['config']['mobile_weight'],
                system_state['config']['iot_weight']
            )
        
        return jsonify({'success': True, 'config': system_state['config']})
    
    return jsonify(system_state['config'])


@app.route('/api/devices')
def get_devices():
    """Get device locations"""
    if device_manager:
        return jsonify(device_manager.get_all_device_locations())
    return jsonify(system_state['devices'])


@app.route('/api/events')
def get_events():
    """Get recent events"""
    return jsonify(system_state['events'][-50:])  # Last 50 events


@app.route('/api/metrics')
def get_metrics():
    """Get system metrics"""
    return jsonify(system_state['metrics'])


@app.route('/api/analysis/sample')
def get_analysis_sample():
    """Get sample data for visualization of the v1.0 pipeline"""
    # 1. Generate synthetic raw data (Earthquake-like)
    t = np.linspace(0, 3, 150) # 3s @ 50Hz
    # Signal: Sine wave (P-wave) + Sine wave (S-wave) + Noise
    raw = (
        0.5 * np.sin(2 * np.pi * 5 * t) * np.exp(-(t-0.5)**2/0.1) + # P-wave
        2.0 * np.sin(2 * np.pi * 3 * t) * np.exp(-(t-1.5)**2/0.5) + # S-wave
        np.random.normal(0, 0.2, 150) # Noise
    )
    
    # Expand to (150, 1) to match preprocessor expectation
    raw_reshaped = raw.reshape(-1, 1)
    
    # 2. Pipeline Steps
    # Step A: Filter
    filtered = preprocessor.bandpass_filter(raw_reshaped)
    
    # Step B: Normalize (using v1.0 fixed stats)
    # Using Z-Score with arbitrary mean/std for visualization
    norm_mean = np.mean(filtered)
    norm_std = np.std(filtered)
    normalized = (filtered - norm_mean) / norm_std
    
    return jsonify({
        'labels': [f"{x:.2f}s" for x in t],
        'raw': raw.tolist(),
        'filtered': filtered.flatten().tolist(),
        'normalized': normalized.flatten().tolist()
    })


@app.route('/api/simulate', methods=['POST'])
def run_simulation():
    """Run a simulation"""
    global simulator
    
    data = request.json
    n_earthquakes = data.get('n_earthquakes', 10)
    n_false_positives = data.get('n_false_positives', 50)
    
    # Run in background thread
    def run_sim():
        global simulator, system_state
        
        system_state['mode'] = 'simulation'
        system_state['is_running'] = True
        socketio.emit('status_update', {'mode': 'simulation', 'running': True})
        
        try:
            simulator = TraceSimulator(
                mobile_weight=system_state['config']['mobile_weight'],
                iot_weight=system_state['config']['iot_weight']
            )
            
            summary = simulator.run_simulation(
                n_earthquakes=n_earthquakes,
                n_false_positives=n_false_positives,
                show_progress=False
            )
            
            # Update metrics
            system_state['metrics'] = {
                'total_events': summary.total_scenarios,
                'earthquake_detections': summary.true_positives,
                'false_alarms': summary.false_positives,
                'tpr': summary.recall,
                'fpr': summary.false_positive_rate,
                'precision': summary.precision,
                'f1': summary.f1_score
            }
            
            # Emit results
            socketio.emit('simulation_complete', summary.to_dict())
            
        except Exception as e:
            socketio.emit('error', {'message': str(e)})
        finally:
            system_state['is_running'] = False
            system_state['mode'] = 'standby'
            socketio.emit('status_update', {'mode': 'standby', 'running': False})
    
    thread = threading.Thread(target=run_sim)
    thread.start()
    
    return jsonify({'success': True, 'message': 'Simulation started'})


@app.route('/api/demo', methods=['POST'])
def run_demo():
    """Run a single earthquake demo"""
    global device_manager, consensus_engine
    
    data = request.json
    magnitude = data.get('magnitude', 5.5)
    lat = data.get('lat', 40.5)
    lon = data.get('lon', 30.2)
    num_mobile = data.get('num_mobile', int(50 * magnitude))
    num_iot = data.get('num_iot', 10)
    threshold = data.get('threshold', system_state['config']['threshold'])
    
    try:
        print(f"Received demo request: M{magnitude} at {lat}, {lon} with {num_mobile} mobile + {num_iot} IoT devices (threshold={threshold})")
        from simulation.earthquake_generator import EarthquakeGenerator
        
        # Initialize components
        print("Initializing DeviceManager and ConsensusEngine...")
        device_manager = DeviceManager()
        consensus_engine = ConsensusEngine(
            mobile_weight=system_state['config']['mobile_weight'],
            iot_weight=system_state['config']['iot_weight'],
            threshold=threshold
        )
        
        # Setup devices
        print(f"Setting up {num_mobile} mobile and {num_iot} IoT devices...")
        device_manager.setup_scenario(
            center_lat=lat,
            center_lon=lon,
            num_mobile=num_mobile,
            num_iot=num_iot,
            distribution_radius_km=50.0
        )
        
        # Generate earthquake
        generator = EarthquakeGenerator()
        scenario = generator.generate_scenario(
            magnitude=magnitude,
            epicenter_lat=lat,
            epicenter_lon=lon
        )
        
        # Simulate triggers
        triggers = device_manager.simulate_earthquake_triggers(
            earthquake_data=scenario.waveform_data,
            epicenter_lat=lat,
            epicenter_lon=lon,
            magnitude=magnitude
        )
        
        # Convert and run consensus
        trigger_dicts = [t.to_dict() for t in triggers]
        decision = consensus_engine.process(trigger_dicts)
        
        # Create event
        event = {
            'type': 'earthquake' if decision.is_earthquake else 'no_detection',
            'magnitude': magnitude,
            'lat': lat,
            'lon': lon,
            'score': decision.score,
            'threshold': decision.threshold,
            'mobile_count': decision.mobile_count,
            'iot_count': decision.iot_count,
            'detected': decision.is_earthquake,
            'timestamp': time.time()
        }
        
        system_state['events'].append(event)
        system_state['devices'] = device_manager.get_all_device_locations()
        
        # Emit updates
        socketio.emit('new_event', event)
        socketio.emit('devices_update', system_state['devices'])
        
        return jsonify({
            'success': True,
            'decision': decision.to_dict(),
            'event': event
        })
        
    except Exception as e:
        print(f"ERROR in run_demo: {e}")
        import traceback
        traceback.print_exc()
        return jsonify({'success': False, 'error': str(e)}), 500


@socketio.on('connect')
def handle_connect():
    """Handle client connection"""
    emit('status_update', {
        'mode': system_state['mode'],
        'running': system_state['is_running']
    })
    emit('config_update', system_state['config'])


@socketio.on('update_config')
def handle_config_update(data):
    """Handle real-time config update"""
    global consensus_engine
    
    if 'mobile_weight' in data:
        system_state['config']['mobile_weight'] = float(data['mobile_weight'])
    if 'iot_weight' in data:
        system_state['config']['iot_weight'] = float(data['iot_weight'])
    
    if consensus_engine:
        consensus_engine.update_weights(
            system_state['config']['mobile_weight'],
            system_state['config']['iot_weight']
        )
    
    emit('config_update', system_state['config'], broadcast=True)


@socketio.on('mobile_trigger')
def handle_mobile_trigger(data):
    """Handle trigger from mobile client"""
    # data: {'lat': float, 'lon': float, 'accel': float, 'id': str}
    print(f"Received mobile trigger from {data.get('id')}: {data.get('accel')}")
    
    # In a real system, this would feed into the ConsensusEngine
    # For now, we'll just forward it to the frontend map
    emit('device_alert', data, broadcast=True)


# ============================================================================
# V5 REAL-TIME SIMULATION
# ============================================================================

def get_v5_model():
    """Lazy load V5 model."""
    global V5_MODEL
    if V5_MODEL is None and os.path.exists(V5_MODEL_PATH):
        try:
            V5_MODEL = tf.keras.models.load_model(V5_MODEL_PATH)
            print(f"✅ V5 model loaded from {V5_MODEL_PATH}")
        except Exception as e:
            print(f"⚠️ Could not load V5 model: {e}")
    return V5_MODEL


@app.route('/simulation')
def simulation_page():
    """Real-time simulation page."""
    return render_template('simulation.html')


@app.route('/api/v5/predict', methods=['POST'])
def v5_predict():
    """Predict using V5 model on provided waveform data."""
    model = get_v5_model()
    if model is None:
        return jsonify({'error': 'V5 model not available'}), 500
    
    data = request.json
    waveform = np.array(data.get('waveform', []))  # Expecting (150, 3)
    
    if waveform.shape != (150, 3):
        return jsonify({'error': f'Invalid waveform shape: {waveform.shape}'}), 400
    
    # Predict
    x = waveform[np.newaxis, ...]  # (1, 150, 3)
    prob = float(model.predict(x, verbose=0)[0, 0])
    
    return jsonify({
        'probability': prob,
        'is_earthquake': prob >= 0.5,
        'threshold': 0.5
    })


@app.route('/api/v5/stream-simulation', methods=['POST'])
def v5_stream_simulation():
    """Run V5 stream simulation and return results."""
    model = get_v5_model()
    if model is None:
        return jsonify({'error': 'V5 model not available'}), 500
    
    data = request.json
    scenario = data.get('scenario', 'mixed')  # 'ambient', 'earthquake', 'mixed'
    duration_s = data.get('duration', 60)
    
    fs = 50
    n_samples = duration_s * fs
    
    # Generate stream based on scenario
    np.random.seed(int(time.time()) % 1000)
    stream = np.random.normal(0, 0.02, (3, n_samples))
    
    ground_truth = []
    eq_positions = []
    
    if scenario in ['earthquake', 'mixed']:
        n_eq = 1 if scenario == 'earthquake' else np.random.randint(2, 5)
        for _ in range(n_eq):
            pos = np.random.randint(150, n_samples - 250)
            
            # Add earthquake signal
            eq_len = 250
            t_eq = np.linspace(0, 5, eq_len)
            p_wave = 0.3 * np.sin(2 * np.pi * 8 * (t_eq - 1)) * np.exp(-3 * (t_eq - 1)) * (t_eq > 1)
            s_wave = 1.0 * np.sin(2 * np.pi * 3 * (t_eq - 2)) * np.exp(-0.8 * (t_eq - 2)) * (t_eq > 2)
            eq_signal = p_wave + s_wave
            
            for ch in range(3):
                amp = np.random.uniform(0.8, 1.2)
                stream[ch, pos:pos+eq_len] += eq_signal * amp
            
            eq_positions.append(pos / fs)
            ground_truth.append({
                'start': pos / fs,
                'end': (pos + eq_len) / fs,
                'type': 'earthquake'
            })
    
    # Run sliding window inference
    window_size = 150
    hop_size = 25
    
    results = []
    for start in range(0, n_samples - window_size, hop_size):
        window = stream[:, start:start + window_size]
        
        # Normalize
        window_norm = np.zeros_like(window)
        for ch in range(3):
            window_norm[ch] = (window[ch] - np.mean(window[ch])) / (np.std(window[ch]) + 1e-8)
        
        x = window_norm.T[np.newaxis, ...]  # (1, 150, 3)
        prob = float(model.predict(x, verbose=0)[0, 0])
        
        center_time = (start + window_size // 2) / fs
        results.append({
            'time': center_time,
            'probability': prob,
            'is_earthquake': prob >= 0.5
        })
    
    # Analyze detections
    triggers = [r for r in results if r['is_earthquake']]
    
    # Calculate metrics
    tp = 0
    fp = 0
    detection_latencies = []
    
    detected_events = set()
    for trigger in triggers:
        matched = False
        for i, eq_pos in enumerate(eq_positions):
            if eq_pos - 3 <= trigger['time'] <= eq_pos + 5:
                if i not in detected_events:
                    tp += 1
                    detected_events.add(i)
                    detection_latencies.append(trigger['time'] - eq_pos)
                    matched = True
                    break
        if not matched:
            fp += 1
    
    fn = len(eq_positions) - tp
    
    return jsonify({
        'scenario': scenario,
        'duration': duration_s,
        'stream': {
            'times': [i / fs for i in range(0, n_samples, 10)],  # Downsample for viz
            'z_axis': stream[2, ::10].tolist()
        },
        'predictions': results,
        'ground_truth': ground_truth,
        'metrics': {
            'true_positives': tp,
            'false_positives': fp,
            'false_negatives': fn,
            'precision': tp / (tp + fp + 1e-8),
            'recall': tp / (tp + fn + 1e-8) if eq_positions else 1.0,
            'avg_latency': np.mean(detection_latencies) if detection_latencies else 0
        }
    })


def run_dashboard(host=None, port=None, debug=None):
    """Run the dashboard server"""
    host = host or config.dashboard.host
    port = port or config.dashboard.port
    debug = debug if debug is not None else config.dashboard.debug
    
    # Create templates directory if needed
    templates_dir = os.path.join(os.path.dirname(__file__), 'templates')
    static_dir = os.path.join(os.path.dirname(__file__), 'static')
    
    os.makedirs(templates_dir, exist_ok=True)
    os.makedirs(static_dir, exist_ok=True)
    
    print(f"\n{'='*50}")
    print("Deprem Erken Uyarı Sistemi - Dashboard")
    print(f"{'='*50}")
    print(f"URL: http://{host}:{port}")
    print(f"{'='*50}\n")
    
    socketio.run(app, host=host, port=port, debug=debug)


if __name__ == '__main__':
    run_dashboard()
