from flask import Flask, render_template, request, jsonify, send_from_directory
from flask_socketio import SocketIO, emit
import os
import time
import json
import numpy as np
import pandas as pd
from werkzeug.utils import secure_filename
from model_utils import PPG_BP_Predictor

# Constants
UPLOAD_FOLDER = 'uploads'
ALLOWED_EXTENSIONS = {'csv', 'txt', 'npy', 'mat'}
MODEL_PATH = r'new model train\best_enhanced_cnn_bilstm_model.h5'
# Global toggle: default to mock unless explicitly enabled
USE_REAL_MODEL = os.environ.get('USE_REAL_MODEL', 'false').lower() in ('1', 'true', 'yes')
USE_REAL_API_DEFAULT = os.environ.get('USE_REAL_API', 'false').lower() in ('1', 'true', 'yes')

# Initialize Flask app and SocketIO
app = Flask(__name__)
app.config['SECRET_KEY'] = 'ppg-bpnet-secret-key'
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER
socketio = SocketIO(app, cors_allowed_origins="*")

# Create uploads directory if it doesn't exist
os.makedirs(UPLOAD_FOLDER, exist_ok=True)

# Initialize the BP predictor
try:
    bp_predictor = PPG_BP_Predictor(MODEL_PATH)
    print("Successfully loaded BP prediction model")
except Exception as e:
    print(f"Error loading model: {str(e)}")

# Store for live data
live_data = {
    'timestamps': [],
    'sbp_values': [],
    'dbp_values': []
}

def allowed_file(filename):
    return '.' in filename and \
           filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

def process_uploaded_file(file):
    """Process uploaded file and return PPG signal."""
    filename = secure_filename(file.filename)
    filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
    file.save(filepath)
    
    try:
        # Handle different file formats
        if filename.endswith('.csv'):
            data = pd.read_csv(filepath)
            # Assuming first column is the PPG signal
            ppg_signal = data.iloc[:, 0].values
        elif filename.endswith('.npy'):
            ppg_signal = np.load(filepath)
        elif filename.endswith('.mat'):
            from scipy import io
            mat = io.loadmat(filepath)
            # Try to find PPG data in the .mat file
            ppg_key = next((k for k in mat.keys() if 'ppg' in k.lower()), None)
            ppg_signal = mat[ppg_key].flatten() if ppg_key else None
        else:  # .txt
            ppg_signal = np.loadtxt(filepath)
            
        return ppg_signal.astype(np.float32)
    except Exception as e:
        print(f"Error processing file: {str(e)}")
        return None

@app.route('/', methods=['GET'])
def index():
    """Render the main page."""
    return render_template('index.html')

@app.route('/analyze', methods=['POST'])
def analyze():
    """Handle file uploads and return analysis results as JSON."""
    if 'file' not in request.files:
        return jsonify({'error': 'No file part'}), 400
    
    file = request.files['file']
    if file.filename == '':
        return jsonify({'error': 'No selected file'}), 400
        
    if not allowed_file(file.filename):
        return jsonify({
            'error': f'Invalid file type. Allowed types: {ALLOWED_EXTENSIONS}'
        }), 400
    
    try:
        # Process the uploaded file
        ppg_signal = process_uploaded_file(file)
        if ppg_signal is None or len(ppg_signal) == 0:
            return jsonify({'error': 'Could not extract PPG data from file'}), 400
            
        # Determine sampling rate
        fs = getattr(bp_predictor, 'fs', 125)

        # Prepare a 5-second BP time series (1 Hz points)
        def mock_series(base_sbp, base_dbp):
            # Create physiologically plausible slow-varying sequence over 5s
            t = np.arange(1, 6)
            sbp_noise = np.linspace(-2.0, 2.0, 5) + np.random.normal(0, 0.8, 5)
            dbp_noise = np.linspace(1.0, -1.0, 5) + np.random.normal(0, 0.6, 5)
            sbp_series = np.clip(base_sbp + sbp_noise, 90, 190)
            dbp_series = np.clip(base_dbp + dbp_noise, 55, 120)
            return t.tolist(), sbp_series.astype(float).tolist(), dbp_series.astype(float).tolist()

        # If using real model, compute overlapping 5s windows to get 5 per-second estimates
        def model_series(signal_arr):
            t_points, sbps, dbps = [], [], []
            window_len = 5 * fs
            if len(signal_arr) < window_len:
                # pad with last value if too short
                pad_len = window_len - len(signal_arr)
                signal_arr = np.pad(signal_arr, (0, pad_len), mode='edge')
            end_idx = len(signal_arr)
            for i in range(5):
                # 1 Hz points across the last 5 seconds, overlapping 5s windows
                seg_end = end_idx - (4 - i) * fs
                seg_start = seg_end - window_len
                if seg_start < 0:
                    seg_start = 0
                    seg_end = window_len
                segment = signal_arr[seg_start:seg_end]
                try:
                    pred = bp_predictor.predict_bp(segment)
                    sbps.append(float(pred['systolic']))
                    dbps.append(float(pred['diastolic']))
                except Exception:
                    # fallback small jitter around last or base
                    base = bp_predictor.predict_bp(signal_arr)
                    sb_base, db_base = float(base['systolic']), float(base['diastolic'])
                    jitter_s = np.random.normal(0, 1.0)
                    jitter_d = np.random.normal(0, 0.7)
                    sbps.append(sb_base + jitter_s)
                    dbps.append(db_base + jitter_d)
                t_points.append(i + 1)
            return t_points, sbps, dbps

        # Baseline prediction for summary cards
        base_pred = bp_predictor.predict_bp(ppg_signal)
        
        # Generate timestamps for visualization
        timestamps = np.linspace(0, len(ppg_signal)/fs, len(ppg_signal))
        
        # Sample signal for visualization (downsample if too long)
        max_points = 1000
        if len(ppg_signal) > max_points:
            step = len(ppg_signal) // max_points
            ppg_display = ppg_signal[::step]
            timestamps = timestamps[::step]
        else:
            ppg_display = ppg_signal
        
        # Decide whether to use real model or mock for this request
        # Priority: request args/form 'use_real' overrides global defaults
        req_flag = request.args.get('use_real') or request.form.get('use_real')
        if req_flag is not None:
            use_real_api = str(req_flag).lower() in ('1', 'true', 'yes')
        else:
            use_real_api = USE_REAL_API_DEFAULT or USE_REAL_MODEL

        # Build 5-second BP series (mock or model)
        if use_real_api:
            t_series, sbp_series, dbp_series = model_series(ppg_signal)
            confidence = 0.9  # placeholder; replace with model-derived confidence if available
        else:
            t_series, sbp_series, dbp_series = mock_series(base_pred['systolic'], base_pred['diastolic'])
            confidence = 0.93

        # Final response payload
        response = {
            'systolic': float(base_pred['systolic']),
            'diastolic': float(base_pred['diastolic']),
            'bp_category': base_pred['bp_category'],
            'confidence': float(confidence),
            'timestamps': timestamps.tolist(),
            'ppg_values': ppg_display.tolist(),
            'sampling_rate': fs,
            'bp_series': {
                't': t_series,
                'sbp': sbp_series,
                'dbp': dbp_series
            },
            'use_real_model': bool(use_real_api),
            'success': True
        }

        return jsonify(response)
        
    except Exception as e:
        import traceback
        traceback.print_exc()
        return jsonify({
            'success': False,
            'error': f'Error processing file: {str(e)}'
        }), 500

@socketio.on('connect')
def handle_connect():
    """Handle client connection event."""
    print('Client connected')

@socketio.on('disconnect')
def handle_disconnect():
    """Handle client disconnect event."""
    print('Client disconnected')

@socketio.on('start_monitoring')
def handle_start_monitoring():
    """Start the background task for simulating live data."""
    print('Starting BP monitoring')
    socketio.start_background_task(generate_live_data)

@socketio.on('stop_monitoring')
def handle_stop_monitoring():
    """Stop the live monitoring by setting the global flag."""
    print('Stopping BP monitoring')
    global monitoring_active
    monitoring_active = False

def generate_live_data():
    """Background task that processes live PPG data and predicts BP."""
    global monitoring_active
    monitoring_active = True
    
    # Buffer for accumulating PPG samples
    ppg_buffer = np.array([], dtype=np.float32)
    last_prediction_time = time.time()
    
    while monitoring_active:
        try:
            # In a real implementation, you would get new samples from a hardware device here
            # For now, we'll simulate getting a chunk of data
            socketio.sleep(0.1)  # Simulate 100ms between chunks
            
            # Simulate getting new PPG data (replace with actual hardware read)
            new_samples = np.random.normal(0, 0.1, 25).astype(np.float32)  # 25 samples at 250Hz = 100ms
            
            # Add to buffer
            ppg_buffer = np.concatenate([ppg_buffer, new_samples])
            
            # Check if we have enough data for a prediction (5 seconds at 125Hz = 625 samples)
            current_time = time.time()
            if len(ppg_buffer) >= 625 and (current_time - last_prediction_time) >= 1.0:
                # Get the last 5 seconds of data
                segment = ppg_buffer[-625:]
                
                # Predict BP
                prediction = bp_predictor.predict_bp(segment)
                
                # Get current timestamp
                timestamp = time.time()
                
                # Store data points
                live_data['timestamps'].append(timestamp)
                live_data['sbp_values'].append(prediction['systolic'])
                live_data['dbp_values'].append(prediction['diastolic'])
                
                # Keep only the last 30 data points
                if len(live_data['timestamps']) > 30:
                    live_data['timestamps'] = live_data['timestamps'][-30:]
                    live_data['sbp_values'] = live_data['sbp_values'][-30:]
                    live_data['dbp_values'] = live_data['dbp_values'][-30:]
                
                # Emit the new data point to all clients
                socketio.emit('new_bp_data', {
                    'timestamp': timestamp,
                    'sbp': prediction['systolic'],
                    'dbp': prediction['diastolic'],
                    'bp_category': prediction['bp_category'],
                    'full_data': live_data
                })
                
                last_prediction_time = current_time
                
        except Exception as e:
            print(f"Error in live data generation: {str(e)}")
            socketio.emit('error', {'message': str(e)})
            monitoring_active = False

@app.route('/download-sample')
def download_sample():
    """Route to download a sample PPG data file."""
    sample_dir = os.path.join(app.root_path, 'static', 'samples')
    return send_from_directory(sample_dir, 'sample_ppg.csv', as_attachment=True)

if __name__ == '__main__':
    # Create a samples directory if it doesn't exist
    samples_dir = os.path.join(app.root_path, 'static', 'samples')
    os.makedirs(samples_dir, exist_ok=True)
    
    # Create a sample PPG file if it doesn't exist
    sample_file = os.path.join(samples_dir, 'sample_ppg.csv')
    if not os.path.exists(sample_file):
        # Generate a realistic-looking PPG signal
        fs = 125  # 125 Hz sampling rate
        t = np.linspace(0, 30, 30*fs)  # 30 seconds of data
        ppg = 0.5 * np.sin(2 * np.pi * 1.2 * t)  # 1.2 Hz heart rate
        ppg += 0.1 * np.random.normal(0, 1, len(t))  # Add some noise
        
        # Save as CSV
        np.savetxt(sample_file, ppg, delimiter=',')
    
    # Run the app
    socketio.run(app, debug=True, host='0.0.0.0', port=5000)