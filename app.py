from flask import Flask, render_template, request, jsonify
from flask_socketio import SocketIO, emit
import os
import time
import random
import json

# Initialize Flask app and SocketIO
app = Flask(__name__)
app.config['SECRET_KEY'] = 'ppg-bpnet-secret-key'
socketio = SocketIO(app, cors_allowed_origins="*")

# Store for live data
live_data = {
    'timestamps': [],
    'sbp_values': [],
    'dbp_values': []
}

@app.route('/', methods=['GET', 'POST'])
def index():
    """Handle GET requests to render the main page and POST requests for file uploads."""
    if request.method == 'POST':
        if 'file' not in request.files:
            return jsonify({'error': 'No file part'})
        
        file = request.files['file']
        if file.filename == '':
            return jsonify({'error': 'No selected file'})
        
        # MOCK DATA: Using random values instead of actual PPG-BP model processing
        # TODO: Replace this with the actual PPG-BPNet model call
        mock_results = {
            'systolic': round(random.uniform(110, 140), 1),
            'diastolic': round(random.uniform(70, 90), 1),
            'timestamps': [i for i in range(10)],
            'values': [random.uniform(0.5, 1.5) for _ in range(10)]
        }
        
        return render_template('index.html', results=mock_results)
    
    return render_template('index.html')

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
    """Background task that simulates live BP data."""
    global monitoring_active
    monitoring_active = True
    
    while monitoring_active:
        # MOCK DATA: Generate random BP values instead of real sensor readings
        sbp = round(random.uniform(110, 140), 1)
        dbp = round(random.uniform(70, 90), 1)
        
        # Get current timestamp
        timestamp = time.time()
        
        # Store data points
        live_data['timestamps'].append(timestamp)
        live_data['sbp_values'].append(sbp)
        live_data['dbp_values'].append(dbp)
        
        # Keep only the last 30 data points
        if len(live_data['timestamps']) > 30:
            live_data['timestamps'] = live_data['timestamps'][-30:]
            live_data['sbp_values'] = live_data['sbp_values'][-30:]
            live_data['dbp_values'] = live_data['dbp_values'][-30:]
        
        # Emit the new data point to all clients
        socketio.emit('new_bp_data', {
            'timestamp': timestamp,
            'sbp': sbp,
            'dbp': dbp,
            'full_data': live_data
        })
        
        # Sleep for a few seconds to simulate real-time data
        socketio.sleep(2)

if __name__ == '__main__':
    socketio.run(app, debug=True) 