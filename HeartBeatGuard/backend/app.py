"""
=================================================================
SMARTHEALTH SENTINEL - COMPLETE BACKEND WITH ML
=================================================================
Enterprise Cardiac Emergency Response Platform
Real-time heart rate monitoring + ML Anomaly Detection
=================================================================
"""

# ============================================
# IMPORTS
# ============================================
from flask import Flask, request, jsonify, send_from_directory
from flask_socketio import SocketIO, emit
from flask_cors import CORS
import os
import json
import time
import random
from datetime import datetime
import threading

# ============================================
# ML IMPORTS - ISOLATION FOREST FOR ANOMALY DETECTION
# ============================================
from sklearn.ensemble import IsolationForest
import numpy as np

# ============================================
# INITIALIZE APP
# ============================================
app = Flask(__name__, static_folder='../frontend', static_url_path='')
CORS(app, origins="*")
socketio = SocketIO(app, cors_allowed_origins="*")

# ============================================
# DATA STORAGE (In-memory - no database needed!)
# ============================================
users = {}
heart_rates = []
emergency_alerts = []
user_locations = []

# ============================================
# ML MODEL - Isolation Forest for Anomaly Detection
# ============================================
print("\n" + "="*50)
print("🤖 SMARTHEALTH ML ENGINE INITIALIZING...")
print("="*50)

# Train on normal heart rates (60-100 BPM)
# Expanded dataset for better training
normal_hr = np.array([
    [65], [72], [68], [75], [71], [78], [82], [69], [73], [70],
    [67], [74], [76], [71], [68], [72], [75], [69], [73], [71],
    [66], [73], [77], [70], [69], [74], [71], [68], [72], [70],
    [64], [71], [69], [76], [72], [79], [81], [68], [74], [73],
    [63], [72], [70], [77], [71], [80], [83], [67], [75], [72]
])

# Also train with some borderline cases
borderline_hr = np.array([
    [58], [59], [101], [102], [98], [99]  # Edge cases
])

# Combine datasets
training_data = np.vstack([normal_hr, borderline_hr])

# Initialize Isolation Forest
anomaly_detector = IsolationForest(
    contamination=0.15,      # Expected 15% anomalies in production
    random_state=42,        # For reproducibility
    n_estimators=200,      # More trees = better accuracy
    max_samples='auto',
    bootstrap=True,
    n_jobs=-1,             # Use all CPU cores
    verbose=0
)

# Train the model
anomaly_detector.fit(training_data)

def is_anomalous(bpm):
    """
    Returns True if heart rate is anomalous (not normal)
    Uses Isolation Forest ML algorithm
    """
    prediction = anomaly_detector.predict([[bpm]])
    return prediction[0] == -1  # -1 = anomaly, 1 = normal

def get_anomaly_score(bpm):
    """
    Returns anomaly score (negative = more anomalous)
    Lower score = more likely to be anomaly
    """
    score = anomaly_detector.score_samples([[bpm]])
    return float(score[0])

def get_severity_with_ml(bpm, is_anomaly, score):
    """
    ML-enhanced severity classification
    Combines rule-based logic with ML confidence
    """
    # Base severity on BPM thresholds
    if bpm < 50:
        base_severity = 'critical'
        base_message = 'Severe bradycardia detected'
    elif bpm < 60:
        base_severity = 'moderate'
        base_message = 'Low heart rate detected'
    elif bpm <= 100:
        base_severity = 'normal'
        base_message = 'Heart rate is normal'
    elif bpm <= 120:
        base_severity = 'moderate'
        base_message = 'Elevated heart rate detected'
    elif bpm <= 140:
        base_severity = 'moderate'
        base_message = 'Tachycardia detected'
    else:
        base_severity = 'critical'
        base_message = 'Severe tachycardia detected'
    
    # Enhance with ML insights
    if is_anomaly:
        if score < -0.3:  # Strong anomaly signal
            if base_severity == 'moderate':
                base_severity = 'critical'
                base_message = f'🚨 ML CRITICAL: {base_message} (anomaly score: {score:.3f})'
            else:
                base_message = f'🚨 ML CONFIRMED: {base_message} (anomaly score: {score:.3f})'
        else:
            base_message = f'⚠️ ML DETECTED: {base_message}'
    else:
        if base_severity == 'normal':
            base_message = f'✅ {base_message}'
    
    return base_severity, base_message

print(f"✅ ML Model loaded successfully!")
print(f"📊 Training data: {len(training_data)} heart rate samples")
print(f"📊 Model parameters: contamination=0.15, n_estimators=200")
print(f"📊 Anomaly threshold: score < -0.3 = strong anomaly")
print("="*50 + "\n")

# ============================================
# HEALTH CHECK
# ============================================
@app.route('/health')
def health():
    return jsonify({
        'status': 'healthy',
        'timestamp': datetime.now().isoformat(),
        'service': 'SmartHealth Sentinel',
        'version': '2.0.0',
        'ml_model': {
            'type': 'Isolation Forest',
            'status': 'loaded',
            'training_samples': len(training_data)
        }
    })

# ============================================
# SERVE FRONTEND
# ============================================
@app.route('/')
def serve_frontend():
    return send_from_directory('../frontend', 'index.html')

# ============================================
# API ENDPOINTS
# ============================================

# Process heart rate data with ML
@app.route('/api/v1/heartrate', methods=['POST'])
def process_heartrate():
    try:
        data = request.json
        bpm = data.get('bpm', 72)
        user_id = data.get('userId', 'demo_user')
        
        # ML Anomaly Detection
        anomalous = is_anomalous(bpm)
        anomaly_score = get_anomaly_score(bpm)
        
        # Get ML-enhanced severity classification
        severity, message = get_severity_with_ml(bpm, anomalous, anomaly_score)
        
        # Calculate deviation from baseline
        baseline = 72  # Default baseline
        deviation = abs(bpm - baseline) / baseline
        
        # Create event with ML metadata
        event_id = f"evt_{int(time.time())}_{random.randint(1000, 9999)}"
        
        # Store heart rate with ML insights
        heart_rate_entry = {
            'id': event_id,
            'userId': user_id,
            'bpm': bpm,
            'severity': severity,
            'anomaly_detected': anomalous,
            'anomaly_score': anomaly_score,
            'deviation': deviation,
            'message': message,
            'timestamp': datetime.now().isoformat()
        }
        
        heart_rates.append(heart_rate_entry)
        
        # Keep only last 1000 readings
        if len(heart_rates) > 1000:
            heart_rates.pop(0)
        
        # Send WebSocket update
        socketio.emit('heartrate:analyzed', {
            'eventId': event_id,
            'severity': severity,
            'heartRate': bpm,
            'deviation': deviation,
            'message': message,
            'anomaly': anomalous,
            'anomalyScore': anomaly_score,
            'timestamp': datetime.now().isoformat()
        }, room=f"user_{user_id}")
        
        # If critical, trigger emergency
        if severity == 'critical':
            trigger_emergency(user_id, bpm, event_id)
            print(f"🤖 ML MODEL triggered emergency for HR {bpm} (score: {anomaly_score:.3f})")
        
        return jsonify({
            'success': True,
            'eventId': event_id,
            'severity': severity,
            'deviation': deviation,
            'message': message,
            'anomaly_detected': anomalous,
            'anomaly_score': anomaly_score,
            'timestamp': datetime.now().isoformat()
        })
        
    except Exception as e:
        print(f"❌ Error in process_heartrate: {e}")
        return jsonify({'success': False, 'error': str(e)}), 500

# Get user events
@app.route('/api/v1/events', methods=['GET'])
def get_events():
    user_id = request.args.get('userId', 'demo_user')
    limit = int(request.args.get('limit', 50))
    
    user_events = [e for e in heart_rates if e['userId'] == user_id][:limit]
    
    return jsonify({
        'success': True,
        'events': user_events,
        'total': len(user_events)
    })

# Get ML analytics
@app.route('/api/v1/ml/analytics', methods=['GET'])
def get_ml_analytics():
    """Return ML model performance metrics"""
    total_readings = len(heart_rates)
    anomalies_detected = sum(1 for hr in heart_rates if hr.get('anomaly_detected', False))
    critical_events = sum(1 for hr in heart_rates if hr.get('severity') == 'critical')
    
    return jsonify({
        'success': True,
        'total_readings': total_readings,
        'anomalies_detected': anomalies_detected,
        'critical_events': critical_events,
        'anomaly_rate': anomalies_detected / total_readings if total_readings > 0 else 0,
        'model_type': 'Isolation Forest',
        'training_samples': len(training_data)
    })

# Update location
@app.route('/api/v1/emergency/location', methods=['POST'])
def update_location():
    try:
        data = request.json
        user_id = data.get('userId', 'demo_user')
        
        user_locations[user_id] = {
            'lat': data.get('lat', 37.7749),
            'lng': data.get('lng', -122.4194),
            'accuracy': data.get('accuracy', 10),
            'timestamp': datetime.now().isoformat()
        }
        
        return jsonify({
            'success': True,
            'message': 'Location updated'
        })
    except Exception as e:
        return jsonify({'success': False, 'error': str(e)}), 500

# Find nearby hospitals
@app.route('/api/v1/emergency/hospitals', methods=['GET'])
def find_hospitals():
    lat = request.args.get('lat', 37.7749)
    lng = request.args.get('lng', -122.4194)
    
    # Sample hospitals (in real app, this would call Google Maps API)
    hospitals = [
        {
            'id': 'hosp_001',
            'name': 'City General Hospital',
            'address': '100 Medical Center Dr',
            'distance': 1200,
            'eta': 180,
            'phone': '(555) 123-4567',
            'coordinates': {'lat': 37.785, 'lng': -122.426}
        },
        {
            'id': 'hosp_002',
            'name': 'St. Mary\'s Medical Center',
            'address': '450 Health Sciences Rd',
            'distance': 2100,
            'eta': 300,
            'phone': '(555) 234-5678',
            'coordinates': {'lat': 37.765, 'lng': -122.435}
        },
        {
            'id': 'hosp_003',
            'name': 'University Hospital',
            'address': '505 Parnassus Ave',
            'distance': 3500,
            'eta': 420,
            'phone': '(555) 345-6789',
            'coordinates': {'lat': 37.762, 'lng': -122.458}
        }
    ]
    
    return jsonify({
        'success': True,
        'hospitals': hospitals,
        'count': len(hospitals)
    })

# ============================================
# EMERGENCY FUNCTIONS
# ============================================
def trigger_emergency(user_id, bpm, event_id):
    """Trigger emergency protocol"""
    
    emergency_id = f"emg_{int(time.time())}_{random.randint(1000, 9999)}"
    
    # Create emergency alert
    alert = {
        'id': emergency_id,
        'userId': user_id,
        'eventId': event_id,
        'heartRate': bpm,
        'timestamp': datetime.now().isoformat(),
        'status': 'active'
    }
    
    emergency_alerts.append(alert)
    
    # Get location (simulated)
    location = user_locations.get(user_id, {
        'lat': 37.7749,
        'lng': -122.4194,
        'accuracy': 10
    })
    
    # Get nearby hospitals (simulated)
    hospitals = [
        {
            'id': 'hosp_001',
            'name': 'City General Hospital',
            'distance': 1200,
            'eta': 180
        }
    ]
    
    # Create emergency protocol
    protocol = {
        'activated': True,
        'timestamp': datetime.now().isoformat(),
        'actions': {
            'gpsAcquired': True,
            'nearbyHospitals': hospitals,
            'emergencyCalled': True,
            'contactsAlerted': ['Emergency Contact'],
            'emrShared': True
        }
    }
    
    # Send WebSocket broadcast
    socketio.emit('emergency:activated', {
        'eventId': event_id,
        'emergencyId': emergency_id,
        'severity': 'critical',
        'heartRate': bpm,
        'protocol': protocol,
        'timestamp': datetime.now().isoformat()
    }, room=f"user_{user_id}")
    
    # Also broadcast to monitoring room
    socketio.emit('critical:alert', {
        'userId': user_id,
        'eventId': event_id,
        'heartRate': bpm,
        'location': location
    }, room='monitoring')
    
    print(f"🚨 EMERGENCY ACTIVATED for user {user_id} - Heart Rate: {bpm} BPM")
    return emergency_id

# ============================================
# WEBSOCKET EVENTS
# ============================================
@socketio.on('connect')
def handle_connect():
    user_id = request.args.get('userId', 'demo_user')
    print(f"✅ Client connected: {request.sid} (User: {user_id})")
    
    # Join user room
    join_room(f"user_{user_id}")
    
    emit('connected', {
        'status': 'connected',
        'sid': request.sid,
        'timestamp': datetime.now().isoformat()
    })

@socketio.on('disconnect')
def handle_disconnect():
    print(f"❌ Client disconnected: {request.sid}")

@socketio.on('heartrate:stream')
def handle_heartrate_stream(data):
    """Handle real-time heart rate stream from wearables"""
    try:
        user_id = request.args.get('userId', 'demo_user')
        bpm = data.get('bpm', 72)
        
        print(f"❤️ Heart rate received: {bpm} BPM (User: {user_id})")
        
        # ML Anomaly Detection
        anomalous = is_anomalous(bpm)
        anomaly_score = get_anomaly_score(bpm)
        
        # Get ML-enhanced severity
        severity, message = get_severity_with_ml(bpm, anomalous, anomaly_score)
        
        # Send analysis back
        event_id = f"evt_{int(time.time())}_{random.randint(1000, 9999)}"
        
        emit('heartrate:analyzed', {
            'eventId': event_id,
            'severity': severity,
            'heartRate': bpm,
            'message': message,
            'anomaly': anomalous,
            'anomalyScore': anomaly_score,
            'timestamp': datetime.now().isoformat()
        })
        
        # If critical, trigger emergency
        if severity == 'critical':
            trigger_emergency(user_id, bpm, event_id)
            
    except Exception as e:
        print(f"Error processing heart rate: {e}")
        emit('error', {'message': 'Failed to process heart rate'})

@socketio.on('location:update')
def handle_location_update(location):
    """Handle location updates from client"""
    try:
        user_id = request.args.get('userId', 'demo_user')
        
        user_locations[user_id] = {
            'lat': location.get('lat', 37.7749),
            'lng': location.get('lng', -122.4194),
            'accuracy': location.get('accuracy', 10),
            'timestamp': datetime.now().isoformat()
        }
        
        emit('location:updated', {
            'timestamp': datetime.now().isoformat()
        })
        
        print(f"📍 Location updated for user {user_id}")
        
    except Exception as e:
        print(f"Error updating location: {e}")
        emit('error', {'message': 'Failed to update location'})

@socketio.on('context:submit')
def handle_context_submit(data):
    """Handle user context submission"""
    try:
        event_id = data.get('eventId')
        context = data.get('context', {})
        
        print(f"📝 Context received for event {event_id}: {context}")
        
        emit('context:received', {
            'eventId': event_id,
            'status': 'processed',
            'timestamp': datetime.now().isoformat()
        })
        
    except Exception as e:
        print(f"Error processing context: {e}")
        emit('error', {'message': 'Failed to process context'})

# Helper for joining rooms
def join_room(room):
    """Join a Socket.IO room"""
    from flask_socketio import join_room as socketio_join_room
    socketio_join_room(room)

# ============================================
# SAMPLE DATA GENERATOR
# ============================================
def generate_sample_data():
    """Generate sample heart rate data for testing"""
    print("📊 Generating sample data...")
    
    # Create demo user
    users['demo_user'] = {
        'id': 'demo_user',
        'name': 'Demo User',
        'email': 'demo@smarthealth.com',
        'baseline': 72
    }
    
    # Generate sample heart rates with ML analysis
    sample_rates = [68, 72, 75, 71, 69, 73, 78, 82, 71, 74, 
                    45, 120, 150, 55, 135, 48, 118, 158]
    
    for i, bpm in enumerate(sample_rates):
        anomalous = is_anomalous(bpm)
        score = get_anomaly_score(bpm)
        severity, message = get_severity_with_ml(bpm, anomalous, score)
        
        heart_rates.append({
            'id': f"sample_{i}",
            'userId': 'demo_user',
            'bpm': bpm,
            'severity': severity,
            'anomaly_detected': anomalous,
            'anomaly_score': score,
            'message': message,
            'timestamp': datetime.now().isoformat()
        })
    
    print(f"✅ Sample data ready! {len(heart_rates)} heart rate readings")
    print(f"👤 Demo user created")
    print(f"🤖 ML model processed {len(heart_rates)} samples")

# ============================================
# MAIN ENTRY POINT
# ============================================
if __name__ == '__main__':
    print("""
    ┌─────────────────────────────────────────────────────┐
    │                                                     │
    │   🚀 SMARTHEALTH SENTINEL v2.0 - WITH ML          │
    │                                                     │
    │   🔥 Enterprise Cardiac Emergency Platform         │
    │   🤖 Isolation Forest Anomaly Detection            │
    │   📡 WebSocket + REST API                         │
    │                                                     │
    └─────────────────────────────────────────────────────┘
    """)
    
    # Generate sample data
    generate_sample_data()
    
    # Print endpoints
    print("\n📌 ENDPOINTS:")
    print("   ├─ Frontend:     http://localhost:5000")
    print("   ├─ Health:       http://localhost:5000/health")
    print("   ├─ API:          http://localhost:5000/api/v1/heartrate")
    print("   ├─ ML Analytics: http://localhost:5000/api/v1/ml/analytics")
    print("   └─ WebSocket:    ws://localhost:5000")
    
    print("\n🚀 Starting server...")
    print("   Press Ctrl+C to stop\n")
    
    # Start server
    socketio.run(app, 
                host='0.0.0.0', 
                port=5000, 
                debug=True,
                allow_unsafe_werkzeug=True)