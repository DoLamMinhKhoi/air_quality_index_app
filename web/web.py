from flask import Flask, request, jsonify, render_template_string
from flask_cors import CORS
import pandas as pd
import numpy as np
import torch
import torch.nn as nn
from datetime import datetime, timedelta
import math
import os
import traceback
from scipy.interpolate import Rbf
from sklearn.preprocessing import MinMaxScaler
# from data_crawl import fetch_all_aqi

app = Flask(__name__)
CORS(app)

# Configuration
CSV_PATH = r"D:\MinhKhoi\SIU\DoAnTotNghiep\Project\data\hourly_aqi_data.csv"
CSV_INPUT_PATH = r"D:\MinhKhoi\SIU\DoAnTotNghiep\Project\data\input.csv"
CSV_FILE = "latest_aqi_snapshot.csv"
MODEL_PATH_TEMPLATE = r"D:\MinhKhoi\SIU\DoAnTotNghiep\Project\models\lstm_model_{station}.pt"

# Station coordinates for interpolation
STATION_COORDS = {
    "District 1": [10.7725, 106.7025],
    "District 2": [10.779354, 106.751480],
    "District 3": [10.782537, 106.683256],
    "District 4": [10.7550, 106.7050],
    "District 5": [10.7574, 106.6739],
    "District 6": [10.7547, 106.6500],
    "District 7": [10.745781, 106.714925],
    "District 9": [10.854345, 106.810686],
    "District 10": [10.7708, 106.6642],
    "District 11": [10.7653, 106.6414],
    "Tan Phu": [10.7817, 106.6111],
    "Binh Thanh": [10.8167, 106.7000],
    "Thu Duc": [10.871060, 106.826204]
}

# Target districts for prediction
TARGET_DISTRICTS = {
    "Phu Nhuan": [10.8000, 106.6667],
    "District 12": [10.8500, 106.6500],
    "Tan Binh": [10.8100, 106.6500],
    "Binh Tan": [10.7600, 106.6000],
    "Go Vap": [10.8300, 106.6600],
    "District 8": [10.7240, 106.6286]
}

# LSTM Model class (you may need to adjust this based on your actual model architecture)
class LSTMModel(nn.Module):
    def __init__(self, input_size=72, hidden_size=64, output_size=24, num_layers=1):
        super(LSTMModel, self).__init__()
        self.lstm = nn.LSTM(input_size, hidden_size, num_layers=num_layers, batch_first=True)
        self.fc = nn.Linear(hidden_size, output_size)

    def forward(self, x):
        x = x.unsqueeze(1)  # Add sequence dimension: (batch_size, seq_len=1, input_size)
        out, _ = self.lstm(x)  # out shape: (batch_size, seq_len, hidden_size)
        out = self.fc(out[:, -1, :])  # Take the last timestep
        return out  # (batch_size, output_size)

def load_aqi_data():
    """Load and preprocess AQI data from CSV"""
    try:
        df = pd.read_csv(CSV_PATH)
        df['timestamp'] = pd.to_datetime(df['timestamp'], format='%Y-%m-%d %H:%M:%S')
        return df
    except Exception as e:
        print(f"Error loading CSV data: {e}")
        return None
    
def load_input_data():
    """Load and preprocess AQI data from CSV"""
    try:
        df = pd.read_csv(CSV_INPUT_PATH)
        df['timestamp'] = pd.to_datetime(df['timestamp'], format='%m/%d/%Y %H:%M')
        return df
    except Exception as e:
        print(f"Error loading CSV data: {e}")
        return None

def get_recent_aqi_data(station, hours=72):
    """Get the last 72 AQI records for a station, with NaN handling"""
    try:
        df = load_input_data()
        if df is None:
            return np.full(hours, 50)  # Return default values if load failed

        # Filter only data for the specific station
        station = station.replace('_', ' ').title()
        df_station = df[df['station'] == station].sort_values('timestamp')

        if df_station.empty:
            print(f"No data available for station: {station}")
            return np.full(hours, 50)

        # Preprocess NaN values
        last_valid_index = df_station['aqi'].last_valid_index()
        df_station = df_station.loc[:last_valid_index]
        df_station['aqi'] = df_station['aqi'].fillna(df_station['aqi'].rolling(5, min_periods=1).mean())
        df_station['aqi'] = df_station['aqi'].interpolate(method='linear')
        df_station['aqi'] = df_station['aqi'].round(0)

        # Take the last 'hours' records
        last_data = df_station['aqi'].iloc[-hours:]

        if len(last_data) < hours:
            print(f"Warning: Only {len(last_data)} records available for {station}. Padding with last known value.")
            if len(last_data) > 0:
                last_value = last_data.iloc[-1]
                padded_data = np.pad(last_data.values, (hours - len(last_data), 0), constant_values=last_value)
            else:
                padded_data = np.full(hours, 50)
            return padded_data

        return last_data.values

    except Exception as e:
        print(f"Error getting recent AQI data: {e}")
        return np.full(hours, 50)

def load_lstm_model(station):
    """Load LSTM model for specific station and prediction hour"""
    try:
        # Replace <station_name> with actual station name (remove spaces and make lowercase)
        station_name = station.lower().replace(' ', '_')
        model_path = MODEL_PATH_TEMPLATE.format(station=station_name)
        
        if not os.path.exists(model_path):
            print(f"Model not found: {model_path}")
            return None
            
        model = LSTMModel()
        model.load_state_dict(torch.load(model_path, map_location=torch.device('cpu')))
        model.eval()
        return model
    except Exception as e:
        print(f"Error loading model for {station}")
        return None

def predict_aqi_for_station(station):
    """Predict AQI for next 24 hours for a given station using single multi-output model"""
    try:
        # Load recent input data
        recent_data = get_recent_aqi_data(station, 72)
        print("Recent AQI input:", recent_data)

        if recent_data is None or len(recent_data) < 72:
            return [50 + np.random.normal(0, 10) for _ in range(24)]

        # Step 1: Load station-specific training data
        station_file = f"D:/MinhKhoi/SIU/DoAnTotNghiep/Project/data/aqi_station_{station.lower().replace(' ', '_')}.csv"
        try:
            df = pd.read_csv(station_file)
        except Exception as e:
            print(f"Could not load training data for {station}: {e}")
            return [max(0, np.mean(recent_data[-24:]) + np.random.normal(0, 5)) for _ in range(24)]

        # Step 2: Fit scalers for this station
        X_train = df[[f"AQI_t-{i}" for i in range(71, 0, -1)] + ['AQI_t0']].values
        Y_train = df[[f"AQI_t+{i}" for i in range(1, 25)]].values
        valid_idx = ~np.isnan(X_train).any(axis=1) & ~np.isnan(Y_train).any(axis=1)
        X_train, Y_train = X_train[valid_idx], Y_train[valid_idx]

        scaler_X = MinMaxScaler()
        scaler_Y = MinMaxScaler()
        scaler_X.fit(X_train)
        scaler_Y.fit(Y_train)

        # Step 3: Load model
        model = load_lstm_model(station)
        if model is None:
            return [max(0, np.mean(recent_data[-24:]) + np.random.normal(0, 5)) for _ in range(24)]

        model.eval()

        # Step 4: Scale input and predict
        input_scaled = scaler_X.transform([recent_data])  # [1, 72]
        input_tensor = torch.FloatTensor(input_scaled)

        with torch.no_grad():
            output = model(input_tensor)  # [1, 24]
            output_np = output.numpy()

        # Step 5: Unscale output
        predictions = scaler_Y.inverse_transform(output_np)[0]  # [24]
        return [max(0, float(p)) for p in predictions]

    except Exception as e:
        print(f"Error predicting AQI for {station}: {e}")
        return [50 + np.random.normal(0, 10) for _ in range(24)]

def idw_predict(target_lat, target_lng, known_points, power=2):
    """Inverse Distance Weighting prediction for a point"""
    weighted_sum = 0
    weight_total = 0

    for station, (lat, lng, aqi_value) in known_points.items():
        try:
            dist = math.sqrt((lat - target_lat) ** 2 + (lng - target_lng) ** 2)
            if dist == 0:
                return aqi_value
            weight = 1 / (dist ** power)
            weighted_sum += weight * float(aqi_value)
            weight_total += weight
        except:
            continue

    return weighted_sum / weight_total if weight_total > 0 else None

@app.route('/')
def index():
    """Serve the main HTML page"""
    try:
        with open('index.html', 'r', encoding='utf-8') as f:
            return f.read()
    except FileNotFoundError:
        # If separate HTML file doesn't exist, return the embedded HTML
        return render_template_string("""
        <!-- You would put the full HTML content here if serving from Flask -->
        <h1>Ho Chi Minh City Air Quality Prediction</h1>
        <p>Please ensure the HTML file is properly set up or use the artifact provided.</p>
        """)

@app.route('/api/historical-data', methods=['POST'])
def get_historical_data():
    """Get historical AQI data for selected date and stations"""
    try:
        data = request.get_json()
        date_str = data.get('date')
        stations = data.get('stations', [])
        
        if not date_str or not stations:
            return jsonify({'error': 'Date and stations are required'}), 400
        
        df = load_aqi_data()
        if df is None:
            return jsonify({'error': 'Failed to load AQI data'}), 500
        
        # Parse the date and get data for the entire day
        target_date = datetime.strptime(date_str, '%Y-%m-%d').date()
        filtered_data = df[df['timestamp'].dt.date == target_date]
        
        if filtered_data.empty:
            return jsonify({'error': f'No data available for {date_str}'}), 404
        
        # Prepare response data
        timestamps = []
        station_data = {station: [] for station in stations}
        
        # Get all unique timestamps for the day
        unique_timestamps = sorted(filtered_data['timestamp'].unique())
        
        for timestamp in unique_timestamps:
            timestamps.append(timestamp.strftime('%H:%M'))
            hour_data = filtered_data[filtered_data['timestamp'] == timestamp]
            
            for station in stations:
                station_row = hour_data[hour_data['station'] == station]
                if not station_row.empty:
                    station_data[station].append(station_row['aqi'].iloc[0])
                else:
                    station_data[station].append(None)
        
        # Format data for Chart.js
        response_data = {
            'timestamps': timestamps,
            'stations': []
        }
        
        for station in stations:
            response_data['stations'].append({
                'name': station,
                'values': station_data[station]
            })
        
        return jsonify(response_data)
        
    except Exception as e:
        print(f"Error in get_historical_data: {e}")
        print(traceback.format_exc())
        return jsonify({'error': str(e)}), 500

@app.route('/api/predict', methods=['POST'])
def predict_aqi():
    """Predict AQI for next 24 hours"""
    try:
        data = request.get_json()
        station = data.get('station')
        
        if not station:
            return jsonify({'error': 'Station is required'}), 400
        
        station = station.lower().replace(' ', '_')
        predictions = predict_aqi_for_station(station)
        
        if predictions is None:
            return jsonify({'error': f'Failed to generate predictions for {station}'}), 500
        
        return jsonify({
            'station': station,
            'predictions': predictions
        })
        
    except Exception as e:
        print(f"Error in predict_aqi: {e}")
        print(traceback.format_exc())
        return jsonify({'error': str(e)}), 500

@app.route('/api/current-aqi')
def get_current_aqi():
    try:
        # Load fresh CSV
        df = pd.read_csv(CSV_FILE)
        df["AQI"] = df["AQI"].astype(str).str.replace("*", "", regex=False)
        df["AQI"] = pd.to_numeric(df["AQI"], errors="coerce")  # handle any remaining issues
        # df = df[df["AQI"].apply(lambda x: x.isnumeric())]
        df["AQI"] = df["AQI"].astype(float)

        # Build known points for IDW
        known_points = {}
        station_aqi = {}
        for _, row in df.iterrows():
            station = row["Station"]
            if station in STATION_COORDS:
                lat, lng = STATION_COORDS[station]
                aqi = row["AQI"]
                known_points[station] = (lat, lng, aqi)
                station_aqi[station] = aqi

        # Predict for missing districts
        predicted_aqi = {}
        for district, (lat, lng) in TARGET_DISTRICTS.items():
            predicted = idw_predict(lat, lng, known_points)
            if predicted is not None and math.isfinite(predicted):
                predicted_aqi[district] = round(predicted)

        return jsonify({
            "timestamp": datetime.now().isoformat(),
            "stations": station_aqi,
            "predicted": predicted_aqi
        })

    except Exception as e:
        print(f"Error in get_current_aqi: {e}")
        print(traceback.format_exc())
        return jsonify({'error': str(e)}), 500

@app.route('/api/health')
def health_check():
    """Health check endpoint"""
    try:
        # Check if CSV file exists
        csv_exists = os.path.exists(CSV_PATH)
        
        # Check if at least one model exists
        model_exists = False
        for station in STATION_COORDS.keys():
            station_name = station.lower().replace(' ', '_')
            model_path = MODEL_PATH_TEMPLATE.replace('<station_name>', station_name).format(1)
            if os.path.exists(model_path):
                model_exists = True
                break
        
        return jsonify({
            'status': 'healthy',
            'csv_file_exists': csv_exists,
            'model_files_exist': model_exists,
            'csv_path': CSV_PATH,
            'stations': list(STATION_COORDS.keys())
        })
        
    except Exception as e:
        return jsonify({
            'status': 'unhealthy',
            'error': str(e)
        }), 500

if __name__ == '__main__':
    print("Starting Ho Chi Minh City AQI Prediction Server...")
    print(f"CSV Path: {CSV_PATH}")
    print(f"Model Path Template: {MODEL_PATH_TEMPLATE}")
    print("Available stations:", list(STATION_COORDS.keys()))

    # fetch_all_aqi()
    
    # Check if CSV file exists
    if not os.path.exists(CSV_PATH):
        print(f"WARNING: CSV file not found at {CSV_PATH}")
    
    # Check for at least one model file
    model_found = False
    for station in list(STATION_COORDS.keys())[:5]:  # Check first 3 stations
        station_name = station.lower().replace(' ', '_')
        model_path = MODEL_PATH_TEMPLATE.format(hour=1, station=station_name).format(1)
        print(model_path)
        if os.path.exists(model_path):
            model_found = True
            print(f"Found model: {model_path}")
            break
    
    if not model_found:
        print("WARNING: No model files found. Predictions will use fallback method.")
    
    app.run(debug=True, host='0.0.0.0', port=5000)
    