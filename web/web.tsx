import React, { useState, useEffect } from 'react';
import { LineChart, Line, XAxis, YAxis, CartesianGrid, Tooltip, Legend, ResponsiveContainer } from 'recharts';
import { Calendar, MapPin, TrendingUp, AlertCircle, Leaf, Skull, AlertTriangle } from 'lucide-react';

// AQI color and level mappings
const getAQIColor = (aqi) => {
  if (aqi <= 50) return '#00e400';      // Good - Green
  if (aqi <= 100) return '#ffff00';     // Moderate - Yellow
  if (aqi <= 150) return '#ff7e00';     // Unhealthy for Sensitive Groups - Orange
  if (aqi <= 200) return '#ff0000';     // Unhealthy - Red
  if (aqi <= 300) return '#8f3f97';     // Very Unhealthy - Purple
  return '#7e0023';                     // Hazardous - Maroon
};

const getAQILevel = (aqi) => {
  if (aqi <= 50) return 'Tốt';
  if (aqi <= 100) return 'Trung bình';
  if (aqi <= 150) return 'Không tốt cho nhóm nhạy cảm';
  if (aqi <= 200) return 'Không tốt';
  if (aqi <= 300) return 'Rất không tốt';
  return 'Nguy hiểm';
};

const getAQIRecommendation = (aqi) => {
  if (aqi <= 50) return {
    icon: <Leaf className="w-4 h-4" />,
    text: 'Chất lượng không khí tốt. An toàn cho mọi hoạt động ngoài trời.'
  };
  if (aqi <= 100) return {
    icon: <AlertCircle className="w-4 h-4" />,
    text: 'Chất lượng không khí ở mức trung bình. Nhóm nhạy cảm nên hạn chế hoạt động ngoài trời kéo dài.'
  };
  if (aqi <= 150) return {
    icon: <AlertTriangle className="w-4 h-4" />,
    text: 'Không tốt cho nhóm nhạy cảm. Trẻ em, người già và người bệnh tim phổi nên hạn chế ra ngoài.'
  };
  if (aqi <= 200) return {
    icon: <AlertTriangle className="w-4 h-4" />,
    text: 'Không tốt cho sức khỏe. Mọi người nên hạn chế hoạt động ngoài trời.'
  };
  if (aqi <= 300) return {
    icon: <Skull className="w-4 h-4" />,
    text: 'Rất không tốt. Tránh mọi hoạt động ngoài trời. Đeo khẩu trang khi ra ngoài.'
  };
  return {
    icon: <Skull className="w-4 h-4" />,
    text: 'Nguy hiểm! Ở trong nhà, đóng cửa sổ và sử dụng máy lọc không khí.'
  };
};

// Station coordinates
const stationCoords = {
  "District 2": { lat: 10.779354, lng: 106.751480 },
  "District 3": { lat: 10.782537, lng: 106.683256 },
  "Phu Nhuan": { lat: 10.8000, lng: 106.6667 },
  "Thu Duc": { lat: 10.871060, lng: 106.826204 },
  "Binh Thanh": { lat: 10.8167, lng: 106.7000 },
  "District 1": { lat: 10.7725, lng: 106.7025 },
  "Tan Phu": { lat: 10.7817, lng: 106.6111 },
  "District 10": { lat: 10.7708, lng: 106.6642 },
  "District 11": { lat: 10.7653, lng: 106.6414 },
  "District 9": { lat: 10.854345, lng: 106.810686 },
  "District 7": { lat: 10.745781, lng: 106.714925 },
  "District 6": { lat: 10.7547, lng: 106.6500 }
};

// Sample data generator (in real app, this would fetch from your CSV files)
const generateSampleData = () => {
  const stations = Object.keys(stationCoords);
  const data = [];
  const now = new Date();
  
  for (let i = 0; i < 168; i++) { // 7 days of hourly data
    const timestamp = new Date(now.getTime() - (168 - i) * 60 * 60 * 1000);
    stations.forEach(station => {
      data.push({
        timestamp: timestamp.toISOString(),
        station,
        aqi: Math.floor(Math.random() * 200) + 20,
        state: 'Ho Chi Minh',
        country: 'Vietnam'
      });
    });
  }
  return data;
};

// Generate 24-hour predictions
const generatePredictions = (station) => {
  const predictions = [];
  const now = new Date();
  
  for (let i = 1; i <= 24; i++) {
    const futureTime = new Date(now.getTime() + i * 60 * 60 * 1000);
    predictions.push({
      hour: i,
      time: futureTime.toLocaleTimeString('vi-VN', { hour: '2-digit', minute: '2-digit' }),
      aqi: Math.floor(Math.random() * 150) + 30
    });
  }
  return predictions;
};

const AQIPredictionWebsite = () => {
  const [selectedDate, setSelectedDate] = useState(new Date().toISOString().split('T')[0]);
  const [selectedStation, setSelectedStation] = useState('Thu Duc');
  const [chartData, setChartData] = useState([]);
  const [predictions, setPredictions] = useState([]);
  const [mapData, setMapData] = useState({});

  // Simulate data loading
  useEffect(() => {
    const sampleData = generateSampleData();
    
    // Filter data for selected date and station
    const filteredData = sampleData.filter(d => {
      const dataDate = new Date(d.timestamp).toISOString().split('T')[0];
      return dataDate === selectedDate && d.station === selectedStation;
    }).map(d => ({
      ...d,
      time: new Date(d.timestamp).toLocaleTimeString('vi-VN', { hour: '2-digit', minute: '2-digit' })
    }));
    
    setChartData(filteredData);
    
    // Generate predictions for selected station
    const newPredictions = generatePredictions(selectedStation);
    setPredictions(newPredictions);
    
    // Generate latest map data
    const latestMapData = {};
    Object.keys(stationCoords).forEach(station => {
      latestMapData[station] = Math.floor(Math.random() * 200) + 20;
    });
    setMapData(latestMapData);
  }, [selectedDate, selectedStation]);

  const CustomTooltip = ({ active, payload, label }) => {
    if (active && payload && payload.length) {
      const aqi = payload[0].value;
      return (
        <div className="bg-white p-3 border border-gray-300 rounded shadow-lg">
          <p className="font-medium">{`Thời gian: ${label}`}</p>
          <p style={{ color: getAQIColor(aqi) }}>
            {`AQI: ${aqi} (${getAQILevel(aqi)})`}
          </p>
        </div>
      );
    }
    return null;
  };

  return (
    <div className="min-h-screen bg-gradient-to-br from-blue-50 to-green-50">
      {/* Header */}
      <div className="bg-white shadow-sm">
        <div className="max-w-7xl mx-auto px-4 py-6">
          <h1 className="text-4xl font-bold text-gray-900 mb-2">
            Dự đoán chất lượng không khí
          </h1>
          <p className="text-xl text-gray-600">
            Dự đoán chất lượng không khí trong 24h giờ tiếp theo ở TPHCM
          </p>
        </div>
      </div>

      <div className="max-w-7xl mx-auto px-4 py-8 space-y-8">
        {/* Section 1: Air Quality Chart */}
        <div className="bg-white rounded-lg shadow-lg p-6">
          <h2 className="text-2xl font-bold text-gray-900 mb-6 flex items-center">
            <TrendingUp className="w-6 h-6 mr-2 text-blue-600" />
            Biểu đồ chất lượng không khí
          </h2>
          
          {/* Filters */}
          <div className="flex flex-wrap gap-4 mb-6">
            <div className="flex items-center space-x-2">
              <Calendar className="w-4 h-4 text-gray-500" />
              <label className="text-sm font-medium text-gray-700">Ngày:</label>
              <input
                type="date"
                value={selectedDate}
                onChange={(e) => setSelectedDate(e.target.value)}
                className="border border-gray-300 rounded px-3 py-1 text-sm focus:outline-none focus:ring-2 focus:ring-blue-500"
              />
            </div>
            
            <div className="flex items-center space-x-2">
              <MapPin className="w-4 h-4 text-gray-500" />
              <label className="text-sm font-medium text-gray-700">Trạm:</label>
              <select
                value={selectedStation}
                onChange={(e) => setSelectedStation(e.target.value)}
                className="border border-gray-300 rounded px-3 py-1 text-sm focus:outline-none focus:ring-2 focus:ring-blue-500"
              >
                {Object.keys(stationCoords).map(station => (
                  <option key={station} value={station}>{station}</option>
                ))}
              </select>
            </div>
          </div>

          {/* Chart */}
          <div className="h-96">
            <ResponsiveContainer width="100%" height="100%">
              <LineChart data={chartData}>
                <CartesianGrid strokeDasharray="3 3" />
                <XAxis 
                  dataKey="time" 
                  tick={{ fontSize: 12 }}
                  interval="preserveStartEnd"
                />
                <YAxis 
                  label={{ value: 'AQI', angle: -90, position: 'insideLeft' }}
                  tick={{ fontSize: 12 }}
                />
                <Tooltip content={<CustomTooltip />} />
                <Line
                  type="monotone"
                  dataKey="aqi"
                  stroke="#3b82f6"
                  strokeWidth={2}
                  dot={(props) => (
                    <circle
                      cx={props.cx}
                      cy={props.cy}
                      r={4}
                      fill={getAQIColor(props.payload?.aqi || 0)}
                      stroke="#fff"
                      strokeWidth={2}
                    />
                  )}
                />
              </LineChart>
            </ResponsiveContainer>
          </div>
        </div>

        {/* Section 2: 24-hour Predictions */}
        <div className="bg-white rounded-lg shadow-lg p-6">
          <h2 className="text-2xl font-bold text-gray-900 mb-6 flex items-center">
            <TrendingUp className="w-6 h-6 mr-2 text-green-600" />
            Dự đoán AQI trong 24h - {selectedStation}
          </h2>
          
          <div className="grid grid-cols-2 md:grid-cols-4 lg:grid-cols-6 xl:grid-cols-8 gap-4">
            {predictions.map((pred, index) => {
              const recommendation = getAQIRecommendation(pred.aqi);
              return (
                <div
                  key={index}
                  className="bg-gray-50 rounded-lg p-4 text-center border-l-4 hover:shadow-md transition-shadow"
                  style={{ borderLeftColor: getAQIColor(pred.aqi) }}
                >
                  <div className="text-sm text-gray-600 mb-1">{pred.time}</div>
                  <div
                    className="text-2xl font-bold mb-1"
                    style={{ color: getAQIColor(pred.aqi) }}
                  >
                    {Math.round(pred.aqi)}
                  </div>
                  <div className="text-xs text-gray-500 mb-2">
                    {getAQILevel(pred.aqi)}
                  </div>
                  <div className="flex justify-center items-center text-xs text-gray-600">
                    {recommendation.icon}
                  </div>
                </div>
              );
            })}
          </div>
          
          {/* Overall Recommendation */}
          {predictions.length > 0 && (
            <div className="mt-6 p-4 bg-blue-50 rounded-lg">
              <h3 className="font-semibold text-gray-900 mb-2">Khuyến nghị chung:</h3>
              <div className="flex items-start space-x-2">
                {getAQIRecommendation(predictions[0]?.aqi || 50).icon}
                <p className="text-sm text-gray-700">
                  {getAQIRecommendation(predictions[0]?.aqi || 50).text}
                </p>
              </div>
            </div>
          )}
        </div>

        {/* Section 3: Ho Chi Minh City AQI Map */}
        <div className="bg-white rounded-lg shadow-lg p-6">
          <h2 className="text-2xl font-bold text-gray-900 mb-6 flex items-center">
            <MapPin className="w-6 h-6 mr-2 text-red-600" />
            Bản đồ AQI ở TPHCM
          </h2>
          
          {/* Map Container */}
          <div className="relative bg-gray-100 rounded-lg overflow-hidden" style={{ height: '500px' }}>
            {/* Simplified Ho Chi Minh City Map */}
            <div className="absolute inset-0 flex items-center justify-center">
              <div className="grid grid-cols-4 gap-2 p-8">
                {Object.entries(mapData).map(([station, aqi]) => (
                  <div
                    key={station}
                    className="relative p-4 rounded-lg border-2 border-white shadow-md cursor-pointer hover:shadow-lg transition-all transform hover:scale-105"
                    style={{ backgroundColor: getAQIColor(aqi) }}
                    title={`${station}: AQI ${Math.round(aqi)} (${getAQILevel(aqi)})`}
                  >
                    <div className="text-white text-center">
                      <div className="text-xs font-medium mb-1">{station}</div>
                      <div className="text-lg font-bold">{Math.round(aqi)}</div>
                      <div className="text-xs opacity-90">{getAQILevel(aqi)}</div>
                    </div>
                  </div>
                ))}
              </div>
            </div>
          </div>

          {/* Legend */}
          <div className="mt-4 flex flex-wrap justify-center gap-4">
            {[
              { range: '0-50', color: '#00e400', label: 'Tốt' },
              { range: '51-100', color: '#ffff00', label: 'Trung bình' },
              { range: '101-150', color: '#ff7e00', label: 'Không tốt (nhạy cảm)' },
              { range: '151-200', color: '#ff0000', label: 'Không tốt' },
              { range: '201-300', color: '#8f3f97', label: 'Rất không tốt' },
              { range: '300+', color: '#7e0023', label: 'Nguy hiểm' }
            ].map((item, index) => (
              <div key={index} className="flex items-center space-x-2">
                <div
                  className="w-4 h-4 rounded"
                  style={{ backgroundColor: item.color }}
                ></div>
                <span className="text-xs text-gray-600">
                  {item.range}: {item.label}
                </span>
              </div>
            ))}
          </div>
        </div>
      </div>

      {/* Footer */}
      <div className="bg-gray-800 text-white py-8 mt-12">
        <div className="max-w-7xl mx-auto px-4 text-center">
          <p className="text-gray-300">
            Dự đoán chất lượng không khí TP.HCM - Cập nhật theo thời gian thực
          </p>
          <p className="text-gray-400 text-sm mt-2">
            Dữ liệu được thu thập từ các trạm quan trắc chất lượng không khí trên toàn thành phố
          </p>
        </div>
      </div>
    </div>
  );
};

export default AQIPredictionWebsite;