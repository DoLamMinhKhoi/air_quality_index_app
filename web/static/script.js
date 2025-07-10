// Coordinates for all districts
const districtCoordinates = {
    "District 2": [10.779354, 106.751480],
    "District 3": [10.782537, 106.683256],
    "Phu Nhuan": [10.8000, 106.6667],
    "Thu Duc": [10.871060, 106.826204],
    "Binh Thanh": [10.8167, 106.7000],
    "District 1": [10.7725, 106.7025],
    "Tan Phu": [10.7817, 106.6111],
    "District 10": [10.7708, 106.6642],
    "District 4": [10.7550, 106.7050],
    "District 5": [10.7574, 106.6739],
    "District 12": [10.8500, 106.6500]
};

// Initialize map
const map = L.map('map').setView([10.7725, 106.7025], 11);

// Add tile layer
L.tileLayer('https://{s}.tile.openstreetmap.org/{z}/{x}/{y}.png', {
    attribution: '&copy; OpenStreetMap contributors'
}).addTo(map);

// Get color based on AQI value
function getColor(aqi) {
    if (aqi < 50) return "green";
    else if (aqi < 100) return "yellow";
    else if (aqi < 150) return "orange";
    else return "red";
}

// Load AQI data from backend
fetch("http://127.0.0.1:5000/aqi")
    .then(response => response.json())
    .then(data => {
        for (const [district, aqi] of Object.entries(data)) {
            const coords = districtCoordinates[district];
            if (!coords) continue;

            const color = getColor(aqi);

            L.circleMarker(coords, {
                color: color,
                fillColor: color,
                fillOpacity: 0.7,
                radius: 10
            }).addTo(map).bindPopup(`<strong>${district}</strong><br>AQI: ${aqi.toFixed(2)}`);
        }
    })
    .catch(error => {
        console.error("Error loading AQI data:", error);
        alert("Failed to load AQI data.");
    });