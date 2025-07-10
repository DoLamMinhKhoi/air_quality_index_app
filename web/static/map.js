const districtCoords = {
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

const aqiColors = [
    { max: 50, color: "green" },
    { max: 100, color: "yellow" },
    { max: 150, color: "orange" },
    { max: 200, color: "red" },
    { max: 300, color: "purple" },
    { max: Infinity, color: "maroon" }
];

function getColor(aqi) {
    for (const level of aqiColors) {
        if (aqi <= level.max) return level.color;
    }
    return "gray";
}

function initMap() {
    const map = L.map("map").setView([10.7725, 106.7025], 11);

    L.tileLayer("https://{s}.tile.openstreetmap.org/{z}/{x}/{y}.png", {
        attribution: "© OpenStreetMap contributors"
    }).addTo(map);

    fetch("/aqi")
        .then(res => res.json())
        .then(data => {
            for (const [district, aqi] of Object.entries(data)) {
                const color = getColor(aqi);
                const coords = districtCoords[district];
                if (coords) {
                    L.circleMarker(coords, {
                        color: color,
                        fillColor: color,
                        fillOpacity: 0.7,
                        radius: 12
                    }).addTo(map)
                      .bindPopup(`<strong>${district}</strong><br/>AQI: ${aqi.toFixed(1)}<br/>Level: ${color.toUpperCase()}`);
                }
            }
        })
        .catch(err => console.error("Error loading AQI data for map:", err));
}

// Call this function in main.js or after DOM loads
initMap();
