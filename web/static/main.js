document.addEventListener("DOMContentLoaded", () => {
    // Elements
    const stationSelect = document.getElementById("stationSelect");
    const startDate = document.getElementById("startDate");
    const endDate = document.getElementById("endDate");
    const chartCanvas = document.getElementById("aqiChart").getContext("2d");
    const predictionBox = document.getElementById("aqiPrediction");
    const recommendationBox = document.getElementById("recommendation");

    let chart;

    // Fetch stations
    fetch("/stations")
        .then(res => res.json())
        .then(stations => {
            stations.forEach(station => {
                const opt = document.createElement("option");
                opt.value = station;
                opt.textContent = station;
                stationSelect.appendChild(opt);
            });
        });

    // Draw chart
    function drawChart(labels, data) {
        if (chart) chart.destroy();
        chart = new Chart(chartCanvas, {
            type: "line",
            data: {
                labels: labels,
                datasets: [{
                    label: "AQI",
                    data: data,
                    borderColor: "#2c3e50",
                    borderWidth: 2,
                    pointBackgroundColor: data.map(getAqiColor),
                    fill: false
                }]
            },
            options: {
                responsive: true,
                scales: {
                    y: {
                        beginAtZero: true
                    }
                }
            }
        });
    }

    // AQI color helper
    function getAqiColor(aqi) {
        if (aqi < 50) return "#009966";
        if (aqi < 100) return "#ffde33";
        if (aqi < 150) return "#ff9933";
        return "#cc0033";
    }

    // AQI level name
    function getAqiLevel(aqi) {
        if (aqi < 50) return "Tốt";
        if (aqi < 100) return "Trung bình";
        if (aqi < 150) return "Kém";
        return "Xấu";
    }

    function getAqiBoxClass(aqi) {
        if (aqi < 50) return "aqi-good-box";
        if (aqi < 100) return "aqi-moderate-box";
        if (aqi < 150) return "aqi-unhealthy-sensitive-box";
        return "aqi-unhealthy-box";
    }

    function getRecommendation(aqi) {
        if (aqi < 50) return "Không khí tốt. Bạn có thể ra ngoài bình thường.";
        if (aqi < 100) return "Không khí trung bình. Hạn chế các hoạt động ngoài trời kéo dài.";
        if (aqi < 150) return "Không khí kém. Nhóm nhạy cảm nên hạn chế ra ngoài.";
        return "Không khí xấu. Tất cả mọi người nên tránh các hoạt động ngoài trời.";
    }

    // Fetch and update chart
    function updateChart() {
        const station = stationSelect.value;
        const start = startDate.value;
        const end = endDate.value;
        if (!station || !start || !end) return;

        fetch(`/aqi-history?station=${station}&start=${start}&end=${end}`)
            .then(res => res.json())
            .then(result => {
                drawChart(result.timestamps, result.aqi_values);
            });
    }

    // Fetch and update 24h prediction
    function updatePrediction() {
        const station = stationSelect.value;
        if (!station) return;

        fetch(`/predict?station=${station}`)
            .then(res => res.json())
            .then(result => {
                predictionBox.innerHTML = "";
                result.forEach(aqi => {
                    const span = document.createElement("span");
                    span.textContent = aqi.toFixed(0);
                    span.classList.add("aqi-box", getAqiBoxClass(aqi));
                    predictionBox.appendChild(span);
                });

                const lastAqi = result[result.length - 1];
                recommendationBox.textContent = getRecommendation(lastAqi);
            });
    }

    // Event listeners
    stationSelect.addEventListener("change", () => {
        updateChart();
        updatePrediction();
    });

    startDate.addEventListener("change", updateChart);
    endDate.addEventListener("change", updateChart);

    // Map (loaded at bottom)
    const map = L.map("map").setView([10.7725, 106.7025], 11);
    L.tileLayer("https://{s}.tile.openstreetmap.org/{z}/{x}/{y}.png").addTo(map);

    fetch("/choropleth-data")
        .then(res => res.json())
        .then(data => {
            data.features.forEach(feature => {
                const aqi = feature.properties.aqi;
                const color = getAqiColor(aqi);
                L.geoJSON(feature, {
                    style: {
                        color: "#555",
                        fillColor: color,
                        weight: 1,
                        fillOpacity: 0.6
                    }
                }).addTo(map).bindPopup(`${feature.properties.name}: AQI ${aqi.toFixed(2)} (${getAqiLevel(aqi)})`);
            });
        });
});
