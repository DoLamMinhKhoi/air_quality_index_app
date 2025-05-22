from playwright.sync_api import sync_playwright
from datetime import datetime
import csv
import os

# Define output file
CSV_FILE = "aqi_data_multi_station.csv"

# List of stations with names and URLs
stations = [
    ("Phu Nhuan", "https://www.iqair.com/vietnam/ho-chi-minh-city/quan-phu-nhuan"),
    ("District 3", "https://www.iqair.com/vietnam/ho-chi-minh-city/quan-ba"),
    ("District 2", "https://www.iqair.com/vietnam/ho-chi-minh-city/quan-hai"),
    ("Tan Phu", "https://www.iqair.com/vietnam/ho-chi-minh-city/quan-tan-phu"),
    ("District 10", "https://www.iqair.com/vietnam/ho-chi-minh-city/quan-muoi"),
    ("District 11", "https://www.iqair.com/vietnam/ho-chi-minh-city/quan-muoi-mot"),
    ("District 5", "https://www.iqair.com/vietnam/ho-chi-minh-city/quan-nam"),
    ("District 6", "https://www.iqair.com/vietnam/ho-chi-minh-city/quan-sau"),
    ("Binh Thanh", "https://www.iqair.com/vietnam/ho-chi-minh-city/quan-binh-thanh"),
    ("District 4", "https://www.iqair.com/vietnam/ho-chi-minh-city/quan-bon"),
    ("District 1", "https://www.iqair.com/vietnam/ho-chi-minh-city/quan-mot"),
    ("District 9", "https://www.iqair.com/vietnam/ho-chi-minh-city/quan-chin"),
    ("District 7", "https://www.iqair.com/vietnam/ho-chi-minh-city/quan-bay"),
    ("Thu Duc", "https://www.iqair.com/vietnam/ho-chi-minh-city/thu-duc"),
]

# Save header if CSV doesn't exist
def init_csv():
    if not os.path.exists(CSV_FILE):
        with open(CSV_FILE, mode='w', newline='', encoding='utf-8') as file:
            writer = csv.writer(file)
            writer.writerow(["Timestamp", "Station", "AQI"])

# Append one row to CSV
def save_to_csv(timestamp, station, aqi_value):
    with open(CSV_FILE, mode='a', newline='', encoding='utf-8') as file:
        writer = csv.writer(file)
        writer.writerow([timestamp, station, aqi_value])

# Main scraping function
def fetch_all_aqi():
    init_csv()
    with sync_playwright() as p:
        browser = p.chromium.launch(headless=True)
        page = browser.new_page()

        for station_name, url in stations:
            print(f"🌐 Fetching AQI for {station_name}...")
            try:
                page.goto(url)
                page.wait_for_timeout(5000)

                aqi_element = page.query_selector("p.aqi-value__estimated")
                if aqi_element:
                    aqi_value = aqi_element.inner_text().strip()
                    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
                    print(f"[{timestamp}] ✅ {station_name}: {aqi_value}")
                    save_to_csv(timestamp, station_name, aqi_value)
                else:
                    print(f"[{datetime.now()}] ⚠️ AQI not found for {station_name}")
                    save_to_csv(datetime.now().strftime("%Y-%m-%d %H:%M:%S"), station_name, "N/A")

            except Exception as e:
                print(f"❌ Error while processing {station_name}: {e}")
                save_to_csv(datetime.now().strftime("%Y-%m-%d %H:%M:%S"), station_name, "Error")

        browser.close()

fetch_all_aqi()
