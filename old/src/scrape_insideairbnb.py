import pandas as pd
from bs4 import BeautifulSoup
from selenium import webdriver
from selenium.webdriver.common.by import By
from selenium.webdriver.chrome.service import Service
from webdriver_manager.chrome import ChromeDriverManager
from selenium.webdriver.chrome.options import Options
import time
from datetime import datetime

def parse_location(location_string):
    """Parses the location string into City, Region, and Country."""
    parts = [p.strip() for p in location_string.split(',')]
    if len(parts) == 3:
        return parts[0], parts[1], parts[2]
    elif len(parts) == 2:
        return parts[0], "", parts[1]
    elif len(parts) == 1:
        return parts[0], "", ""
    else:
        return location_string, "", ""

def parse_date(date_string):
    """Parses the date string and returns it in YYYY-MM-DD format."""
    try:
        # Format is "DD Month, YYYY"
        date_obj = datetime.strptime(date_string, '%d %B, %Y')
        return date_obj.strftime('%Y-%m-%d')
    except ValueError:
        # Handle cases where the date might be in a different format or missing
        return date_string

def scrape_airbnb_data():
    """
    Scrapes Inside Airbnb's data page to get all current and archived data links.
    """
    url = "https://insideairbnb.com/get-the-data/"

    # --- Selenium Setup for Headless Chrome ---
    print("Setting up headless browser...")
    chrome_options = Options()
    chrome_options.add_argument("--headless")
    chrome_options.add_argument("--window-size=1920,1080")
    
    # Use webdriver-manager to automatically handle the driver
    service = Service(ChromeDriverManager().install())
    driver = webdriver.Chrome(service=service, options=chrome_options)

    # --- Load Page and Reveal Archived Data ---
    print(f"Navigating to {url}...")
    driver.get(url)

    # Allow some time for the initial page to load
    time.sleep(3)

    print("Finding and clicking all 'show archived data' links...")
    try:
        show_archive_buttons = driver.find_elements(By.CLASS_NAME, 'showArchivedData')
        for button in show_archive_buttons:
            # Scroll the button into view and click it
            driver.execute_script("arguments[0].scrollIntoView(true);", button)
            time.sleep(0.1) # small delay before click
            driver.execute_script("arguments[0].click();", button)
        print(f"Clicked {len(show_archive_buttons)} 'show' links.")
    except Exception as e:
        print(f"Could not find or click 'show' buttons: {e}")

    # Wait for all the dynamic content to load
    print("Waiting for archived data to load...")
    time.sleep(5) 

    # --- Parse the Fully-Rendered HTML ---
    print("Parsing the complete page HTML...")
    html_content = driver.page_source
    soup = BeautifulSoup(html_content, 'html.parser')
    
    # Close the browser
    driver.quit()

    # --- Extract Data into a List ---
    all_data = []
    
    # Find all data tables. Each table corresponds to one date for one city.
    data_tables = soup.find_all('table', class_='data')
    print(f"Found {len(data_tables)} data tables to process.")

    for table in data_tables:
        try:
            # Find the date and location headers preceding each table
            date_header = table.find_previous_sibling('h4')
            location_header = table.find_previous_sibling('h3')

            if not date_header or not location_header:
                continue

            # Extract and clean the text
            location_str = location_header.get_text(strip=True)
            date_str = date_header.get_text(strip=True).split('(')[0].strip()
            
            city, region, country = parse_location(location_str)
            formatted_date = parse_date(date_str)

            # Find the required links within the table
            links = table.find_all('a')
            listings_url = None
            geojson_url = None

            for link in links:
                href = link.get('href', '')
                if 'listings.csv.gz' in href:
                    listings_url = href
                elif 'neighbourhoods.geojson' in href:
                    geojson_url = href
            
            # Append the found data to our list
            if listings_url or geojson_url:
                all_data.append({
                    'City': city,
                    'Region': region,
                    'Country': country,
                    'Date': formatted_date,
                    'listings_url': listings_url,
                    'geojson_url': geojson_url,
                })
        except Exception as e:
            print(f"Error processing a table: {e}")

    # --- Create DataFrame and Save to CSV ---
    if not all_data:
        print("No data was extracted. The script may need updating.")
        return

    print("Creating DataFrame and saving to CSV...")
    df = pd.DataFrame(all_data)
    
    # Ensure columns are in the desired order
    df = df[['City', 'Region', 'Country', 'Date', 'listings_url', 'geojson_url']]
    
    output_filename = 'insideairbnb_all_data.csv'
    df.to_csv(output_filename, index=False)
    print(f"Successfully created '{output_filename}' with {len(df)} rows.")

if __name__ == "__main__":
    scrape_airbnb_data()