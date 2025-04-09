import pandas as pd
import wikipedia as wp
import re
from bs4 import BeautifulSoup
import requests
import time
from pathlib import Path

def get_museum_data(regenerate=False):
    output_file = Path("data/museum_data.csv")
    if output_file.exists() and not regenerate:
        return pd.read_csv(output_file)
    else:
        df = generate_museum_dataset()
        df.to_csv(output_file)
        return df

def generate_museum_dataset():
    soup = BeautifulSoup(wp.page("List of most-visited museums").html(), 'html.parser')
    
    # Find the correct table
    tables = soup.find_all('table', {'class': 'wikitable'})
    target_table = tables[0]
        
    # Extract Wikipedia links
    museum_data = []
    for row in target_table.find_all('tr')[1:]:  # Skip header
        cells = row.find_all('td')
        if len(cells) > 1:
            link = cells[0].find('a')
            if link and link.get('href'):
                url = f"https://en.wikipedia.org{link['href']}"
                museum_data.append(url)
    
    # Get DataFrame and add links
    df = pd.read_html(str(target_table))[0]
    df['wikipedia_link'] = museum_data[:len(df)]  # Match lengths
    
    # Add characteristics
    df['type'] = ''
    df['collection_size'] = ''
    
    for i, url in enumerate(museum_data):
        if i >= len(df):
            break
        chars = get_museum_characteristics(url)
        df.at[i, 'type'] = chars['type']
        df.at[i, 'collection_size'] = chars['collection_size']
        #time.sleep(1)
    
    return clean_museum_table(df)

def get_museum_characteristics(url):
    """Scrape and standardize museum characteristics"""

    response = requests.get(url)
    response.encoding = 'utf-8'  # Force UTF-8 encoding
    soup = BeautifulSoup(response.text, 'html.parser')
    infobox = soup.find('table', {'class': 'infobox'})
    
    raw_type = 'N/A'
    raw_size = 'N/A'

    if infobox:
        for row in infobox.find_all('tr'):
            headers = row.find_all(['th', 'td'])
            if len(headers) >= 2:
                key = headers[0].get_text().lower()
                value = headers[1].get_text()
                if 'type' in key or 'genre' in key:
                    raw_type = value
                elif 'collection size' in key or 'holdings' in key:
                    raw_size = clean_collection_size(value)
    
    return {
        'type': raw_type,
        'collection_size': raw_size
    }

def clean_collection_size(raw_size):
    """Clean and standardize collection size data"""
    # Remove citations and special characters
    clean = re.sub(r'\[\d+\]|[≈~]', '', raw_size).strip()
    
    # Extract numerical value and units
    match = re.match(r"([\d,\.]+)\s*([a-zA-Z]+)", clean)
    if not match:
        return 'N/A'
    
    quantity, unit = match.groups()
    quantity = float(quantity.replace(',', ''))

    if unit == 'million':
        return quantity*1000000
    else:
        return quantity
    
def convert_million_values(df):
    extracted = df['visitors'].str.extract(r'([\d,]+(?:\.\d+)?)\s*(million)', flags=re.IGNORECASE)
    mask = extracted[1].notna()
    numbers = extracted[0].str.replace(',', '').astype(float)
    df.loc[mask, 'visitors'] = (numbers[mask] * 1_000_000).apply(lambda x: f"{int(x):,}")

def extract_first_city_part(df, column_name):
    df[column_name] = df[column_name].str.split(r',\s*', n=1).str[0]
    return df

def clean_museum_table(df):
    # Clean up data
    df = df.dropna(how='all')
    df = df.reset_index(drop=True)
    
    # Rename columns
    df.rename(columns={
        "Name": "name",
        "Visitors in 2023 or 2024": "visitors",
        "City": "city",
        "Country": "country"
    }, inplace=True)
    
    # Clean citations and years
    df = df.applymap(lambda x: x.split('[')[0] if isinstance(x, str) else x)
    df['visitors'] = df['visitors'].str.replace(r'\s*\(.*?\)', '', regex=True).str.strip()
    
    # Convert values
    convert_million_values(df)
    
    #Extract first part of city name
    extract_first_city_part(df, 'city')
    
    # Reorder columns
    df = df[['name', 'type', 'collection_size', 'visitors', 'city', 'country']]

    return df

def run():
    df = get_museum_data()
    print(df.to_string())

run()