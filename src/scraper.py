import pandas as pd
import wikipedia as wp
import re
from bs4 import BeautifulSoup
import requests
import time
from pathlib import Path
from io import StringIO
import numpy as np

def get_museum_data(regenerate=False) -> pd.DataFrame: 
    """
    A high level function to return the museum dataset,
    it will only regenerate the dataset if it is not cached locally
    or if the regenerate flag is set to true.
    
    TODO: As the size of the dataset increases, and to allow for many users to access the data
    we would have to save it elsewhere. 
    For now since this is a MVP we will save the file locally. 
    
    :param regenerate: Whether we want to regenerate the dataset
    :return: the dataframe containing museum data
    """
    output_file = Path('../data/museum_data.csv')
    
    if output_file.exists() and not regenerate:
        return pd.read_csv(output_file)
    
    df = generate_museum_dataset()
    df.to_csv(output_file)
    return df

def generate_museum_dataset() -> pd.DataFrame:
    """
        Generate the museum dataset by scraping and cleaning the wikipedia page
        :return: cleaned museum dataframe
    """
    soup = BeautifulSoup(wp.page("List of most-visited museums").html(), 'html.parser')
    
    # Find the correct table
    tables = soup.find_all('table', {'class': 'wikitable'})
    target_table = tables[0]
    df = pd.read_html(StringIO(str(target_table)))[0]
        
    # Add characteristics
    df['type'] = ''
    df['collection_size'] = ''
    
    for i, url in museum_wiki_link_generator(target_table):
        if i >= len(df):
            break
        chars = get_museum_characteristics(url)
        df.at[i, 'type'] = chars['type']
        df.at[i, 'collection_size'] = chars['collection_size']
        
    return clean_museum_table(df)

def museum_wiki_link_generator(table) -> any:
    """
        A generator function to iterate through the table of museums
        and return the wiki link to the museum, this will allow 
        us to scrape additional characteristics later on
        
        :param table: the museum table
        :return: the wiki link to the museum
    """
    i = 0
    for row in table.find_all('tr')[1:]:  # Skip header
        cells = row.find_all('td')
        if len(cells) > 1:
            link = cells[0].find('a')
            if link and link.get('href'):
                url = f"https://en.wikipedia.org{link['href']}"
                yield(i, url)
                i += 1

def get_museum_characteristics(url) -> dict:
    """
    Scrape and standardize museum characteristics
    
    :param url: the url we want to scrape for museum data
    :return: the characteristics we want to add to the df
    """

    response = requests.get(url)
    response.encoding = 'utf-8'  # Force UTF-8 encoding
    soup = BeautifulSoup(response.text, 'html.parser')
    
    if infobox := soup.find('table', {'class': 'infobox'}):
        raw_type, raw_size = handle_infobox(infobox)
        return {
            'type': raw_type,
            'collection_size': raw_size
        }
    else:
        return {
            'type': 'N/A',
            'collection_size': np.nan
        }

def handle_infobox(infobox) -> tuple[str, int]:
    raw_type = 'N/A'
    raw_size = np.nan
    for row in infobox.find_all('tr'):
        headers = row.find_all(['th', 'td'])
        if len(headers) >= 2:
            key = headers[0].get_text().lower()
            value = headers[1].get_text()
            if 'type' in key or 'genre' in key:
                raw_type = value
            elif 'collection size' in key or 'holdings' in key:
                raw_size = clean_collection_size(value)
    return raw_type, raw_size
    
def clean_collection_size(raw_size) -> float:
    """
    Clean and standardize collection size data
    
    :param raw_size: The raw collection size value
    :return: the size of the collection as an integer
    """
    # Remove citations and special characters
    clean = re.sub(r'\[\d+\]|[≈~]', '', raw_size).strip()

    # Extract numerical value and units
    match = re.match(r"([\d,\.]+)\s*([a-zA-Z]+)", clean)
    if not match:
        return np.nan

    quantity, unit = match.groups()
    quantity = float(quantity.replace(',', ''))

    return quantity*1000000 if unit.lower() == 'million' else quantity
    
def convert_million_values(df) -> None:
    """
    Some visitor values are in the form 4.3 million,
    here we convert it to a int 

    :param df: dataframe containing incorrect numbers
    """
    extracted = df['visitors'].str.extract(r'([\d,]+(?:\.\d+)?)\s*(million)', flags=re.IGNORECASE)
    mask = extracted[1].notna()
    numbers = extracted[0].str.replace(',', '').astype(float)
    df.loc[mask, 'visitors'] = (numbers[mask] * 1_000_000).apply(lambda x: f"{int(x):,}")

def extract_first_city_part(df) -> pd.DataFrame:
    """
    Convert city names with multiple parts, into one part

    :param df: museum dataset

    :return: museum dataset
    """
    df['city'] = df['city'].str.split(r',\s*', n=1).str[0]
    return df

def clean_museum_table(df) -> pd.DataFrame:
    """
    Clean and standardize the museum data table.

    This function performs various cleaning and transformation steps on the
    museum dataframe to prepare it for analysis.

    :param df: The raw museum dataframe.
    :return: The cleaned museum dataframe.
    """
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
    df = df.map(lambda x: x.split('[')[0] if isinstance(x, str) else x)
    df['visitors'] = df['visitors'].str.replace(r'\s*\(.*?\)', '', regex=True).str.strip()
    
    # Clean the one visitor value that contains a leading >
    df['visitors'] = df['visitors'].str.replace(r'^>', '', regex=True).str.strip()
    
    # Clean the name of the M+ museum to M_plus
    df['name'] = df['name'].str.replace(r'\+', '_plus', regex=True).str.strip()

    # Convert values
    convert_million_values(df)

    # Convert numerical values
    df['visitors'] = df['visitors'].str.replace(',', '').astype('int64')
    df['collection_size'] = df['collection_size'].astype(float)
    
    #Extract first part of city name
    extract_first_city_part(df)
    
    # Reorder columns
    df = df[['name', 'type', 'collection_size', 'visitors', 'city', 'country']]

    return df