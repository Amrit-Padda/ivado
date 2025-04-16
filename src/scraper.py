import pandas as pd
import wikipedia as wp
import re
from bs4 import BeautifulSoup
import requests
import time
from pathlib import Path
from io import StringIO
import numpy as np
from typing import Generator

class Scraper:
    TYPE = 'type'
    COLLECTION_SIZE = 'collection_size'

    # Reverse mapping of wikipedia infobox titles to our feature list
    # Key = the wikipedia name
    # Value = the feature we want to aggregate on
    MODEL_FEATURE_REVERSE_MAPPING = {
        'type': TYPE,
        'genre': TYPE,
        'collection size': COLLECTION_SIZE, 
        'holdings': COLLECTION_SIZE}

    # Defining default values for the above features
    # Key = the feature name
    # Value = Default value
    MODEL_FEATURES = {
        TYPE: "N/A",
        COLLECTION_SIZE: np.nan
    }

    def get_museum_data(self, regenerate: bool = False) -> pd.DataFrame: 
        """
        A high level function to return the museum dataset,
        it will only regenerate the dataset if it is not cached locally
        or if the regenerate flag is set to true.
        
        TODO: As the size of the dataset increases, and to allow for many users to access the data
        we would have to save it elsewhere. 
        For now since this is a MVP we will save the file locally. 

        Args:
            regenerate (bool, optional): Whether we want to regenerate the dataset, Defaults to False.

        Returns:
            pd.DataFrame: The full museum dataset
        """
        output_file: str = Path('../data/museum_data.csv')
        
        if output_file.exists() and not regenerate:
            return pd.read_csv(output_file)
        
        df: pd.DataFrame = self.generate_museum_dataset()
        df.to_csv(output_file)
        return df

    def generate_museum_dataset(self) -> pd.DataFrame:
        """
        Generate the museum dataset by scraping and cleaning the wikipedia page
        
        Returns:
            pd.DataFrame: cleaned museum dataframe
        """
        soup = BeautifulSoup(wp.page("List of most-visited museums").html(), 'html.parser')
        
        # Find the correct table
        tables = soup.find_all('table', {'class': 'wikitable'})
        target_table = tables[0]
        df = pd.read_html(StringIO(str(target_table)))[0]
        df = self.add_features(df, target_table)
        return self.clean_museum_table(df)
            
    def add_features(self, df: pd.DataFrame, target_table: str) -> pd.DataFrame:
        """
        Return a dataframe containing all the features we were able to
        scrape from the wiki page

        Args:
            df (pd.DataFrame): The df to add features to
            target_table (str): The html table containing the data we need to scrape

        Returns:
            pd.DataFrame: The museum dataframe with additional features
        """

        for i, url in self.museum_wiki_link_generator(target_table):
            if i >= len(df):
                break
            features: dict = self.get_museum_features(url)

            for f in self.MODEL_FEATURES:
                df.at[i, f] = features.get(f, self.MODEL_FEATURES[f])
        return df

    def museum_wiki_link_generator(self, table: str) -> Generator:
        """
        A generator function to iterate through the table of museums
        and return the wiki link to the museum, this will allow 
        us to scrape additional characteristics later on.

        Args:
            table (str): The HTML table we will parse

        Returns:
            Generator: A generator which yields a tuple containing the url to the museum
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

    def get_museum_features(self, url:str) -> dict:
        """
        Scrape and standardize museum characteristics

        Args:
            url (str): The URL to load

        Returns:
            dict: Dict containing the features we need
        """

        response = requests.get(url)
        response.encoding = 'utf-8'  # Force UTF-8 encoding
        soup = BeautifulSoup(response.text, 'html.parser')
        
        features = {}
        if infobox := soup.find('table', {'class': 'infobox'}):
            for key, value in self.handle_infobox(infobox):
                if key in self.MODEL_FEATURE_REVERSE_MAPPING:
                    if self.MODEL_FEATURE_REVERSE_MAPPING[key] == self.TYPE:
                        features[self.TYPE] = value
                    elif self.MODEL_FEATURE_REVERSE_MAPPING[key] == self.COLLECTION_SIZE:
                        features[self.COLLECTION_SIZE] = self.clean_collection_size(value)
                    
        return features

    def handle_infobox(self, infobox: str) -> Generator:
        """
        Given a wikipedia infobox, yield the key value pairs.

        Args:
            infobox (str): The HTML element to parse

        Yields:
            Generator: A generator which yields all title, value pairs in the infobox
        """
        for row in infobox.find_all('tr'):
            headers = row.find_all(['th', 'td'])
            if len(headers) >= 2:
                key = headers[0].get_text().lower()
                value = headers[1].get_text()
                yield key, value
        
    def clean_collection_size(self, raw_size: str) -> float:
        """
        Clean and standardize collection size data

        Args:
            raw_size (str): The collection size as a string

        Returns:
            float: The collection size as a float
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
        
    def convert_million_values(self, df: pd.DataFrame) -> None:
        """    
        Some visitor values are in the form 4.3 million,
        here we convert it to a int 

        Args:
            df (pd.DataFrame): The museum dataset with inconsistent visitor data 
        """
        extracted = df['visitors'].str.extract(r'([\d,]+(?:\.\d+)?)\s*(million)', flags=re.IGNORECASE)
        mask = extracted[1].notna()
        numbers = extracted[0].str.replace(',', '').astype(float)
        df.loc[mask, 'visitors'] = (numbers[mask] * 1_000_000).apply(lambda x: f"{int(x):,}")

    def extract_first_city_part(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Convert city names with multiple parts, into one part

        Args:
            df (pd.DataFrame): The museum dataset with malformed city names

        Returns:
            pd.DataFrame: The museum dataset with cleaned city names
        """
        df['city'] = df['city'].str.split(r',\s*', n=1).str[0]
        return df

    def clean_museum_table(self, df: pd.DataFrame) -> pd.DataFrame:
        """    
        Clean and standardize the museum data table.

        This function performs various cleaning and transformation steps on the
        museum dataframe to prepare it for analysis.

        Args:
            df (pd.DataFrame): The Museum dataset

        Returns:
            pd.DataFrame: The Museum dataset after cleaning
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
        self.convert_million_values(df)

        # Convert numerical values
        df['visitors'] = df['visitors'].str.replace(',', '').astype('int64')
        df['collection_size'] = df['collection_size'].astype(float)
        
        #Extract first part of city name
        self.extract_first_city_part(df)
        
        # Reorder columns
        df = df[['name', 'type', 'collection_size', 'visitors', 'city', 'country']]

        return df