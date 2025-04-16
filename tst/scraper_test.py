import pandas as pd
import numpy as np
import pytest
from src.scraper import Scraper
import wikipedia as wp
from bs4 import BeautifulSoup
from io import StringIO
import requests
from pathlib import Path

# Mocking wikipedia.page to avoid actual network calls
class MockWikipediaPage:
    def __init__(self, content):
        self.content = content

    def html(self):
        return self.content

# Mocking requests.get to avoid actual network calls
class MockResponse:
    def __init__(self, content):
        self.content = content
        self.encoding = 'utf-8'

    def text(self):
        return self.content

@pytest.fixture
def scraper():
    return Scraper()

@pytest.mark.parametrize(
    "test_id, raw_size, expected",
    [
        ("tc004_number_with_million", "1 million", 1000000.0),
        ("tc009_number_with_million_and_approx", "≈1 million", 1000000.0),
        ("tc010_number_with_million_and_citation", "1 million[1]", 1000000.0),
    ],
)
def test_clean_collection_size(scraper, test_id: str, raw_size: str, expected: float):
    """
    Test cases for clean_collection_size method.
    """

    # Act
    result = scraper.clean_collection_size(raw_size)

    # Assert
    assert result == expected

@pytest.mark.parametrize(
    "test_id, input_df, expected_df",
    [
        (
            "tc015_multiple_city_parts",
            pd.DataFrame({'city': ['New York, USA', 'London, UK']}),
            pd.DataFrame({'city': ['New York', 'London']}),
        ),
        (
            "tc016_single_city_part",
            pd.DataFrame({'city': ['Paris']}),
            pd.DataFrame({'city': ['Paris']}),
        ),
        (
            "tc017_empty_city",
            pd.DataFrame({'city': ['']}),
            pd.DataFrame({'city': ['']}),
        ),
    ],
)
def test_extract_first_city_part(scraper, test_id, input_df, expected_df):
    """
    Test cases for extract_first_city_part method
    """

    # Act
    result_df = scraper.extract_first_city_part(input_df.copy())

    # Assert
    pd.testing.assert_frame_equal(result_df, expected_df)


@pytest.mark.parametrize(
    "test_id, infobox_html, expected_pairs",
    [
        (
            "tc029_valid_infobox",
            """
            <table class="infobox">
                <tr><th>Key1</th><td>Value1</td></tr>
                <tr><th>Key2</th><td>Value2</td></tr>
            </table>
            """,
            [('key1', 'Value1'), ('key2', 'Value2')],
        ),
        (
            "tc030_empty_infobox",
            """<table class="infobox"></table>""",
            [],
        ),
        (
            "tc031_infobox_with_missing_value",
            """
            <table class="infobox">
                <tr><th>Key1</th></tr>
            </table>
            """,
            [],
        ),
    ],
)
def test_handle_infobox(scraper, test_id, infobox_html, expected_pairs):
    """
    Test cases for handle_infobox method.
    """

    # Arrange
    infobox = BeautifulSoup(infobox_html, 'html.parser')

    # Act
    pairs = list(scraper.handle_infobox(infobox))

    # Assert
    assert pairs == expected_pairs
