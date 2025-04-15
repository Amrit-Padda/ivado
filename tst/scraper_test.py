import pandas as pd
import pytest
import numpy as np
from bs4 import BeautifulSoup
from src.scraper import clean_collection_size, convert_million_values, extract_first_city_part, museum_wiki_link_generator, handle_infobox

@pytest.mark.parametrize(
    "raw_size, expected_size",
    [
        ("1.2 million", 1200000.0),
        ("1,234,567 million", 1234567000000.0),
        ("≈1.2 million", 1200000.0),
        ("1.2 million[1]", 1200000.0),
        ("  1.2 million ", 1200000.0),
        ("1.2 MILLION", 1200000.0),
        ("", np.nan),
        ("[1]", np.nan),
        ("≈~", np.nan),
        ("0 million", 0.0),
    ],
)
def test_clean_collection_size(raw_size, expected_size):
    actual_size = clean_collection_size(raw_size)
    if np.isnan(expected_size):
        assert np.isnan(actual_size)
    else:
        assert actual_size == expected_size

@pytest.mark.parametrize(
    "test_id, input_df, expected_df",
    [
        (
            "happy_path_single_million",
            pd.DataFrame({"visitors": ["4.3 million"]}),
            pd.DataFrame({"visitors": ["4,300,000"]}),
        ),
        (
            "happy_path_multiple_millions",
            pd.DataFrame({"visitors": ["4.3 million", "1 million", "0.5 million"]}),
            pd.DataFrame({"visitors": ["4,300,000", "1,000,000", "500,000"]}),
        ),
        (
            "happy_path_with_commas",
            pd.DataFrame({"visitors": ["1,234.5 million"]}),
            pd.DataFrame({"visitors": ["1,234,500,000"]}),
        ),
        (
            "happy_path_mixed_values",
            pd.DataFrame({"visitors": ["4.3 million", "12345", "0.5 million"]}),
            pd.DataFrame({"visitors": ["4,300,000", "12345", "500,000"]}),
        ),
        (
            "edge_case_zero_million",
            pd.DataFrame({"visitors": ["0 million"]}),
            pd.DataFrame({"visitors": ["0"]}),
        ),
        (
            "edge_case_million_with_leading_space",
            pd.DataFrame({"visitors": [" 4.3 million"]}),
            pd.DataFrame({"visitors": ["4,300,000"]}),
        ),
        (
            "edge_case_MILLION_case",
            pd.DataFrame({"visitors": ["4.3 MILLION"]}),
            pd.DataFrame({"visitors": ["4,300,000"]}),

        ),
    ],
)
def test_convert_million_values_happy_path(test_id, input_df, expected_df):
    actual_df = input_df.copy()
    convert_million_values(actual_df)
    pd.testing.assert_frame_equal(actual_df, expected_df)


@pytest.mark.parametrize(
    "test_id, input_df, expected_df",
    [
        (
            "happy_path_single_comma",
            pd.DataFrame({"city": ["Montreal, QC"]}),
            pd.DataFrame({"city": ["Montreal"]}),
        ),
        (
            "happy_path_multiple_commas",
            pd.DataFrame({"city": ["Montreal, QC, Canada"]}),
            pd.DataFrame({"city": ["Montreal"]}),
        ),
        (
            "happy_path_no_comma",
            pd.DataFrame({"city": ["Montreal"]}),
            pd.DataFrame({"city": ["Montreal"]}),
        ),
        (
            "happy_path_empty_string",
            pd.DataFrame({"city": [""]}),
            pd.DataFrame({"city": [""]}),
        )
    ],
)
def test_extract_first_city_part_happy_path(test_id, input_df, expected_df):
    actual_df = extract_first_city_part(input_df.copy())
    pd.testing.assert_frame_equal(actual_df, expected_df)
    

@pytest.mark.parametrize(
    "test_id, input_html, expected_output",
    [
        (
            "edge_case_empty_table",
            """<table><tr><th>Museum</th></tr></table>""",
            [],
        ),
        (
            "edge_case_no_links",
            """<table><tr><th>Museum</th></tr><tr><td>Museum of Natural History</td></tr></table>""",
            [],
        ),
        (
            "edge_case_link_without_href",
            """<table><tr><th>Museum</th></tr><tr><td><a>Museum of Natural History</a></td></tr></table>""",
            [],
        ),
        (
            "edge_case_empty_link",
            """<table><tr><th>Museum</th></tr><tr><td><a href=""></a></td></tr></table>""",
            [],
        ),
        (
            "error_case_invalid_html",  # Invalid HTML (missing closing tag for 'a')
            """<table><tr><th>Museum</th></tr><tr><td><a href="/wiki/Museum_of_Natural_History,_London">Museum of Natural History</td></tr></table>""",
            [],
        ),
        (
            "edge_case_missing_cells",
            """<table><tr><th>Museum</th></tr><tr></tr></table>""",  # Missing <td>
            [],
        ),
    ],
)
def test_museum_wiki_link_generator(test_id, input_html, expected_output):

    # Act
    soup = BeautifulSoup(input_html, "html.parser")
    table = soup.find('table')
    actual_output = list(museum_wiki_link_generator(table))

    # Assert
    assert actual_output == expected_output
    

@pytest.mark.parametrize(
    "test_id, input_html, expected_type, expected_size",
    [
        (
            "happy_path_only_type",
            """<table><tr><th>Type</th><td>Art</td></tr></table>""",
            "Art",
            np.nan,
        ),
        (
            "edge_case_empty_table",
            """<table></table>""",
            "N/A",
            np.nan,
        ),
        (
            "edge_case_no_matching_keys",
            """<table><tr><th>Location</th><td>London</td></tr></table>""",
            "N/A",
            np.nan,
        ),
        (
            "edge_case_less_than_two_headers",
            """<table><tr><th>Type</th></tr></table>""",
            "N/A",
            np.nan,
        ),
        (
            "error_case_invalid_size",
            """<table><tr><th>Collection Size</th><td>Invalid</td></tr></table>""",
            "N/A",
            np.nan,
        ),
        (
            "edge_case_empty_values",
            """<table><tr><th>Type</th><td></td></tr><tr><th>Collection Size</th><td></td></tr></table>""",
            "",
            np.nan,
        ),
    ],
)
def test_handle_infobox(test_id, input_html, expected_type, expected_size):
    # Act
    soup = BeautifulSoup(input_html, "html.parser")
    infobox = soup.find('table')
    actual_type, actual_size = handle_infobox(infobox)

    # Assert
    assert actual_type == expected_type
    assert actual_size == expected_size or (np.isnan(actual_size) and np.isnan(expected_size))



