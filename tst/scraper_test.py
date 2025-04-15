import pandas as pd
import pytest
import numpy as np
from src.scraper import clean_collection_size, convert_million_values, extract_first_city_part

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


