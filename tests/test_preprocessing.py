import pytest
import pandas as pd
import numpy as np
from src.preprocessing import clean_data, split_data_chronologically

@pytest.fixture
def raw_dummy_df() -> pd.DataFrame:
    """Fixture that generates raw, messy data for cleaning tests."""
    data = {
        "date": ["2022-03-26", "2022-03-27", "2022-03-28", "2022-03-29", "2022-03-30"],
        "season": [2022, 2022, 2022, 2022, 2022],
        "city": [None, "Mumbai", "", "Pune", "Dubai"], # None and empty string
        "venue": [
            "Wankhede Stadium, Mumbai", 
            "Brabourne Stadium, Mumbai", 
            "M Chinnaswamy Stadium, Bengaluru", 
            "Maharashtra Cricket Association Stadium, Pune", 
            "Dubai International Cricket Stadium"
        ],
        "team1": ["Chennai Super Kings", "Kings XI Punjab", "Mumbai Indians", "Rajasthan Royals", "Delhi Daredevils"],
        "team2": ["Kolkata Knight Riders", "Delhi Capitals", "Royal Challengers Bangalore", "Sunrisers Hyderabad", "Mumbai Indians"],
        "toss_winner": ["Kolkata Knight Riders", "Delhi Capitals", "Mumbai Indians", "Sunrisers Hyderabad", "Delhi Daredevils"],
        "toss_decision": ["field", "field", "bat", "field", "field"],
        "winner": ["Kolkata Knight Riders", "Delhi Capitals", "Mumbai Indians", "no result", "Delhi Daredevils"] # No result match should be dropped
    }
    return pd.DataFrame(data)

def test_clean_data_city_imputation(raw_dummy_df):
    """Verifies that missing cities are imputed based on the venue name map."""
    cleaned = clean_data(raw_dummy_df)
    # The first row (index 0) had None city at Wankhede, which is in Mumbai
    assert cleaned.loc[0, "city_clean"] == "Mumbai"
    # The third row (index 2 after sorting) had "" city at M Chinnaswamy, which is in Bengaluru
    assert cleaned.loc[2, "city_clean"] == "Bengaluru"

def test_clean_data_standardization(raw_dummy_df):
    """Verifies that historical rebranding and name variants are standard."""
    cleaned = clean_data(raw_dummy_df)
    # "Kings XI Punjab" (row 2, sorted index 1) standardizes to "Punjab Kings"
    assert cleaned.loc[1, "team1_clean"] == "Punjab Kings"
    # "Delhi Daredevils" (row 5, sorted index 3) standardizes to "Delhi Capitals"
    assert cleaned.loc[3, "team1_clean"] == "Delhi Capitals"

def test_clean_data_filters_invalid(raw_dummy_df):
    """Verifies that matches with 'no result' winner are correctly dropped."""
    cleaned = clean_data(raw_dummy_df)
    # Total input: 5 rows. 1 is "no result" winner (dropped), leaving 4 rows.
    assert len(cleaned) == 4
    assert "no result" not in cleaned["winner_clean"].values

def test_clean_data_creates_target(raw_dummy_df):
    """Verifies that 'winner_is_team1' binary label is generated correctly."""
    cleaned = clean_data(raw_dummy_df)
    # Row 3 (sorted index 2): team1="Mumbai Indians", winner="Mumbai Indians" -> winner_is_team1=1
    assert cleaned.loc[2, "winner_is_team1"] == 1
    # Row 2 (sorted index 1): team1="Punjab Kings", winner="Delhi Capitals" -> winner_is_team1=0
    assert cleaned.loc[1, "winner_is_team1"] == 0

def test_split_data_chronologically(raw_dummy_df):
    """Verifies that splitting splits sequentially and respects test size ratio."""
    cleaned = clean_data(raw_dummy_df) # Length is 4
    train, test = split_data_chronologically(cleaned, test_size=0.25)
    
    # 25% of 4 = 1 row in test, 3 rows in train
    assert len(train) == 3
    assert len(test) == 1
    
    # Assert chronological ordering (all train dates should be <= test date)
    assert train["date"].max() <= test["date"].min()
