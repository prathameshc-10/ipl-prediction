import pytest
import pandas as pd
from src.preprocessing import clean_data
from src.feature_engineering import compute_historical_features, prepare_inference_row

@pytest.fixture
def clean_dummy_df() -> pd.DataFrame:
    """Fixture that generates a cleaned dataframe representing 3 consecutive matches."""
    # Dates are chronological
    data = {
        "date": pd.to_datetime(["2022-03-20", "2022-03-21", "2022-03-22"]),
        "season": [2022, 2022, 2022],
        "city": ["Mumbai", "Mumbai", "Mumbai"],
        "venue": ["Wankhede Stadium", "Wankhede Stadium", "Wankhede Stadium"],
        "team1": ["Chennai Super Kings", "Mumbai Indians", "Chennai Super Kings"],
        "team2": ["Mumbai Indians", "Kolkata Knight Riders", "Mumbai Indians"],
        "toss_winner": ["Chennai Super Kings", "Mumbai Indians", "Mumbai Indians"],
        "toss_decision": ["field", "field", "field"],
        "winner": ["Chennai Super Kings", "Mumbai Indians", "Chennai Super Kings"]
    }
    df = pd.DataFrame(data)
    # Run standardized clean first
    return clean_data(df)

def test_compute_historical_features_columns(clean_dummy_df):
    """Verifies that all required numerical engineered features are generated."""
    featured_df, latest_stats = compute_historical_features(clean_dummy_df)
    
    expected_cols = [
        "team1_win_rate_cumulative",
        "team2_win_rate_cumulative",
        "team1_form_recent",
        "team2_form_recent",
        "h2h_team1_win_rate",
        "team1_venue_win_rate",
        "team2_venue_win_rate",
        "team1_toss_win_rate",
        "team2_toss_win_rate"
    ]
    for col in expected_cols:
        assert col in featured_df.columns
        assert not featured_df[col].isnull().any()

def test_no_leakage_on_first_match(clean_dummy_df):
    """Verifies that rolling features default to 0.5 for the first match since history is empty."""
    featured_df, _ = compute_historical_features(clean_dummy_df)
    
    # Check row 0 (the first match in date order)
    first_row = featured_df.iloc[0]
    assert first_row["team1_win_rate_cumulative"] == 0.5
    assert first_row["team2_win_rate_cumulative"] == 0.5
    assert first_row["h2h_team1_win_rate"] == 0.5
    assert first_row["team1_venue_win_rate"] == 0.5

def test_inference_row_builder(clean_dummy_df):
    """Verifies that the inference row is constructed correctly using lookup tables."""
    _, latest_stats = compute_historical_features(clean_dummy_df)
    
    # Build a prediction row for CSK vs Mumbai at Wankhede
    inference_df = prepare_inference_row(
        team1="Chennai Super Kings",
        team2="Mumbai Indians",
        venue="Wankhede Stadium",
        toss_winner="Mumbai Indians",
        toss_decision="field",
        city="Mumbai",
        season=2026,
        latest_stats=latest_stats
    )
    
    assert isinstance(inference_df, pd.DataFrame)
    assert len(inference_df) == 1
    # Check values mapped from lookup dict
    # At the end of clean_dummy_df:
    # CSK played 2 games (won 2) -> win_rate should be 100% (1.0)
    assert inference_df.loc[0, "team1_win_rate_cumulative"] == 1.0
    # Mumbai played 3 games (won 1, lost 2) -> win_rate should be 1/3 (0.33)
    assert round(inference_df.loc[0, "team2_win_rate_cumulative"], 2) == 0.33
