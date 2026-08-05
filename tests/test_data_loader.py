import pytest
import pandas as pd
from pathlib import Path
from src.data_loader import load_dataset

@pytest.fixture
def temp_csv_path(tmp_path) -> Path:
    """Fixture that creates a temporary valid match CSV dataset."""
    csv_file = tmp_path / "valid_matches.csv"
    data = {
        "Date": ["2022-03-26", "2022-03-27"],
        "Season": [2022, 2022],
        "City": ["Mumbai", "Mumbai"],
        "Venue": ["Wankhede Stadium, Mumbai", "Brabourne Stadium, Mumbai"],
        "Team1": ["Chennai Super Kings", "Mumbai Indians"],
        "Team2": ["Kolkata Knight Riders", "Delhi Capitals"],
        "Toss_Winner": ["Kolkata Knight Riders", "Delhi Capitals"],
        "Toss_Decision": ["field", "field"],
        "Winner": ["Kolkata Knight Riders", "Delhi Capitals"],
        "team1_runs": [131, 177],
        "team1_wickets": [5, 5],
        "team2_runs": [133, 179],
        "team2_wickets": [4, 6]
    }
    df = pd.DataFrame(data)
    df.to_csv(csv_file, index=False)
    return csv_file

@pytest.fixture
def temp_invalid_csv_path(tmp_path) -> Path:
    """Fixture that creates an invalid match CSV dataset (missing 'Winner')."""
    csv_file = tmp_path / "invalid_matches.csv"
    data = {
        "Date": ["2022-03-26"],
        "Season": [2022],
        "City": ["Mumbai"],
        "Venue": ["Wankhede Stadium, Mumbai"],
        "Team1": ["Chennai Super Kings"],
        "Team2": ["Kolkata Knight Riders"],
        "Toss_Winner": ["Kolkata Knight Riders"],
        "Toss_Decision": ["field"]
        # 'Winner' column is missing!
    }
    df = pd.DataFrame(data)
    df.to_csv(csv_file, index=False)
    return csv_file

def test_load_dataset_valid(temp_csv_path):
    """Verifies that a valid CSV dataset is loaded correctly and standardized."""
    df = load_dataset(temp_csv_path)
    assert isinstance(df, pd.DataFrame)
    assert len(df) == 2
    # Standardized columns should be lowercase
    assert "date" in df.columns
    assert "toss_winner" in df.columns
    assert "winner" in df.columns

def test_load_dataset_invalid_schema(temp_invalid_csv_path):
    """Verifies that loading a CSV with missing required columns raises a ValueError."""
    with pytest.raises(ValueError) as excinfo:
        load_dataset(temp_invalid_csv_path)
    assert "Missing required columns" in str(excinfo.value)

def test_load_dataset_nonexistent_file():
    """Verifies that loading a non-existent file raises a FileNotFoundError."""
    with pytest.raises(FileNotFoundError):
        load_dataset(Path("non_existent_file.csv"))
