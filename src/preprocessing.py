import pandas as pd
import numpy as np
import logging
from typing import Tuple
from src.config import TEAM_NAME_MAPPING, RANDOM_STATE

logger = logging.getLogger(__name__)

# Venue-to-city map for imputing missing cities
VENUE_CITY_MAP = {
    "Wankhede Stadium": "Mumbai",
    "Brabourne Stadium": "Mumbai",
    "Dr DY Patil Sports Academy": "Mumbai",
    "Maharashtra Cricket Association Stadium": "Pune",
    "Eden Gardens": "Kolkata",
    "Narendra Modi Stadium": "Ahmedabad",
    "M Chinnaswamy Stadium": "Bengaluru",
    "M.Chinnaswamy Stadium": "Bengaluru",
    "Rajiv Gandhi International Stadium": "Hyderabad",
    "Arun Jaitley Stadium": "Delhi",
    "Feroz Shah Kotla": "Delhi",
    "Punjab Cricket Association IS Bindra Stadium": "Chandigarh",
    "Punjab Cricket Association Stadium, Mohali": "Chandigarh",
    "MA Chidambaram Stadium": "Chennai",
    "MA Chidambaram Stadium, Chepauk": "Chennai",
    "Barsapara Cricket Stadium": "Guwahati",
    "Sawai Mansingh Stadium": "Jaipur",
    "Bharat Ratna Shri Atal Bihari Vajpayee Ekana Cricket Stadium": "Lucknow",
    "Ekana Cricket Stadium": "Lucknow",
    "Himachal Pradesh Cricket Association Stadium": "Dharamshala",
    "Dubai International Cricket Stadium": "Dubai",
    "Sharjah Cricket Stadium": "Sharjah",
    "Sheikh Zayed Stadium": "Abu Dhabi"
}

def clean_data(df: pd.DataFrame) -> pd.DataFrame:
    """
    Cleans raw IPL match data. Standardizes names, handles missing values,
    parses dates, and removes invalid matches.
    """
    logger.info("Cleaning raw match data...")
    df_clean = df.copy()
    
    # 1. Parse date column to datetime
    # We try multiple date formats to handle both DD-MM-YYYY and YYYY-MM-DD
    # errors='coerce' turns unparseable dates into NaT
    df_clean["date"] = pd.to_datetime(df_clean["date"], errors="coerce")
    
    # Drop rows with invalid dates
    initial_rows = len(df_clean)
    df_clean = df_clean.dropna(subset=["date"])
    dropped_dates = initial_rows - len(df_clean)
    if dropped_dates > 0:
        logger.warning(f"Dropped {dropped_dates} rows due to invalid dates.")
        
    # Sort chronologically by date
    df_clean = df_clean.sort_values(by="date").reset_index(drop=True)
    
    # 2. Handle missing city values using venue mapping
    def impute_city(row):
        city = row["city"]
        venue = str(row["venue"])
        if pd.isna(city) or str(city).strip() == "" or city == "nan":
            for ven_key, city_val in VENUE_CITY_MAP.items():
                if ven_key.lower() in venue.lower():
                    return city_val
            return "Unknown"
        return str(city).strip()
        
    df_clean["city"] = df_clean.apply(impute_city, axis=1)
    
    # 3. Drop matches with missing or undefined winners (e.g. no results, abandoned matches)
    df_clean = df_clean.dropna(subset=["winner"])
    df_clean = df_clean[df_clean["winner"].str.strip() != ""]
    df_clean = df_clean[df_clean["winner"] != "no result"]
    df_clean = df_clean[~df_clean["winner"].isna()]
    
    # 4. Standardize team names (handles historical rebrands)
    def clean_team_name(team):
        if pd.isna(team):
            return "Unknown"
        team_str = str(team).strip()
        return TEAM_NAME_MAPPING.get(team_str, team_str)
        
    df_clean["team1_clean"] = df_clean["team1"].apply(clean_team_name)
    df_clean["team2_clean"] = df_clean["team2"].apply(clean_team_name)
    df_clean["toss_winner_clean"] = df_clean["toss_winner"].apply(clean_team_name)
    df_clean["winner_clean"] = df_clean["winner"].apply(clean_team_name)
    
    # 5. Clean venue and city strings
    df_clean["venue_clean"] = df_clean["venue"].astype(str).str.strip()
    df_clean["city_clean"] = df_clean["city"].astype(str).str.strip()
    df_clean["toss_decision"] = df_clean["toss_decision"].astype(str).str.strip().str.lower()
    
    # 6. Validate that winner is one of the two competing teams
    # (sometimes there is "tie" or similar values, we filter those to maintain binary winner modeling)
    def validate_winner(row):
        w = row["winner_clean"]
        t1 = row["team1_clean"]
        t2 = row["team2_clean"]
        return w == t1 or w == t2
        
    valid_mask = df_clean.apply(validate_winner, axis=1)
    df_clean = df_clean[valid_mask].reset_index(drop=True)
    
    # 7. Create binary prediction target: 1 if team1 won, 0 if team2 won
    df_clean["winner_is_team1"] = (df_clean["winner_clean"] == df_clean["team1_clean"]).astype(int)
    
    logger.info(f"Data cleaning finished. Valid matches remaining: {len(df_clean)}")
    return df_clean

def split_data_chronologically(df: pd.DataFrame, test_size: float = 0.2) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """
    Splits the cleaned dataset chronologically into train and test sets to prevent data leakage.
    
    Args:
        df (pd.DataFrame): Sorted cleaned DataFrame.
        test_size (float): Portion of matches to include in test set (e.g. 0.2).
        
    Returns:
        Tuple[pd.DataFrame, pd.DataFrame]: (train_df, test_df)
    """
    logger.info(f"Splitting data chronologically with test size ratio: {test_size}")
    
    # Sort just in case
    df_sorted = df.sort_values(by="date").reset_index(drop=True)
    
    split_index = int(len(df_sorted) * (1 - test_size))
    train_df = df_sorted.iloc[:split_index].copy()
    test_df = df_sorted.iloc[split_index:].copy()
    
    logger.info(f"Split details: Train size: {len(train_df)} (from {train_df['date'].min().date()} to {train_df['date'].max().date()})")
    logger.info(f"Split details: Test size: {len(test_df)} (from {test_df['date'].min().date()} to {test_df['date'].max().date()})")
    
    return train_df, test_df
