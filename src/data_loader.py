import pandas as pd
import logging
from pathlib import Path
from typing import Optional

# Setup logging
logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)

# Required core columns for pre-match winner prediction
REQUIRED_COLUMNS = [
    "date",
    "season",
    "city",
    "venue",
    "team1",
    "team2",
    "toss_winner",
    "toss_decision",
    "winner"
]

def load_dataset(file_path: Path) -> pd.DataFrame:
    """
    Loads the IPL match dataset from a CSV file.
    
    Args:
        file_path (Path): Path to the CSV file.
        
    Returns:
        pd.DataFrame: Loaded pandas DataFrame.
        
    Raises:
        FileNotFoundError: If the file does not exist.
        ValueError: If the dataset does not meet schema requirements.
    """
    logger.info(f"Loading dataset from {file_path}...")
    
    if not file_path.exists():
        logger.error(f"File not found at: {file_path}")
        raise FileNotFoundError(f"Dataset file not found at: {file_path}")
        
    try:
        # Load the CSV file
        df = pd.read_csv(file_path)
    except Exception as e:
        logger.error(f"Error reading CSV file: {str(e)}")
        raise ValueError(f"Could not read CSV file: {str(e)}")
        
    if df.empty:
        logger.error("The dataset is empty.")
        raise ValueError("Loaded DataFrame is empty.")
        
    # Clean column names (strip whitespaces and convert to lowercase for uniform mapping)
    df.columns = [col.strip().lower() for col in df.columns]
    
    # Map slightly different spelling/naming conventions from various Kaggle versions
    column_mappings = {
        "match_date": "date",
        "toss-winner": "toss_winner",
        "toss-decision": "toss_decision",
        "match_winner": "winner",
        "team_1": "team1",
        "team_2": "team2",
        "match_venue": "venue",
        "match_city": "city"
    }
    df = df.rename(columns=column_mappings)
    
    # Validate columns
    missing_cols = [col for col in REQUIRED_COLUMNS if col not in df.columns]
    if missing_cols:
        logger.error(f"Missing required columns in dataset: {missing_cols}")
        logger.info(f"Available columns: {list(df.columns)}")
        raise ValueError(f"Schema Validation Error: Missing required columns: {missing_cols}")
        
    logger.info(f"Dataset loaded successfully. Shape: {df.shape}")
    return df

if __name__ == "__main__":
    from src.config import RAW_DATA_PATH
    try:
        data = load_dataset(RAW_DATA_PATH)
        print(data.head(2))
    except Exception as e:
        print(f"Error: {e}")
