import pandas as pd
import numpy as np
import logging
from typing import Tuple, Dict, Any

logger = logging.getLogger(__name__)

def compute_historical_features(df: pd.DataFrame) -> Tuple[pd.DataFrame, Dict[str, Any]]:
    """
    Computes rolling historical features for each match in the dataframe.
    Features are computed using only matches that occurred BEFORE the match date to prevent data leakage.
    
    Args:
        df (pd.DataFrame): The cleaned DataFrame.
        
    Returns:
        Tuple[pd.DataFrame, Dict[str, Any]]:
            - DataFrame with features appended.
            - A dictionary containing the final calculated stats for inference.
    """
    logger.info("Computing historical features...")
    df_feat = df.copy()
    
    # Initialize empty lists for engineered features
    t1_win_rates = []
    t2_win_rates = []
    t1_forms = []
    t2_forms = []
    h2h_rates = []
    t1_venue_rates = []
    t2_venue_rates = []
    t1_toss_rates = []
    t2_toss_rates = []
    
    # Loop over matches to compute features chronologically
    for idx, row in df_feat.iterrows():
        current_date = row["date"]
        t1 = row["team1_clean"]
        t2 = row["team2_clean"]
        venue = row["venue_clean"]
        
        # Filter matches played strictly BEFORE the current match date
        past_matches = df_feat[df_feat["date"] < current_date]
        
        # 1. Team 1 Cumulative Stats
        t1_past = past_matches[(past_matches["team1_clean"] == t1) | (past_matches["team2_clean"] == t1)]
        t1_played = len(t1_past)
        t1_won = len(t1_past[t1_past["winner_clean"] == t1])
        t1_win_rate = t1_won / t1_played if t1_played > 0 else 0.5
        t1_win_rates.append(t1_win_rate)
        
        # Team 1 Recent Form (last 5 matches)
        t1_last_5 = t1_past.tail(5)
        t1_l5_played = len(t1_last_5)
        t1_l5_won = len(t1_last_5[t1_last_5["winner_clean"] == t1])
        t1_form = t1_l5_won / t1_l5_played if t1_l5_played > 0 else 0.5
        t1_forms.append(t1_form)
        
        # 2. Team 2 Cumulative Stats
        t2_past = past_matches[(past_matches["team1_clean"] == t2) | (past_matches["team2_clean"] == t2)]
        t2_played = len(t2_past)
        t2_won = len(t2_past[t2_past["winner_clean"] == t2])
        t2_win_rate = t2_won / t2_played if t2_played > 0 else 0.5
        t2_win_rates.append(t2_win_rate)
        
        # Team 2 Recent Form (last 5 matches)
        t2_last_5 = t2_past.tail(5)
        t2_l5_played = len(t2_last_5)
        t2_l5_won = len(t2_last_5[t2_last_5["winner_clean"] == t2])
        t2_form = t2_l5_won / t2_l5_played if t2_l5_played > 0 else 0.5
        t2_forms.append(t2_form)
        
        # 3. Head-to-Head Win Rate (t1 vs t2)
        h2h_past = past_matches[
            ((past_matches["team1_clean"] == t1) & (past_matches["team2_clean"] == t2)) |
            ((past_matches["team1_clean"] == t2) & (past_matches["team2_clean"] == t1))
        ]
        h2h_played = len(h2h_past)
        h2h_won_t1 = len(h2h_past[h2h_past["winner_clean"] == t1])
        h2h_rate = h2h_won_t1 / h2h_played if h2h_played > 0 else 0.5
        h2h_rates.append(h2h_rate)
        
        # 4. Venue win rate for Team 1
        t1_venue_past = t1_past[t1_past["venue_clean"] == venue]
        t1_ven_played = len(t1_venue_past)
        t1_ven_won = len(t1_venue_past[t1_venue_past["winner_clean"] == t1])
        t1_venue_rate = t1_ven_won / t1_ven_played if t1_ven_played > 0 else 0.5
        t1_venue_rates.append(t1_venue_rate)
        
        # Venue win rate for Team 2
        t2_venue_past = t2_past[t2_past["venue_clean"] == venue]
        t2_ven_played = len(t2_venue_past)
        t2_ven_won = len(t2_venue_past[t2_venue_past["winner_clean"] == t2])
        t2_venue_rate = t2_ven_won / t2_ven_played if t2_ven_played > 0 else 0.5
        t2_venue_rates.append(t2_venue_rate)
        
        # 5. Toss win-rate to match-win correlation (win match when win toss)
        t1_toss_past = t1_past[t1_past["toss_winner_clean"] == t1]
        t1_toss_won = len(t1_toss_past[t1_toss_past["winner_clean"] == t1])
        t1_toss_rate = t1_toss_won / len(t1_toss_past) if len(t1_toss_past) > 0 else 0.5
        t1_toss_rates.append(t1_toss_rate)
        
        t2_toss_past = t2_past[t2_past["toss_winner_clean"] == t2]
        t2_toss_won = len(t2_toss_past[t2_toss_past["winner_clean"] == t2])
        t2_toss_rate = t2_toss_won / len(t2_toss_past) if len(t2_toss_past) > 0 else 0.5
        t2_toss_rates.append(t2_toss_rate)
        
    # Append features to dataframe
    df_feat["team1_win_rate_cumulative"] = t1_win_rates
    df_feat["team2_win_rate_cumulative"] = t2_win_rates
    df_feat["team1_form_recent"] = t1_forms
    df_feat["team2_form_recent"] = t2_forms
    df_feat["h2h_team1_win_rate"] = h2h_rates
    df_feat["team1_venue_win_rate"] = t1_venue_rates
    df_feat["team2_venue_win_rate"] = t2_venue_rates
    df_feat["team1_toss_win_rate"] = t1_toss_rates
    df_feat["team2_toss_win_rate"] = t2_toss_rates
    
    # Create the final stats state dictionary to use for inference
    # (using all matches in the dataset to represent the final state)
    latest_stats = {}
    
    unique_teams = pd.concat([df_feat["team1_clean"], df_feat["team2_clean"]]).unique()
    latest_stats["teams"] = {}
    for team in unique_teams:
        team_m = df_feat[(df_feat["team1_clean"] == team) | (df_feat["team2_clean"] == team)]
        t_played = len(team_m)
        t_won = len(team_m[team_m["winner_clean"] == team])
        win_rate = t_won / t_played if t_played > 0 else 0.5
        
        # Recent matches (last 5)
        last_5 = team_m.tail(5)
        l5_won = len(last_5[last_5["winner_clean"] == team])
        form = l5_won / len(last_5) if len(last_5) > 0 else 0.5
        
        # Toss win stats
        t_toss = team_m[team_m["toss_winner_clean"] == team]
        t_toss_won = len(t_toss[t_toss["winner_clean"] == team])
        toss_win_rate = t_toss_won / len(t_toss) if len(t_toss) > 0 else 0.5
        
        latest_stats["teams"][team] = {
            "win_rate": win_rate,
            "form": form,
            "toss_win_rate": toss_win_rate
        }
        
    # Head-to-Head lookup
    latest_stats["h2h"] = {}
    for i in range(len(unique_teams)):
        for j in range(i+1, len(unique_teams)):
            tA = unique_teams[i]
            tB = unique_teams[j]
            h2h_m = df_feat[((df_feat["team1_clean"] == tA) & (df_feat["team2_clean"] == tB)) |
                            ((df_feat["team1_clean"] == tB) & (df_feat["team2_clean"] == tA))]
            played = len(h2h_m)
            won_tA = len(h2h_m[h2h_m["winner_clean"] == tA])
            latest_stats["h2h"][f"{tA}_vs_{tB}"] = won_tA / played if played > 0 else 0.5
            latest_stats["h2h"][f"{tB}_vs_{tA}"] = (played - won_tA) / played if played > 0 else 0.5
            
    # Venue lookup
    latest_stats["venue"] = {}
    unique_venues = df_feat["venue_clean"].unique()
    for ven in unique_venues:
        latest_stats["venue"][ven] = {}
        for team in unique_teams:
            team_ven_m = df_feat[((df_feat["team1_clean"] == team) | (df_feat["team2_clean"] == team)) & 
                                 (df_feat["venue_clean"] == ven)]
            played = len(team_ven_m)
            won = len(team_ven_m[team_ven_m["winner_clean"] == team])
            latest_stats["venue"][ven][team] = won / played if played > 0 else 0.5
            
    logger.info("Feature engineering finished successfully.")
    return df_feat, latest_stats

def prepare_inference_row(
    team1: str,
    team2: str,
    venue: str,
    toss_winner: str,
    toss_decision: str,
    city: str,
    season: int,
    latest_stats: Dict[str, Any]
) -> pd.DataFrame:
    """
    Constructs a single row of features for a new match to be predicted.
    Uses the saved latest_stats state dictionary to look up values.
    
    Args:
        team1 (str): Name of team 1.
        team2 (str): Name of team 2.
        venue (str): Name of the venue.
        toss_winner (str): Toss winner.
        toss_decision (str): Toss decision.
        city (str): City.
        season (int): Season year.
        latest_stats (dict): The lookup dictionary created during training.
        
    Returns:
        pd.DataFrame: A single-row DataFrame formatted for prediction.
    """
    # 1. Clean Inputs (Standardize team names)
    from src.config import TEAM_NAME_MAPPING
    t1_clean = TEAM_NAME_MAPPING.get(team1.strip(), team1.strip())
    t2_clean = TEAM_NAME_MAPPING.get(team2.strip(), team2.strip())
    toss_clean = TEAM_NAME_MAPPING.get(toss_winner.strip(), toss_winner.strip())
    venue_clean = venue.strip()
    city_clean = city.strip()
    
    # 2. Extract stats from lookup dict
    t_stats = latest_stats.get("teams", {})
    t1_stats = t_stats.get(t1_clean, {"win_rate": 0.5, "form": 0.5, "toss_win_rate": 0.5})
    t2_stats = t_stats.get(t2_clean, {"win_rate": 0.5, "form": 0.5, "toss_win_rate": 0.5})
    
    # Head-to-Head win rate
    h2h_key = f"{t1_clean}_vs_{t2_clean}"
    h2h_win_rate = latest_stats.get("h2h", {}).get(h2h_key, 0.5)
    
    # Venue stats
    v_stats = latest_stats.get("venue", {}).get(venue_clean, {})
    t1_ven_rate = v_stats.get(t1_clean, 0.5)
    t2_ven_rate = v_stats.get(t2_clean, 0.5)
    
    # 3. Assemble row dictionary
    row_dict = {
        "team1_clean": [t1_clean],
        "team2_clean": [t2_clean],
        "venue_clean": [venue_clean],
        "toss_winner_clean": [toss_clean],
        "toss_decision": [toss_decision.strip().lower()],
        "city_clean": [city_clean],
        "team1_win_rate_cumulative": [t1_stats["win_rate"]],
        "team2_win_rate_cumulative": [t2_stats["win_rate"]],
        "team1_form_recent": [t1_stats["form"]],
        "team2_form_recent": [t2_stats["form"]],
        "h2h_team1_win_rate": [h2h_win_rate],
        "team1_venue_win_rate": [t1_ven_rate],
        "team2_venue_win_rate": [t2_ven_rate],
        "team1_toss_win_rate": [t1_stats["toss_win_rate"]],
        "team2_toss_win_rate": [t2_stats["toss_win_rate"]]
    }
    
    return pd.DataFrame(row_dict)
