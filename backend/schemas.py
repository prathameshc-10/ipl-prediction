from pydantic import BaseModel, Field
from typing import Dict, Optional
from datetime import datetime

class PredictionRequest(BaseModel):
    team1: str = Field(..., description="Name of Team 1", example="Chennai Super Kings")
    team2: str = Field(..., description="Name of Team 2", example="Mumbai Indians")
    venue: str = Field(..., description="Match venue stadium", example="Wankhede Stadium, Mumbai")
    toss_winner: str = Field(..., description="Toss winning team name", example="Mumbai Indians")
    toss_decision: str = Field(..., description="Toss decision: 'bat' or 'field'", example="field")
    city: str = Field(..., description="City where the match is played", example="Mumbai")
    season: int = Field(2026, description="IPL Season/Year", example=2026)

class PredictionResponse(BaseModel):
    predicted_winner: str = Field(..., description="The team predicted to win the match")
    confidence: float = Field(..., description="Probability / confidence score of the predicted winner (0.0 to 1.0)")
    timestamp: datetime = Field(default_factory=datetime.utcnow, description="Time of prediction")
    model_used: str = Field(..., description="Classifier algorithm name used for prediction")
    engineered_features: Dict[str, float] = Field(..., description="The rolling/historical features engineered for this prediction")
