import os
import joblib
import logging
from datetime import datetime
from fastapi import FastAPI, HTTPException, status
from fastapi.middleware.cors import CORSMiddleware
from pathlib import Path

from src.config import MODEL_PATH, TEAM_NAME_MAPPING
from src.feature_engineering import prepare_inference_row
from backend.schemas import PredictionRequest, PredictionResponse

# Setup logging
logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)

app = FastAPI(
    title="Third Umpire - IPL Predictor API",
    description="Production API for predicting IPL match outcomes using historical rolling features.",
    version="1.0.0"
)

# Enable CORS (Cross-Origin Resource Sharing)
# This allows our Streamlit frontend (or any external UI) to call the API safely
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Global variables for model state
model_pipeline = None
latest_stats = None
model_name = None
model_metrics = None

@app.on_event("startup")
def load_model_artifact():
    """
    Startup handler that loads the trained model pipeline and the historical
    lookup stats from the joblib artifact file.
    """
    global model_pipeline, latest_stats, model_name, model_metrics
    logger.info(f"Looking for model artifact at: {MODEL_PATH}")
    
    if not MODEL_PATH.exists():
        logger.warning("Model artifact not found. Please run training script first (python src/model_training.py) to train the model.")
        return
        
    try:
        artifact = joblib.load(MODEL_PATH)
        model_pipeline = artifact["pipeline"]
        latest_stats = artifact["latest_stats"]
        model_name = artifact["model_name"]
        model_metrics = artifact.get("test_metrics", {})
        logger.info(f"Successfully loaded '{model_name}' model. Test Accuracy: {model_metrics.get('accuracy', 0.0):.4f}")
    except Exception as e:
        logger.error(f"Error loading model artifact: {str(e)}")

@app.get("/")
def read_root():
    """
    Health check / Welcome endpoint.
    """
    model_status = "Loaded" if model_pipeline is not None else "Not Trained (Please run training)"
    return {
        "project": "Third Umpire - IPL Prediction System",
        "status": "online",
        "model_loaded": model_pipeline is not None,
        "model_status": model_status,
        "model_name": model_name,
        "test_accuracy": model_metrics.get("accuracy") if model_metrics else None
    }

@app.post("/predict", response_model=PredictionResponse, status_code=status.HTTP_200_OK)
def predict_winner(request: PredictionRequest):
    """
    Predicts the winner of an IPL match based on pre-match variables.
    
    Validates input teams, cleans names, engineers rolling statistics on the fly
    using the model's lookup dictionary, and feeds features to the pipeline.
    """
    global model_pipeline, latest_stats, model_name
    
    # 1. Check if model is loaded
    if model_pipeline is None:
        logger.error("Prediction request failed: Model is not loaded.")
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="Model is not trained or loaded. Please run the training pipeline first."
        )
        
    # 2. Extract and validate input values
    t1 = request.team1.strip()
    t2 = request.team2.strip()
    v = request.venue.strip()
    toss_w = request.toss_winner.strip()
    toss_d = request.toss_decision.strip().lower()
    city = request.city.strip()
    season = request.season
    
    # Standardize input names to check validity
    t1_clean = TEAM_NAME_MAPPING.get(t1, t1)
    t2_clean = TEAM_NAME_MAPPING.get(t2, t2)
    toss_clean = TEAM_NAME_MAPPING.get(toss_w, toss_w)
    
    if t1_clean == t2_clean:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Team 1 and Team 2 must be different teams."
        )
        
    if toss_clean not in [t1_clean, t2_clean]:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=f"Toss winner must be either {t1} or {t2}."
        )
        
    if toss_d not in ["bat", "field"]:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Toss decision must be either 'bat' or 'field'."
        )
        
    try:
        # 3. Engineer features on the fly using the latest_stats lookup dictionary
        inference_df = prepare_inference_row(
            team1=t1,
            team2=t2,
            venue=v,
            toss_winner=toss_w,
            toss_decision=toss_d,
            city=city,
            season=season,
            latest_stats=latest_stats
        )
        
        # 4. Perform Prediction
        # The pipeline handles scaling (StandardScaler) and encoding (OneHotEncoder) internally!
        pred_label = model_pipeline.predict(inference_df)[0]  # Binary: 1 = Team 1 wins, 0 = Team 2 wins
        pred_proba = model_pipeline.predict_proba(inference_df)[0]  # [Prob(0), Prob(1)]
        
        # 5. Map predicted binary label to team names
        if pred_label == 1:
            predicted_winner = t1_clean
            confidence = float(pred_proba[1])
        else:
            predicted_winner = t2_clean
            confidence = float(pred_proba[0])
            
        # 6. Extract engineered features as dictionary to return in response (for learning purposes)
        # Drop categorical features so we only return the calculated numeric rates
        feature_dict = inference_df.drop(columns=[
            "team1_clean", "team2_clean", "venue_clean", 
            "toss_winner_clean", "toss_decision", "city_clean"
        ]).iloc[0].to_dict()
        
        logger.info(f"Match predicted: {t1} vs {t2} -> Winner: {predicted_winner} (Conf: {confidence:.2f})")
        
        return PredictionResponse(
            predicted_winner=predicted_winner,
            confidence=confidence,
            timestamp=datetime.utcnow(),
            model_used=model_name,
            engineered_features=feature_dict
        )
        
    except Exception as e:
        logger.error(f"Error during prediction pipeline execution: {str(e)}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"An error occurred during prediction: {str(e)}"
        )
