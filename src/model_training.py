import os
import joblib
import logging
import pandas as pd
from typing import Tuple, Dict, Any

from src.config import (
    RAW_DATA_PATH,
    PROCESSED_DATA_PATH,
    MODEL_PATH,
    ALL_FEATURES,
    NUMERICAL_FEATURES,
    CATEGORICAL_FEATURES,
    TARGET_COL,
    TEST_SIZE
)
from src.data_loader import load_dataset
from src.preprocessing import clean_data, split_data_chronologically
from src.feature_engineering import compute_historical_features
from src.hyperparameter_tuning import run_hyperparameter_tuning
from src.evaluation import evaluate_predictions, plot_and_save_confusion_matrix, plot_and_save_feature_importance

# Setup logging
logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)

def train_and_select_best_model() -> Dict[str, Any]:
    """
    Orchestrates the training pipeline:
    1. Loads raw dataset
    2. Cleans and preprocesses data
    3. Computes rolling features (and generates inference stats)
    4. Chronologically splits data
    5. Tunes and evaluates Logistic Regression, Random Forest, and Gradient Boosting
    6. Saves the best pipeline and stats to a unified joblib file.
    """
    logger.info("Starting the training pipeline orchestration...")
    
    # 1. Load Data
    raw_df = load_dataset(RAW_DATA_PATH)
    
    # 2. Clean Data
    cleaned_df = clean_data(raw_df)
    
    # 3. Feature Engineering
    featured_df, latest_stats = compute_historical_features(cleaned_df)
    
    # Save the processed data with engineered features for reference/EDA
    featured_df.to_csv(PROCESSED_DATA_PATH, index=False)
    logger.info(f"Engineered dataset saved to: {PROCESSED_DATA_PATH}")
    
    # 4. Split data chronologically
    train_df, test_df = split_data_chronologically(featured_df, test_size=TEST_SIZE)
    
    # Prepare X, y
    X_train = train_df[ALL_FEATURES]
    y_train = train_df[TARGET_COL]
    X_test = test_df[ALL_FEATURES]
    y_test = test_df[TARGET_COL]
    
    candidate_models = ["logistic_regression", "random_forest", "gradient_boosting"]
    best_overall_score = -1.0
    best_model_name = None
    best_pipeline = None
    best_test_metrics = None
    
    results_summary = {}
    
    # 5. Tune and evaluate each model candidate
    for model_name in candidate_models:
        logger.info(f"=== Tuning and Evaluating: {model_name} ===")
        # Run grid search
        tuned_pipeline, best_params, cv_score = run_hyperparameter_tuning(X_train, y_train, model_name)
        
        # Fit on whole train set (GridSearchCV does this automatically, but let's be explicit)
        tuned_pipeline.fit(X_train, y_train)
        
        # Evaluate on holdout test set
        y_pred = tuned_pipeline.predict(X_test)
        test_metrics = evaluate_predictions(y_test, y_pred)
        
        results_summary[model_name] = {
            "cv_accuracy": cv_score,
            "test_metrics": test_metrics,
            "params": best_params
        }
        
        logger.info(f"Model {model_name} - Test Accuracy: {test_metrics['accuracy']:.4f}, F1-score: {test_metrics['f1_score']:.4f}")
        
        # Choose best model based on holdout test set accuracy
        if test_metrics["accuracy"] > best_overall_score:
            best_overall_score = test_metrics["accuracy"]
            best_model_name = model_name
            best_pipeline = tuned_pipeline
            best_test_metrics = test_metrics
            
    logger.info(f"\n=========================================\n"
                f"Winner Model: {best_model_name} with Test Accuracy: {best_overall_score:.4f}\n"
                f"=========================================")
                
    # 6. Generate final plots for the winning model on test set
    y_pred_best = best_pipeline.predict(X_test)
    plot_and_save_confusion_matrix(y_test, y_pred_best)
    
    # Extract feature importances if tree-based model
    classifier = best_pipeline.named_steps["classifier"]
    if hasattr(classifier, "feature_importances_"):
        # Get one-hot encoded feature names from preprocessor to map importances correctly
        preprocessor = best_pipeline.named_steps["preprocessor"]
        
        # Numerical feature names remain unchanged
        num_features = NUMERICAL_FEATURES
        
        # Categorical feature names are one-hot encoded
        cat_features_transformed = []
        if "cat" in preprocessor.named_transformers_:
            ohe = preprocessor.named_transformers_["cat"]
            # Check version support for get_feature_names_out vs get_feature_names
            if hasattr(ohe, "get_feature_names_out"):
                cat_features_transformed = list(ohe.get_feature_names_out(CATEGORICAL_FEATURES))
            else:
                cat_features_transformed = list(ohe.get_feature_names(CATEGORICAL_FEATURES))
                
        feature_names_transformed = num_features + cat_features_transformed
        importances = classifier.feature_importances_
        
        # Map back to group by core feature name for simpler explanation, or plot the full list
        # Let's plot the raw importances but capped/visualized nicely
        plot_and_save_feature_importance(importances, feature_names_transformed)
    
    # 7. Save model artifact (unified pipeline + latest stats lookup)
    model_artifact = {
        "model_name": best_model_name,
        "pipeline": best_pipeline,
        "latest_stats": latest_stats,
        "features": ALL_FEATURES,
        "test_metrics": best_test_metrics,
        "all_results": results_summary
    }
    
    os.makedirs(os.path.dirname(MODEL_PATH), exist_ok=True)
    joblib.dump(model_artifact, MODEL_PATH)
    logger.info(f"Model artifact successfully saved to: {MODEL_PATH}")
    
    return model_artifact

if __name__ == "__main__":
    train_and_select_best_model()
