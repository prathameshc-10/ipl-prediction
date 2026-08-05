import logging
from sklearn.model_selection import GridSearchCV, TimeSeriesSplit
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from typing import Dict, Any, Tuple
from src.config import CATEGORICAL_FEATURES, NUMERICAL_FEATURES, RANDOM_STATE

logger = logging.getLogger(__name__)

def create_preprocessing_pipeline() -> ColumnTransformer:
    """
    Creates a scikit-learn ColumnTransformer that scales numerical features 
    and one-hot encodes categorical features.
    """
    numeric_transformer = StandardScaler()
    categorical_transformer = OneHotEncoder(handle_unknown="ignore")
    
    preprocessor = ColumnTransformer(
        transformers=[
            ("num", numeric_transformer, NUMERICAL_FEATURES),
            ("cat", categorical_transformer, CATEGORICAL_FEATURES)
        ]
    )
    return preprocessor

def get_param_grid(model_type: str) -> Dict[str, list]:
    """
    Returns the hyperparameter grid for tuning the given model type.
    Note that parameters are prefixed with 'classifier__' because they are part of a Pipeline.
    """
    if model_type == "logistic_regression":
        return {
            "classifier__C": [0.01, 0.1, 1.0, 10.0],
            "classifier__solver": ["liblinear", "lbfgs"]
        }
    elif model_type == "random_forest":
        return {
            "classifier__n_estimators": [50, 100, 200],
            "classifier__max_depth": [3, 5, 8, None],
            "classifier__min_samples_split": [2, 5, 10]
        }
    elif model_type == "gradient_boosting":
        return {
            "classifier__n_estimators": [50, 100, 150],
            "classifier__learning_rate": [0.01, 0.05, 0.1, 0.2],
            "classifier__max_depth": [3, 4, 5]
        }
    else:
        raise ValueError(f"Unknown model type: {model_type}")

def run_hyperparameter_tuning(
    X_train: Any, 
    y_train: Any, 
    model_type: str
) -> Tuple[Pipeline, Dict[str, Any], float]:
    """
    Executes GridSearchCV using a TimeSeriesSplit to find the best hyperparameters
    for the selected model type.
    
    Args:
        X_train: Features training set.
        y_train: Target training set.
        model_type (str): Type of model ('logistic_regression', 'random_forest', 'gradient_boosting')
        
    Returns:
        Tuple[Pipeline, dict, float]:
            - The optimized pipeline.
            - The best hyperparameters.
            - The best cross-validation score.
    """
    logger.info(f"Setting up hyperparameter tuning for {model_type}...")
    
    # 1. Create base classifier
    if model_type == "logistic_regression":
        base_model = LogisticRegression(random_state=RANDOM_STATE, max_iter=1000)
    elif model_type == "random_forest":
        base_model = RandomForestClassifier(random_state=RANDOM_STATE)
    elif model_type == "gradient_boosting":
        base_model = GradientBoostingClassifier(random_state=RANDOM_STATE)
    else:
        raise ValueError(f"Unsupported model type: {model_type}")
        
    # 2. Build Pipeline
    preprocessor = create_preprocessing_pipeline()
    pipeline = Pipeline(steps=[
        ("preprocessor", preprocessor),
        ("classifier", base_model)
    ])
    
    # 3. Setup TimeSeriesSplit (respects chronology of cricket matches)
    tscv = TimeSeriesSplit(n_splits=3)
    
    # 4. Run Grid Search
    param_grid = get_param_grid(model_type)
    grid_search = GridSearchCV(
        estimator=pipeline,
        param_grid=param_grid,
        cv=tscv,
        scoring="accuracy",
        n_jobs=-1,
        verbose=1
    )
    
    logger.info(f"Starting GridSearchCV for {model_type} on train split...")
    grid_search.fit(X_train, y_train)
    
    best_pipeline = grid_search.best_estimator_
    best_params = grid_search.best_params_
    best_score = grid_search.best_score_
    
    logger.info(f"GridSearchCV completed. Best CV Accuracy: {best_score:.4f}")
    logger.info(f"Best parameters: {best_params}")
    
    return best_pipeline, best_params, best_score
