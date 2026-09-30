import streamlit as st
import pandas as pd
import requests
import joblib
import os
import datetime
from pathlib import Path

# Set Page Configuration
st.set_page_config(
    page_title="Third Umpire | IPL Match Predictor",
    page_icon="🏏",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom Premium Styling
st.markdown("""
<style>
    .main-header {
        font-size: 2.8rem;
        font-weight: 800;
        background: linear-gradient(135deg, #1e3a8a, #3b82f6);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        margin-bottom: 0.5rem;
    }
    .sub-header {
        font-size: 1.2rem;
        color: #4b5563;
        margin-bottom: 2rem;
    }
    .metric-card {
        background-color: #f8fafc;
        border: 1px solid #e2e8f0;
        padding: 1.5rem;
        border-radius: 0.75rem;
        text-align: center;
        box-shadow: 0 1px 3px rgba(0,0,0,0.05);
    }
    .metric-value {
        font-size: 2rem;
        font-weight: 700;
        color: #1e3a8a;
    }
    .metric-label {
        font-size: 0.9rem;
        color: #64748b;
        font-weight: 500;
    }
    .prediction-card {
        background: linear-gradient(135deg, #eff6ff, #dbeafe);
        border: 1px solid #bfdbfe;
        padding: 2rem;
        border-radius: 1rem;
        margin-top: 1.5rem;
    }
    .prediction-card h3,
    .prediction-card h3 strong {
        color: #000000 !important;
    }
</style>
""", unsafe_allow_html=True)

# Helper Paths
PROJECT_ROOT = Path(__file__).resolve().parent.parent
MODEL_PATH = PROJECT_ROOT / "models" / "best_model.pkl"
RAW_DATA_PATH = PROJECT_ROOT / "data" / "raw_matches.csv"
API_URL = "http://127.0.0.1:8000/predict"

# Default fallback values for dropdowns
DEFAULT_TEAMS = [
    "Chennai Super Kings",
    "Delhi Capitals",
    "Gujarat Titans",
    "Kolkata Knight Riders",
    "Lucknow Super Giants",
    "Mumbai Indians",
    "Punjab Kings",
    "Rajasthan Royals",
    "Royal Challengers Bangalore",
    "Sunrisers Hyderabad"
]

DEFAULT_VENUES = [
    "Wankhede Stadium, Mumbai",
    "Brabourne Stadium, Mumbai",
    "Dr DY Patil Sports Academy, Mumbai",
    "M Chinnaswamy Stadium, Bengaluru",
    "Eden Gardens, Kolkata",
    "Narendra Modi Stadium, Ahmedabad",
    "Rajiv Gandhi International Stadium, Uppal, Hyderabad",
    "MA Chidambaram Stadium, Chepauk, Chennai",
    "Arun Jaitley Stadium, Delhi",
    "Punjab Cricket Association IS Bindra Stadium, Mohali, Chandigarh",
    "Bharat Ratna Shri Atal Bihari Vajpayee Ekana Cricket Stadium, Lucknow"
]

DEFAULT_CITIES = [
    "Mumbai", "Bengaluru", "Kolkata", "Ahmedabad", "Hyderabad", 
    "Chennai", "Delhi", "Chandigarh", "Lucknow", "Pune", "Jaipur", "Dharamshala"
]

# ----------------- LOAD ARTIFACTS & DATA -----------------
@st.cache_resource
def load_model_artifact():
    if MODEL_PATH.exists():
        try:
            return joblib.load(MODEL_PATH)
        except Exception:
            return None
    return None

@st.cache_data
def load_raw_data():
    if RAW_DATA_PATH.exists():
        try:
            return pd.read_csv(RAW_DATA_PATH)
        except Exception:
            return None
    return None

model_artifact = load_model_artifact()
raw_data = load_raw_data()

# Extract dynamic names from trained model lookup tables if available
if model_artifact and "latest_stats" in model_artifact:
    teams_list = sorted(list(model_artifact["latest_stats"]["teams"].keys()))
    venues_list = sorted(list(model_artifact["latest_stats"]["venue"].keys()))
else:
    teams_list = sorted(DEFAULT_TEAMS)
    venues_list = sorted(DEFAULT_VENUES)

cities_list = sorted(DEFAULT_CITIES)

# ----------------- SIDEBAR: EDUCATION CENTER -----------------
with st.sidebar:
    st.image("https://img.icons8.com/color/96/cricket.png", width=80)
    st.markdown("## 🧠 ML Education Center")
    st.markdown("Welcome, freshers! Here are the core concepts used in this prediction project:")
    
    with st.expander("📊 Exploratory Data Analysis (EDA)"):
        st.write("""
        **EDA** is the process of examining the dataset to summarize its main characteristics. 
        In our project:
        - We analyzed the **toss decision** (bat vs field).
        - We verified if **toss winners** have an advantage.
        - We verified target balance to ensure class balance.
        """)
        
    with st.expander("🛠️ Feature Engineering"):
        st.write("""
        **Feature Engineering** is creating new input features from raw data to improve model learning.
        We engineered rolling features strictly up to the day before the match:
        - **Team Strength**: Cumulative historical win rate.
        - **Recent Form**: Win rate of the last 5 matches.
        - **Head-to-Head (H2H)**: Win rate of Team 1 vs Team 2 in their past encounters.
        - **Venue Advantage**: Venue win rate for both teams.
        """)
        
    with st.expander("⏳ Chronological Split"):
        st.write("""
        Instead of a random train-test split, we split matches **chronologically** (e.g., training on earlier seasons and testing on the final season).
        
        **Why?** In the real world, you cannot use future match results to predict past match results. Random splitting causes data leakage!
        """)
        
    with st.expander("📈 Evaluation Metrics"):
        st.write("""
        We evaluate our model using:
        - **Accuracy**: Portion of matches predicted correctly.
        - **Precision**: Of all matches we predicted team wins, how many did they actually win?
        - **Recall**: Of all matches team actually won, how many did we predict correctly?
        - **F1 Score**: Harmonic mean of Precision and Recall.
        """)
        
    st.divider()

# ----------------- MAIN INTERFACE -----------------
st.markdown('<div class="main-header">Third Umpire: IPL Prediction System</div>', unsafe_allow_html=True)
st.markdown('<div class="sub-header">An educational machine learning application for predicting IPL match winners</div>', unsafe_allow_html=True)

# Create Tabs
tab_predict, tab_analytics, tab_insights = st.tabs([
    "🔮 Predict Match Winner", 
    "📈 Model Analytics & Performance", 
    "🗃️ Dataset Explorer & Insights"
])

# ----------------- TAB 1: PREDICT -----------------
with tab_predict:
    st.write("Enter details for an upcoming match to predict the winner using the trained ML model.")
    
    # Check if model is loaded
    if model_artifact is None:
        st.warning("⚠️ **Model artifact not trained yet.** Please run the training pipeline first: `python src/model_training.py` or run the setup script.")
    else:
        # Form for user input
        with st.form("prediction_form"):
            col1, col2 = st.columns(2)
            
            with col1:
                team1 = st.selectbox("Select Team 1 (Home Team)", teams_list, index=0)
                venue = st.selectbox("Select Venue", venues_list, index=0)
                toss_winner = st.selectbox("Select Toss Winner", teams_list, index=0)
                city = st.selectbox("Select City", cities_list, index=0)
                
            with col2:
                team2 = st.selectbox("Select Team 2 (Away Team)", teams_list, index=1)
                toss_decision = st.selectbox("Toss Decision", ["field", "bat"], index=0)
                season = st.number_input("Season (Year)", min_value=2008, max_value=2030, value=2026)
                
            submit_button = st.form_submit_button("⚾ Predict Winner")
            
        if submit_button:
            # Inputs validation
            if team1 == team2:
                st.error("Error: Team 1 and Team 2 must be different teams.")
            elif toss_winner not in [team1, team2]:
                st.error(f"Error: Toss winner must be either {team1} or {team2}.")
            else:
                with st.spinner("Analyzing historical matchups and generating prediction..."):
                    # Assemble request payload
                    payload = {
                        "team1": team1,
                        "team2": team2,
                        "venue": venue,
                        "toss_winner": toss_winner,
                        "toss_decision": toss_decision,
                        "city": city,
                        "season": int(season)
                    }
                    
                    try:
                        # Call FastAPI endpoint
                        response = requests.post(API_URL, json=payload, timeout=5)
                        
                        if response.status_code == 200:
                            result = response.json()
                            pred_winner = result["predicted_winner"]
                            confidence = result["confidence"]
                            eng_features = result["engineered_features"]
                            model_used = result["model_used"]
                            
                            # Success Card
                            st.markdown(f"""
                            <div class="prediction-card">
                                <h3>🏆 Predicted Winner: <strong>{pred_winner}</strong></h3>
                                <p style="font-size:1.1rem; color:#1e40af;">Confidence Score: <strong>{confidence*100:.1f}%</strong></p>
                            </div>
                            """, unsafe_allow_html=True)
                            
                            st.progress(confidence)
                            st.caption(f"Model used: {model_used} | Prediction computed at: {result['timestamp']}")
                            
                            # Display engineered features (very educational)
                            st.divider()
                            st.subheader("🛠️ Engineered Pre-Match Features (Inference Lookup)")
                            st.write("These historical rates were fetched from the database for the prediction:")
                            
                            feat_df = pd.DataFrame({
                                "Feature Description": [
                                    f"Historical Win Rate of {team1}",
                                    f"Historical Win Rate of {team2}",
                                    f"Recent Form of {team1} (Last 5 Matches)",
                                    f"Recent Form of {team2} (Last 5 Matches)",
                                    f"Head-to-Head Win Rate of {team1} vs {team2}",
                                    f"Win Rate of {team1} at {venue.split(',')[0]}",
                                    f"Win Rate of {team2} at {venue.split(',')[0]}",
                                    f"Toss Win Correlation of {team1}",
                                    f"Toss Win Correlation of {team2}"
                                ],
                                "Value": [
                                    eng_features["team1_win_rate_cumulative"],
                                    eng_features["team2_win_rate_cumulative"],
                                    eng_features["team1_form_recent"],
                                    eng_features["team2_form_recent"],
                                    eng_features["h2h_team1_win_rate"],
                                    eng_features["team1_venue_win_rate"],
                                    eng_features["team2_venue_win_rate"],
                                    eng_features["team1_toss_win_rate"],
                                    eng_features["team2_toss_win_rate"]
                                ]
                            })
                            
                            # Format value as percentage
                            feat_df["Value"] = feat_df["Value"].apply(lambda val: f"{val*100:.1f}%")
                            st.table(feat_df)
                            
                        else:
                            st.error(f"API Error ({response.status_code}): {response.json().get('detail', 'Unknown error')}")
                            
                    except requests.exceptions.ConnectionError:
                        st.warning("⚠️ **Could not connect to FastAPI server.** Falling back to local model loading...")
                        
                        # Resilient prediction using direct joblib load
                        from src.feature_engineering import prepare_inference_row
                        
                        latest_stats = model_artifact["latest_stats"]
                        pipeline = model_artifact["pipeline"]
                        model_name = model_artifact["model_name"]
                        
                        inference_df = prepare_inference_row(
                            team1=team1, team2=team2, venue=venue, 
                            toss_winner=toss_winner, toss_decision=toss_decision,
                            city=city, season=int(season), latest_stats=latest_stats
                        )
                        
                        pred_label = pipeline.predict(inference_df)[0]
                        pred_proba = pipeline.predict_proba(inference_df)[0]
                        
                        if pred_label == 1:
                            pred_winner = team1
                            confidence = float(pred_proba[1])
                        else:
                            pred_winner = team2
                            confidence = float(pred_proba[0])
                            
                        # Success Card Local
                        st.markdown(f"""
                        <div class="prediction-card">
                            <h3>🏆 Predicted Winner: <strong>{pred_winner}</strong> (Local Forecast)</h3>
                            <p style="font-size:1.1rem; color:#1e40af;">Confidence Score: <strong>{confidence*100:.1f}%</strong></p>
                        </div>
                        """, unsafe_allow_html=True)
                        st.progress(confidence)
                        st.caption(f"Model used: {model_name} (Loaded locally) | Start FastAPI server for complete production setup.")

# ----------------- TAB 2: ANALYTICS -----------------
with tab_analytics:
    st.subheader("📊 Model Performance & Metrics")
    
    if model_artifact is None:
        st.warning("⚠️ No model trained. Go to terminal and run the training pipeline to see performance analytics.")
    else:
        # Display global metrics
        metrics = model_artifact.get("test_metrics", {})
        
        col_acc, col_prec, col_rec, col_f1 = st.columns(4)
        with col_acc:
            st.markdown(f"""
            <div class="metric-card">
                <div class="metric-value">{metrics.get('accuracy', 0.0)*100:.1f}%</div>
                <div class="metric-label">Test Accuracy</div>
            </div>
            """, unsafe_allow_html=True)
        with col_prec:
            st.markdown(f"""
            <div class="metric-card">
                <div class="metric-value">{metrics.get('precision', 0.0)*100:.1f}%</div>
                <div class="metric-label">Precision Score</div>
            </div>
            """, unsafe_allow_html=True)
        with col_rec:
            st.markdown(f"""
            <div class="metric-card">
                <div class="metric-value">{metrics.get('recall', 0.0)*100:.1f}%</div>
                <div class="metric-label">Recall Score</div>
            </div>
            """, unsafe_allow_html=True)
        with col_f1:
            st.markdown(f"""
            <div class="metric-card">
                <div class="metric-value">{metrics.get('f1_score', 0.0)*100:.1f}%</div>
                <div class="metric-label">F1 Score</div>
            </div>
            """, unsafe_allow_html=True)
            
        st.divider()
        
        # Display Plots
        col_plot1, col_plot2 = st.columns(2)
        
        with col_plot1:
            st.subheader("Confusion Matrix")
            cm_path = PROJECT_ROOT / "models" / "confusion_matrix.png"
            if cm_path.exists():
                st.image(str(cm_path), caption="Confusion Matrix on Holdout Test Set", use_container_width=True)
            else:
                st.info("Confusion matrix image not found. Re-run training.")
                
        with col_plot2:
            st.subheader("Feature Importance")
            fi_path = PROJECT_ROOT / "models" / "feature_importance.png"
            if fi_path.exists():
                st.image(str(fi_path), caption="Feature Importance for Match Winner Prediction", use_container_width=True)
            else:
                st.info("Feature importance plot not found. This is normal for Logistic Regression models.")
                
        st.divider()
        
        # Display Candidate Comparison
        st.subheader("🤖 Model Candidates Comparison")
        st.write("Comparison of CV accuracies and Holdout Test scores across tuned models during training:")
        
        all_results = model_artifact.get("all_results", {})
        if all_results:
            comparison_rows = []
            for name, details in all_results.items():
                m_details = details["test_metrics"]
                comparison_rows.append({
                    "Model Algorithm": name.replace("_", " ").title(),
                    "CV Avg Accuracy": f"{details['cv_accuracy']*100:.2f}%",
                    "Holdout Accuracy": f"{m_details['accuracy']*100:.2f}%",
                    "Precision": f"{m_details['precision']*100:.2f}%",
                    "Recall": f"{m_details['recall']*100:.2f}%",
                    "F1 Score": f"{m_details['f1_score']*100:.2f}%",
                })
            st.table(pd.DataFrame(comparison_rows))

# ----------------- TAB 3: INSIGHTS & EXPLORER -----------------
with tab_insights:
    st.subheader("🗃️ Historical Match Data Explorer")
    
    if raw_data is None:
        st.warning("⚠️ Raw dataset 'data/raw_matches.csv' not found. Please place your raw CSV file in the data folder.")
    else:
        st.write(f"Showing the raw matches dataset containing {len(raw_data)} match records:")
        
        # Sidebar/Filter options inside the tab
        col_filter1, col_filter2 = st.columns(2)
        with col_filter1:
            filter_team = st.selectbox("Filter matches by Team Involvement:", ["All Teams"] + list(teams_list))
        with col_filter2:
            filter_season = st.selectbox("Filter matches by Season:", ["All Seasons"] + sorted(list(raw_data["season"].unique())))
            
        display_df = raw_data.copy()
        if filter_team != "All Teams":
            display_df = display_df[
                (display_df["team1"].str.strip() == filter_team) | 
                (display_df["team2"].str.strip() == filter_team)
            ]
        if filter_season != "All Seasons":
            display_df = display_df[display_df["season"] == filter_season]
            
        st.dataframe(display_df.head(100), use_container_width=True)
        st.caption(f"Filtered results: {len(display_df)} matches displayed.")
        
        st.divider()
        
        # Extra Insights
        st.subheader("💡 Dataset Statistical Insights")
        col_stat1, col_stat2 = st.columns(2)
        
        with col_stat1:
            st.markdown("#### Toss Winner Advantage")
            raw_data["toss_winner_won"] = raw_data["toss_winner"] == raw_data["winner"]
            toss_stats = raw_data["toss_winner_won"].value_counts(normalize=True) * 100
            
            st.write(f"In **{toss_stats.get(True, 0.0):.1f}%** of all matches, the team that **won the toss** also **won the match**.")
            
            # Simple bar chart of toss decision impact
            decision_counts = raw_data["toss_decision"].str.strip().str.lower().value_counts()
            st.bar_chart(decision_counts)
            st.caption("Distribution of Toss Decisions (Bat vs Field)")
            
        with col_stat2:
            st.markdown("#### Top 10 Most Visited Venues")
            top_v = raw_data["venue"].value_counts().head(10)
            st.bar_chart(top_v)
            st.caption("Venue occurrences in the raw match dataset")
