import streamlit as st
import requests
import pandas as pd

# ---------------------------------------------------------
# 1. Configuration & Cloud Connection
# ---------------------------------------------------------
# The permanent Render URL hosting your Docker container
API_URL = "https://el-matador-ai.onrender.com/predict"

st.set_page_config(page_title="UFC MLOps Predictor", layout="centered")
st.title("🥊 UFC Interactive Matchup Predictor")
st.markdown("Select two fighters from historical data. The app automatically calculates their physical differentials and queries the live XGBoost cloud model.")

# ---------------------------------------------------------
# 2. High-Performance Data Loading
# ---------------------------------------------------------
@st.cache_data
def load_data():
    # Ensure your cleaned dataset path matches your folder structure
    # Update this path if your CSV is named differently or in a different folder
    df = pd.read_csv(r"C:\Users\PRUTHVI\OneDrive\Desktop\EL Matador AI\data\ufc_clean.csv") 
    return df

try:
    df = load_data()
except Exception as e:
    st.error(f"Error loading dataset: {e}. Ensure your CSV file path is correct.")
    st.stop()

# Extract unique fighter names for the dropdowns
red_fighters = df['R_fighter'].dropna().unique()
blue_fighters = df['B_fighter'].dropna().unique()
all_fighters = sorted(list(set(list(red_fighters) + list(blue_fighters))))

# ---------------------------------------------------------
# 3. The User Interface (Dropdowns)
# ---------------------------------------------------------
st.subheader("Select Matchup")
col1, col2 = st.columns(2)

with col1:
    fighter_a = st.selectbox("🔴 Red Corner Fighter", all_fighters)

with col2:
    fighter_b = st.selectbox("🔵 Blue Corner Fighter", all_fighters)

# Prevent selecting the same fighter twice
if fighter_a == fighter_b:
    st.warning("Please select two different fighters for a valid matchup.")
else:
    # ---------------------------------------------------------
    # 4. The Execution Trigger & Extraction Engine
    # ---------------------------------------------------------
    if st.button("Simulate Matchup"):
        
        # Pull the most recent historical record for both fighters
        f_a_data = df[(df['R_fighter'] == fighter_a) | (df['B_fighter'] == fighter_a)].tail(1)
        f_b_data = df[(df['R_fighter'] == fighter_b) | (df['B_fighter'] == fighter_b)].tail(1)
        
        if f_a_data.empty or f_b_data.empty:
            st.error("Could not find sufficient historical records for one or both fighters.")
        else:
            with st.spinner("Calculating aligned physiological & combat differentials..."):
                try:
                    # Helper function to safely calculate differences even if data is missing
                    def get_dif(col_base):
                        r_val = f_a_data.get(f'R_{col_base}', pd.Series([0])).values[0]
                        b_val = f_b_data.get(f'B_{col_base}', pd.Series([0])).values[0]
                        if pd.isna(r_val): r_val = 0
                        if pd.isna(b_val): b_val = 0
                        return float(r_val - b_val)

                    # Assemble the EXACT 19 features your XGBoost model expects
                    payload_stats = {
                        "title_bout": 0,             
                        "no_of_rounds": 3,           
                        "lose_streak_dif": get_dif('current_lose_streak'),
                        "win_streak_dif": get_dif('current_win_streak'),
                        "longest_win_streak_dif": get_dif('longest_win_streak'),
                        "win_dif": get_dif('wins'),
                        "loss_dif": get_dif('losses'),
                        "total_round_dif": get_dif('total_rounds_fought'),
                        "total_title_bout_dif": get_dif('total_title_bouts'),
                        "ko_dif": get_dif('win_by_KO/TKO'),
                        "sub_dif": get_dif('win_by_Submission'),
                        "height_dif": get_dif('Height_cms'), 
                        "reach_dif": get_dif('reach'),
                        "age_dif": get_dif('age'),
                        "sig_str_dif": get_dif('avg_SIG_STR_landed'),
                        "avg_sub_att_dif": get_dif('avg_SUB_ATT'),
                        "avg_td_dif": get_dif('avg_TD_landed'),
                        
                        # Post-fight Leakage Features (Dummy Values to prevent model crash)
                        "finish_round": 3,           
                        "total_fight_time_secs": 900 
                    }

                    payload = {
                        "stats": payload_stats
                    }

                    # --- ENGINEERING DEBUGGER ---
                    # Keep this so you can verify the payload structure if anything breaks
                    with st.expander("🐛 View Engineering Payload (JSON)"):
                        st.json(payload)
                    # --------------------------------

                    # ---------------------------------------------------------
                    # 5. Network Bridge & Results Display
                    # ---------------------------------------------------------
                    response = requests.post(API_URL, json=payload)
                    
                    if response.status_code == 200:
                        data = response.json()
                        red_prob = data.get('red_corner_win_probability', 0)
                        blue_prob = data.get('blue_corner_win_probability', 0)

                        st.success("Prediction Generated Successfully via Cloud MLOps Pipeline!")

                        # Visual Output Presentation
                        res_col1, res_col2 = st.columns(2)
                        with res_col1:
                            st.metric(label=f"🔴 {fighter_a} Win Probability", value=f"{red_prob}%")
                        with res_col2:
                            st.metric(label=f"🔵 {fighter_b} Win Probability", value=f"{blue_prob}%")
                        
                        st.progress(int(red_prob))

                    else:
                        st.error(f"API Error {response.status_code}: {response.text}")
                        
                except Exception as calc_error:
                    st.error(f"Error computing differentials from dataset: {calc_error}")