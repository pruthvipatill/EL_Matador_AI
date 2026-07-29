import streamlit as st
import requests

API_URL = "https://el-matador-ai.onrender.com/predict"

st.set_page_config(page_title='UFC Predictor',layout='centered')
st.title('El-Matador-AI')
st.markdown('Enter fight differentials')

col1,col2 = st.columns(2)

with col1:
    age_diff = st.number_input('Age difference(Years)',value= 0.0)
    reach_diff = st.number_input('Reach difference(Inches)',value = 0.0)

with col2:
    strike_diff = st.number_input('Strike difference',value=0.0)
    td_diff = st.number_input('Takedown difference',value=0.0)

if st.button('Calculate Probability'):

    payload = {
        'stats':{
            'age_differential':age_diff,
            'reach_diff':reach_diff,
            'strike_landed_diff':strike_diff,
            'td_diff':td_diff
        }
    }

    with st.spinner('Accessing Render Container'):
        try:
            response = requests.post(API_URL,json=payload)

            if response.status_code == 200:
                data = response.json()
                red_prob = data.get('red_corner_win_probability',0)
                blue_prob = data.get('blue_corner_win_probability',0)

                st.success('Prediction generated Successfully')

                st.write(f'Red Corner Winning probability: {red_prob}%')
                st.write(f'Blue corner winning probaility: {blue_prob}%')

                st.progress(int(red_prob))
            else:
                st.error(f'API error {response.status_code} - {response.text}')

        except Exception as e:
            st.error(f'Failed to connect to MLOps pipeline. Error: {e}')