import streamlit as st
import pandas as pd
from src.automl import run_automl
from src.serializer import save_model, load_model

st.title("🚀 AutoML System")

uploaded_file = st.file_uploader("Upload CSV", type=["csv"])

if uploaded_file:
    df = pd.read_csv(uploaded_file)
    st.dataframe(df.head())

    target = st.selectbox("Select Target Column", df.columns)

    if st.button("Run AutoML"):

        best_model, leaderboard, problem_type = run_automl(df, target)

        st.write("### Problem Type:", problem_type)
        st.write("### Leaderboard")
        st.dataframe(leaderboard)

        save_model(best_model)

        st.success("Best model saved!")
        
