"""
Streamlit UI for Student-Performance Predictor
----------------------------------------------
• Collects feature inputs in an intuitive form
• Calls the PredictPipeline to get the math-score prediction
• Shows the result with a progress-bar style visual
    (green = good, yellow = average, red = low)

Author: <your-name>
Date  : <today>
"""

import pandas as pd
import streamlit as st

# ──────────────────────────────────────────────────────────────────────────────
# INTERNAL IMPORTS – point to the modules you already created
# ──────────────────────────────────────────────────────────────────────────────
from student_performance_indicator.pipeline.prediction_pipeline import (
    CustomData,
    PredictPipeline,
)

# ──────────────────────────────────────────────────────────────────────────────
# PAGE CONFIG – favicon, title, and wide layout
# ──────────────────────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="Student Performance – Math Score Predictor",
    page_icon="📊",
    layout="wide",
)

# ──────────────────────────────────────────────────────────────────────────────
# SIDEBAR – project info & quick links
# ──────────────────────────────────────────────────────────────────────────────
with st.sidebar:
    st.title("📚 Student Performance")
    st.markdown(
        """
        **Model:** LinearRegression
        **Data:** Kaggle – https://www.kaggle.com/datasets/spscientist/students-performance-in-exams?datasetId=74977
        **Author:** Gabriel Adebayo  
        ---
        [GitHub Repo](https://github.com/iyan-coder) | 
        [LinkedIn](https://www.linkedin.com/in/gabriel-adebayo-2a0ba2281/)
        """
    )

# ──────────────────────────────────────────────────────────────────────────────
# MAIN HEADER
# ──────────────────────────────────────────────────────────────────────────────
st.title("🎯  Predict a Student's Math Score")

# ──────────────────────────────────────────────────────────────────────────────
# 1️⃣ INPUT FORM  – use two columns for better UX
# ──────────────────────────────────────────────────────────────────────────────
with st.form(key="prediction_form"):
    col1, col2 = st.columns(2)

    with col1:
        gender = st.selectbox("Gender ♂️♀️", ("male", "female"))
        race_ethnicity = st.selectbox(
            "Race / Ethnicity",
            (
                "group A",
                "group B",
                "group C",
                "group D",
                "group E",
            ),
        )
        lunch = st.selectbox(
            "Lunch Type",
            ("standard", "free/reduced"),
        )
        reading_score = st.number_input(
            "Reading Score (0-100)", min_value=0, max_value=100, value=70
        )

    with col2:
        parental_level_of_education = st.selectbox(
            "Parental Level of Education",
            (
                "some high school",
                "high school",
                "some college",
                "associate's degree",
                "bachelor's degree",
                "master's degree",
            ),
        )
        test_preparation_course = st.selectbox(
            "Test Preparation Course",
            ("none", "completed"),
        )
        writing_score = st.number_input(
            "Writing Score (0-100)", min_value=0, max_value=100, value=70
        )

    # Submit button (inside the form → triggers rerun)
    submit_btn = st.form_submit_button(label="🔮 Predict Math Score")

# ──────────────────────────────────────────────────────────────────────────────
# 2️⃣ RUN PREDICTION after user clicks the button
# ──────────────────────────────────────────────────────────────────────────────
if submit_btn:
    # 2.1 Pack user inputs into CustomData → DataFrame
    user_data = CustomData(
        gender=gender,
        race_ethnicity=race_ethnicity,
        parental_level_of_education=parental_level_of_education,
        lunch=lunch,
        test_preparation_course=test_preparation_course,
        reading_score=reading_score,
        writing_score=writing_score,
    )
    input_df: pd.DataFrame = user_data.get_data_as_data_frame()

    # 2.2 Feed into the prediction pipeline
    predictor = PredictPipeline()
    math_pred = predictor.predict(input_df)[0]  # scalar value

    # 2.3 Nicely display the result
    st.subheader("📈 Predicted Math Score")
    st.metric(label="Score (0-100)", value=f"{math_pred:.1f}")

    # 2.4 Simple color-coded progress bar
    pct = math_pred / 100.0
    bar_color = "#e74c3c" if pct < 0.5 else "#f1c40f" if pct < 0.75 else "#2ecc71"

    st.markdown(
        f"""
        <div style="height:25px; background:#ddd; border-radius:4px;">
            <div style="width:{pct*100}%; background:{bar_color};
                        height:100%; border-radius:4px;"></div>
        </div>
        """,
        unsafe_allow_html=True,
    )

    st.info("Remember: This is a model prediction, not an absolute truth.")

# ──────────────────────────────────────────────────────────────────────────────
# FOOTER
# ──────────────────────────────────────────────────────────────────────────────
st.caption("© 2025 Gabriel Adebayo. All rights reserved.")
