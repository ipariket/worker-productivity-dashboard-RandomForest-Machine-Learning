# ============================================================
#  Worker Productivity Dashboard — Phase 3 MVP Skeleton
#  Run with: streamlit run app/app.py
# ============================================================

import streamlit as st
import pandas as pd
import numpy as np
import joblib
import os

# ── Page config ───────────────────────────────────────────────
st.set_page_config(
    page_title="Productivity Dashboard",
    layout="centered"
)

# ── Load model artifacts ───────────────────────────────────────
MODEL_DIR = os.path.join(os.path.dirname(__file__), '..', 'model')

@st.cache_resource
def load_artifacts():
    model         = joblib.load(os.path.join(MODEL_DIR, 'productivity_model.pkl'))
    feature_cols  = joblib.load(os.path.join(MODEL_DIR, 'feature_columns.pkl'))
    importances   = joblib.load(os.path.join(MODEL_DIR, 'feature_importances.pkl'))
    return model, feature_cols, importances

model, feature_cols, importances = load_artifacts()

# ── Helper: build feature vector from raw UI inputs ───────────
def build_feature_vector(
    targeted_productivity, smv, wip, over_time,
    incentive, idle_time, idle_men, no_of_style_change,
    no_of_workers, quarter, department, day
):
    """
    Converts raw user inputs (including dropdown strings) into
    the exact feature vector the model was trained on.
    One-hot columns default to 0; the selected category is set to 1.
    """
    row = {col: 0 for col in feature_cols}

    # Numerical features
    row['targeted_productivity'] = targeted_productivity
    row['smv']                   = smv
    row['wip']                   = wip
    row['over_time']             = over_time
    row['incentive']             = incentive
    row['idle_time']             = idle_time
    row['idle_men']              = idle_men
    row['no_of_style_change']    = no_of_style_change
    row['no_of_workers']         = no_of_workers

    # Engineered features
    row['output_per_worker'] = targeted_productivity * no_of_workers
    row['overtime_flag']     = 1 if over_time > 0 else 0

    # One-hot: quarter (reference = Quarter1)
    quarter_col = f'quarter_{quarter}'
    if quarter_col in row:
        row[quarter_col] = 1

    # One-hot: department (reference = finishing without trailing space)
    dept_col = f'department_{department}'
    if dept_col in row:
        row[dept_col] = 1

    # One-hot: day (reference = Monday)
    day_col = f'day_{day}'
    if day_col in row:
        row[day_col] = 1

    return pd.DataFrame([row])[feature_cols]


# ════════════════════════════════════════════════════════════════
#  UI
# ════════════════════════════════════════════════════════════════

st.title("Worker Productivity Dashboard")
st.caption("Predict shift productivity for garment factory teams.")
st.divider()

# ── Input panel ───────────────────────────────────────────────
st.subheader("Shift Parameters")

col1, col2 = st.columns(2)

with col1:
    targeted_productivity = st.slider(
        "Targeted Productivity",
        min_value=0.0, max_value=1.0, value=0.75, step=0.01,
        help="The productivity quota set for this shift (0 = 0%, 1 = 100%)"
    )
    over_time = st.slider(
        "Overtime (minutes)",
        min_value=0, max_value=15240, value=0, step=60,
        help="Total overtime worked this shift in minutes"
    )
    incentive = st.slider(
        "Incentive (BDT)",
        min_value=0, max_value=125, value=0, step=5,
        help="Financial incentive offered to the team"
    )
    idle_time = st.slider(
        "Idle Time (minutes)",
        min_value=0.0, max_value=300.0, value=0.0, step=5.0,
        help="Minutes lost due to line stoppages or supply delays"
    )
    idle_men = st.slider(
        "Idle Workers",
        min_value=0, max_value=45, value=0, step=1,
        help="Number of workers sitting idle this shift"
    )

with col2:
    no_of_workers = st.slider(
        "Team Size",
        min_value=1.0, max_value=90.0, value=30.0, step=1.0,
        help="Total number of workers on this team"
    )
    wip = st.slider(
        "Work In Progress (WIP)",
        min_value=0.0, max_value=1252.0, value=500.0, step=10.0,
        help="Number of unfinished items in the pipeline"
    )
    smv = st.slider(
        "SMV (Standard Minute Value)",
        min_value=0.0, max_value=55.0, value=20.0, step=0.5,
        help="Task complexity — higher SMV = harder task"
    )
    no_of_style_change = st.slider(
        "Style Changes",
        min_value=0, max_value=10, value=0, step=1,
        help="Number of garment style switches mid-shift"
    )

st.divider()
st.subheader("Shift Context")

col3, col4, col5 = st.columns(3)

with col3:
    quarter = st.selectbox(
        "Quarter",
        options=["Quarter1", "Quarter2", "Quarter3", "Quarter4", "Quarter5"],
        help="Which quarter of the month this shift falls in"
    )
with col4:
    department = st.selectbox(
        "Department",
        options=["finishing ", "sewing"],
        help="Production department"
    )
with col5:
    day = st.selectbox(
        "Day of Week",
        options=["Monday", "Tuesday", "Wednesday", "Thursday", "Saturday", "Sunday"],
        help="Day this shift runs (no Fridays in dataset)"
    )

st.divider()

# ── Evaluate button ───────────────────────────────────────────
if st.button("⚡ Evaluate Productivity", type="primary", use_container_width=True):

    input_df   = build_feature_vector(
        targeted_productivity, smv, wip, over_time,
        incentive, idle_time, idle_men, no_of_style_change,
        no_of_workers, quarter, department, day
    )
    prediction = float(model.predict(input_df)[0])
    prediction = np.clip(prediction, 0.0, 1.0)   # safety clip

    gap   = targeted_productivity - prediction
    delta = f"{gap:+.1%} vs target"

    # Colour-coded status
    if prediction >= targeted_productivity:
        status = "On Target"
        colour = "normal"
    elif prediction >= targeted_productivity * 0.85:
        status = "Near Target"
        colour = "off"
    else:
        status = "At Risk"
        colour = "inverse"

    st.divider()
    st.subheader("Prediction Result")

    m1, m2, m3 = st.columns(3)
    m1.metric("Predicted Productivity", f"{prediction:.1%}", delta=delta,
              delta_color=colour)
    m2.metric("Target",                 f"{targeted_productivity:.1%}")
    m3.metric("Status",                 status)

    st.caption(
        f"_Raw score: {prediction:.4f} | "
        f"Model: Random Forest (R² ≈ 0.52 on held-out test set)_"
    )

else:
    st.info("Set the shift parameters above and click **Evaluate Productivity** to get a prediction.")
