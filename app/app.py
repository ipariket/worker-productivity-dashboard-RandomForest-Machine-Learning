# ============================================================
#  Worker Productivity Dashboard — Phase 4 Complete Build
#  Features: 4A 4B 4C 4D 4E 4F 4G 4I 4L
# ============================================================

import streamlit as st
import pandas as pd
import numpy as np
import joblib
import os
import io
import plotly.graph_objects as go

# ── Page config ──────────────────────────────────────────────
st.set_page_config(
    page_title="Productivity Dashboard",
    page_icon="",
    layout="wide"
)

# ── Paths ────────────────────────────────────────────────────
_HERE      = os.path.dirname(os.path.abspath(__file__))
MODEL_DIR  = os.path.join(_HERE, '..', 'model')
DATA_DIR   = os.path.join(_HERE, '..', 'data', 'cleaned')

# ── Load artifacts ───────────────────────────────────────────
@st.cache_resource
def load_artifacts():
    m    = joblib.load(os.path.join(MODEL_DIR, 'productivity_model.pkl'))
    fc   = joblib.load(os.path.join(MODEL_DIR, 'feature_columns.pkl'))
    imp  = joblib.load(os.path.join(MODEL_DIR, 'feature_importances.pkl'))
    df_r = pd.read_csv(os.path.join(DATA_DIR, 'garment_clean.csv'))
    return m, fc, imp, df_r

model, feature_cols, importances, df_ref = load_artifacts()

# ── Constants ────────────────────────────────────────────────
DEPT_MAP        = {"Finishing": "finishing ", "Sewing": "sweing",
                   "finishing ": "finishing ", "sweing": "sweing"}
QUARTER_OPTIONS = ["Quarter1","Quarter2","Quarter3","Quarter4","Quarter5"]
DEPT_OPTIONS    = ["Finishing", "Sewing"]
DAY_OPTIONS     = ["Monday","Tuesday","Wednesday","Thursday","Saturday","Sunday"]

REQUIRED_BATCH_COLS = [
    'targeted_productivity','smv','wip','over_time','incentive',
    'idle_time','idle_men','no_of_style_change','no_of_workers',
    'quarter','department','day'
]

# ── Helper: build feature vector ─────────────────────────────
def build_feature_vector(raw):
    """
    raw: dict with keys matching REQUIRED_BATCH_COLS.
    Returns a single-row DataFrame aligned to feature_cols.
    """
    row = {col: 0 for col in feature_cols}

    row['targeted_productivity'] = raw['targeted_productivity']
    row['smv']                   = raw['smv']
    row['wip']                   = raw['wip']
    row['over_time']             = raw['over_time']
    row['incentive']             = raw['incentive']
    row['idle_time']             = raw['idle_time']
    row['idle_men']              = raw['idle_men']
    row['no_of_style_change']    = raw['no_of_style_change']
    row['no_of_workers']         = raw['no_of_workers']
    row['output_per_worker']     = raw['targeted_productivity'] * raw['no_of_workers']
    row['overtime_flag']         = 1 if raw['over_time'] > 0 else 0

    dept_raw = DEPT_MAP.get(str(raw['department']), str(raw['department']))
    for key, col in [('quarter', f"quarter_{raw['quarter']}"),
                     ('department', f"department_{dept_raw}"),
                     ('day', f"day_{raw['day']}")]:
        if col in row:
            row[col] = 1

    return pd.DataFrame([row])[feature_cols]

# ── Helper: predict with confidence (4I) ─────────────────────
def predict_with_confidence(input_df):
    """Returns (mean, std, lower, upper) using all RF tree predictions."""
    tree_preds = np.array([t.predict(input_df)[0] for t in model.estimators_])
    mean  = float(np.clip(np.mean(tree_preds), 0, 1))
    std   = float(np.std(tree_preds))
    lower = float(np.clip(mean - std, 0, 1))
    upper = float(np.clip(mean + std, 0, 1))
    return mean, std, lower, upper

# ── Helper: gauge chart (4L) ─────────────────────────────────
def make_gauge(value, threshold, key):
    pct = round(value * 100, 1)
    thr = round(threshold * 100, 1)
    fig = go.Figure(go.Indicator(
        mode="gauge+number",
        value=pct,
        number={'suffix': '%', 'font': {'size': 38, 'color': '#1E2761'}},
        gauge={
            'axis': {'range': [0, 100], 'ticksuffix': '%',
                     'tickfont': {'size': 11}},
            'bar': {'color': '#2196F3', 'thickness': 0.25},
            'steps': [
                {'range': [0,          thr * 0.85], 'color': '#FEE2E2'},
                {'range': [thr * 0.85, thr],        'color': '#FEF9C3'},
                {'range': [thr,        100],         'color': '#DCFCE7'},
            ],
            'threshold': {
                'line': {'color': '#EF4444', 'width': 3},
                'thickness': 0.8,
                'value': thr
            }
        }
    ))
    fig.update_layout(
        height=240,
        margin=dict(l=20, r=20, t=30, b=10),
        paper_bgcolor='rgba(0,0,0,0)'
    )
    st.plotly_chart(fig, use_container_width=True, key=key)

# ── Helper: importance chart (4B) ────────────────────────────
def make_importance_chart(key):
    top    = importances.head(10).sort_values(ascending=True)
    labels = [c.replace('_', ' ').replace('quarter ', 'Quarter: ')
               .replace('department ', 'Dept: ')
               .replace('day ', 'Day: ').title()
              for c in top.index]
    fig = go.Figure(go.Bar(
        x=top.values, y=labels,
        orientation='h',
        marker_color='#2196F3', marker_line_width=0
    ))
    fig.update_layout(
        height=300,
        margin=dict(l=10, r=20, t=10, b=30),
        xaxis_title='Importance Score',
        paper_bgcolor='rgba(0,0,0,0)',
        plot_bgcolor='rgba(0,0,0,0)',
        xaxis={'gridcolor': '#E2E8F0'},
        yaxis={'gridcolor': 'rgba(0,0,0,0)'},
        font={'family': 'Calibri', 'size': 11}
    )
    st.plotly_chart(fig, use_container_width=True, key=key)

# ── Helper: composite score (4D) ─────────────────────────────
def composite_score(prediction, over_time, incentive,
                    w_model, w_overtime, w_incentive):
    total = w_model + w_overtime + w_incentive or 1
    nw_m, nw_ot, nw_in = w_model/total, w_overtime/total, w_incentive/total
    ot_score  = 1.0 - min(over_time / 15240, 1.0)
    inc_score = min(incentive / 125, 1.0)
    return float(np.clip(
        nw_m * prediction + nw_ot * ot_score + nw_in * inc_score, 0, 1
    ))

# ── Helper: render full result block ─────────────────────────
def render_result(pred, lower, upper, raw, threshold,
                  w_model, w_overtime, w_incentive,
                  cost_per_idle_hour, value_per_unit, kp):
    """
    kp = key prefix string (unique per call site to avoid duplicate widget keys)
    """
    gap    = raw['targeted_productivity'] - pred
    delta  = f"{-gap:+.1%} vs target"
    total_w = w_model + w_overtime + w_incentive or 1

    if pred >= threshold:
        status, colour = "On Target",   "normal"
    elif pred >= threshold * 0.85:
        status, colour = " Near Target", "off"
    else:
        status, colour = "At Risk",      "inverse"

    comp = composite_score(pred, raw['over_time'], raw['incentive'],
                           w_model, w_overtime, w_incentive)

    # ── Gauge + core metrics ──────────────────────────────────
    g_col, m_col = st.columns([1, 1.3])
    with g_col:
        make_gauge(pred, threshold, key=f"{kp}_gauge")
    with m_col:
        st.metric("Predicted Productivity", f"{pred:.1%}",
                  delta=delta, delta_color=colour)
        st.metric("Status", status)
        st.metric("Confidence Band", f"{lower:.1%} – {upper:.1%}",
                  help=f"±1 std dev across {len(model.estimators_)} trees. "
                       "Narrower = higher model certainty.")

    st.divider()

    # ── Composite score (4D) ─────────────────────────────────
    with st.expander(" Composite Manager Score", expanded=True):
        cs_col, info_col = st.columns([1, 2])
        cs_col.metric("Composite Score", f"{comp:.1%}")
        info_col.caption(
            f"Weighted blend based on your sidebar settings:\n\n"
            f"- Model Prediction: **{w_model/total_w:.0%}**\n"
            f"- Overtime Minimization: **{w_overtime/total_w:.0%}**\n"
            f"- Incentive Efficiency: **{w_incentive/total_w:.0%}**\n\n"
            "_Adjust weights in the sidebar to match your priorities._"
        )

    # ── ROI card (4F) ────────────────────────────────────────
    with st.expander("Financial Impact Estimate", expanded=True):
        if gap > 0:
            labour_loss = gap * raw['no_of_workers'] * cost_per_idle_hour
            units_lost  = gap * raw['no_of_workers'] * value_per_unit * 8
            r1, r2 = st.columns(2)
            r1.metric("Estimated Labour Cost Gap", f"${labour_loss:,.2f}",
                      delta=f"-{gap:.1%} shortfall", delta_color="inverse")
            r2.metric("Est. Units Lost", f"{units_lost:,.0f}",
                      delta=f"@ ${value_per_unit}/unit", delta_color="inverse")
            st.caption(
                "_Based on constants set in the sidebar. "
                "Assumes an 8-hour shift. Adjust to match your factory._"
            )
        else:
            st.success("On or above target — no projected financial loss for this shift.")

    # ── Why breakdown (4B) ───────────────────────────────────
    with st.expander("Why This Score? — Feature Importance Breakdown",
                     expanded=False):
        st.caption(
            "Global feature importance from the trained Random Forest "
            "(200 trees). Shows which inputs drive predictions most overall."
        )
        make_importance_chart(key=f"{kp}_imp")

        st.markdown("**Your inputs vs. dataset averages:**")
        NUM_DISPLAY = ['targeted_productivity','over_time','incentive',
                       'idle_time','idle_men','no_of_style_change',
                       'no_of_workers','wip','smv']
        avgs = df_ref[NUM_DISPLAY].mean()
        rows = []
        for col in NUM_DISPLAY:
            uval = raw.get(col, 0)
            avg  = avgs[col]
            flag = ("↑ Above avg" if uval > avg * 1.05
                    else "↓ Below avg" if uval < avg * 0.95
                    else "≈ Average")
            rows.append({
                'Feature': col.replace('_', ' ').title(),
                'Your Input': round(float(uval), 2),
                'Dataset Avg': round(float(avg), 2),
                'vs Average': flag
            })
        st.dataframe(pd.DataFrame(rows), use_container_width=True,
                     hide_index=True)

# ── Helper: shared input panel ───────────────────────────────
def input_panel(key_suffix=""):
    """Returns a raw_inputs dict from slider/selectbox widgets."""
    col1, col2 = st.columns(2)
    with col1:
        tp  = st.slider("Targeted Productivity",   0.0,  1.0, 0.75, 0.01,
                        key=f"tp{key_suffix}")
        ot  = st.slider("Overtime (minutes)",        0, 15240,    0, 60,
                        key=f"ot{key_suffix}")
        inc = st.slider("Incentive (BDT)",            0,   125,    0,  5,
                        key=f"inc{key_suffix}")
        it  = st.slider("Idle Time (minutes)",       0.0, 300.0,  0.0, 5.0,
                        key=f"it{key_suffix}")
        im  = st.slider("Idle Workers",               0,    45,    0,  1,
                        key=f"im{key_suffix}")
    with col2:
        nw  = st.slider("Team Size",                 1.0, 90.0,  30.0, 1.0,
                        key=f"nw{key_suffix}")
        wip = st.slider("Work In Progress (WIP)",   0.0, 1252.0, 500.0, 10.0,
                        key=f"wip{key_suffix}")
        smv = st.slider("SMV (Task Complexity)",    0.0,  55.0,  20.0,  0.5,
                        key=f"smv{key_suffix}")
        sc  = st.slider("Style Changes",              0,    10,    0,   1,
                        key=f"sc{key_suffix}")

    c3, c4, c5 = st.columns(3)
    with c3:
        quarter = st.selectbox("Quarter", QUARTER_OPTIONS,
                               key=f"q{key_suffix}")
    with c4:
        dept_display = st.selectbox("Department", DEPT_OPTIONS,
                                    key=f"dept{key_suffix}")
    with c5:
        day = st.selectbox("Day of Week", DAY_OPTIONS,
                           key=f"day{key_suffix}")

    return {
        'targeted_productivity': tp, 'smv': smv, 'wip': wip,
        'over_time': ot, 'incentive': inc, 'idle_time': it,
        'idle_men': im, 'no_of_style_change': sc, 'no_of_workers': nw,
        'quarter': quarter, 'department': dept_display, 'day': day
    }

# ════════════════════════════════════════════════════════════
# SIDEBAR
# ════════════════════════════════════════════════════════════
with st.sidebar:
    st.title("Dashboard Controls")
    st.divider()

    # 4C — Dynamic threshold
    st.subheader("Evaluation Threshold")
    threshold = st.slider(
        "Minimum Acceptable Productivity",
        min_value=0.50, max_value=0.90, value=0.70, step=0.01,
        help="Shifts predicted below this are flagged At Risk."
    )
    st.caption(f"Current threshold: **{threshold:.0%}**")
    st.divider()

    # 4D — Composite weights
    st.subheader("Composite Score Weights")
    st.caption("Auto-normalized to 100%.")
    w_model     = st.slider("Model Prediction",       0, 100, 60)
    w_overtime  = st.slider("Overtime Minimization",  0, 100, 25)
    w_incentive = st.slider("Incentive Efficiency",   0, 100, 15)
    _tw = w_model + w_overtime + w_incentive
    if _tw > 0:
        st.caption(
            f"Model **{w_model/_tw:.0%}** · "
            f"Overtime **{w_overtime/_tw:.0%}** · "
            f"Incentive **{w_incentive/_tw:.0%}**"
        )
    else:
        st.warning("At least one weight must be > 0")
    st.divider()

    # 4F — ROI constants
    st.subheader("ROI Constants")
    cost_per_idle_hour = st.number_input(
        "Cost per Worker/Hour ($)", min_value=1, max_value=500, value=25
    )
    value_per_unit = st.number_input(
        "Value per Finished Unit ($)", min_value=1, max_value=100, value=5
    )
    st.divider()
    st.caption(
        "_Model: Random Forest · 200 trees · "
        f"R² = 0.52 on held-out test set_"
    )

    st.divider()
    with st.expander("About This Tool"):
        st.markdown("""
**Worker Productivity Dashboard**
Built for CSC-492 Senior Design (Summer 2026)
*Brent Uyguangco & Pariket Koirala*
California State University, Dominguez Hills

**Purpose:** A decision-support tool for garment factory supervisors to predict shift productivity before work begins.

**Model:** Random Forest (200 trees) trained on 1,197 real-world shift records. R² = 0.52.

** Ethical Guardrails:**
- **Team-level only** — operates on shift/team aggregates, not individual worker surveillance
-  **Decision-support, not decision-maker** — a human supervisor always makes the final call
- **Manager-adjustable thresholds** — context matters; rigid algorithms don't account for human factors
- **Model transparency** — R² score and confidence band are displayed so users know the model's limitations

** Acknowledged Limitations:**
- R² of 0.52 means ~48% of productivity variance is unexplained
- Dataset is from a single factory and may not generalize to all cultural contexts
- Scores must never be used for hiring, firing, or disciplinary actions
        """)

    with st.expander("Data Source"):
        st.markdown("""
**Dataset:** Productivity Prediction of Garment Employees

**Source:** Kaggle
[View Dataset](https://www.kaggle.com/datasets/ishadss/productivity-prediction-of-garment-employees)

**Original Authors:** Imran et al. (2021)

The dataset contains 1,197 real-world factory shift records collected manually from garment factories and validated by industry experts. It includes 15 features such as overtime, team size, incentives, idle time, and work in progress.
        """)

    with st.expander("Privacy Policy"):
        st.markdown("""
**This tool is built strictly for academic purposes.**

- This dashboard was developed as a senior capstone project (CSC-492) at California State University, Dominguez Hills
- No real employee data is collected, stored, or transmitted
- Any data entered into this tool is processed locally and is not saved
- The dataset used for training is publicly available on Kaggle and contains no personally identifiable information
- This tool is not intended for commercial use or deployment in real workforce management systems

*For questions, contact the project authors.*
        """)

# ════════════════════════════════════════════════════════════
# MAIN
# ════════════════════════════════════════════════════════════
st.title("Worker Productivity Dashboard")
st.caption(
    "Predict shift productivity for garment factory teams. "
    "Adjust threshold and weights in the sidebar."
)

tab1, tab2, tab3 = st.tabs([
    "Single Shift Evaluation",
    "What-If Simulator",
    "Batch Shift Analysis"
])

# ════════════════════════════════════════════════════════════
# TAB 1 — Single Shift Evaluation (4A 4B 4C 4D 4F 4I 4L)
# ════════════════════════════════════════════════════════════
with tab1:
    st.subheader(" Shift Parameters")
    raw = input_panel(key_suffix="_t1")

    st.divider()
    if st.button("⚡ Evaluate Productivity", type="primary",
                 use_container_width=True, key="eval_t1"):
        input_df = build_feature_vector(raw)
        pred, std, lower, upper = predict_with_confidence(input_df)

        st.divider()
        st.subheader("Prediction Results")
        render_result(
            pred, lower, upper, raw, threshold,
            w_model, w_overtime, w_incentive,
            cost_per_idle_hour, value_per_unit,
            kp="t1"
        )
    else:
        st.info(
            "Set the shift parameters above and click "
            "**Evaluate Productivity** to get a prediction."
        )

# ════════════════════════════════════════════════════════════
# TAB 2 — What-If Simulator (4G)
# ════════════════════════════════════════════════════════════
with tab2:
    st.subheader("What-If Simulator")
    st.caption(
        "Compare two shift configurations side by side. "
        "Predictions update live as you adjust sliders."
    )

    scen_a, scen_b = st.columns(2)

    with scen_a:
        st.markdown("### 🔵 Scenario A — Current Shift")
        raw_a = input_panel(key_suffix="_a")

    with scen_b:
        st.markdown("### 🟢 Scenario B — Adjusted Shift")
        raw_b = input_panel(key_suffix="_b")

    st.divider()
    st.subheader(" Side-by-Side Comparison")

    df_a = build_feature_vector(raw_a)
    df_b = build_feature_vector(raw_b)
    pred_a, _, low_a, up_a = predict_with_confidence(df_a)
    pred_b, _, low_b, up_b = predict_with_confidence(df_b)

    delta_ab = pred_b - pred_a
    delta_str = f"{delta_ab:+.1%}"

    # Summary metrics row
    m1, m2, m3 = st.columns(3)
    m1.metric("Scenario A", f"{pred_a:.1%}",
              "On Target" if pred_a >= threshold else "At Risk")
    m2.metric("Scenario B", f"{pred_b:.1%}",
              "On Target" if pred_b >= threshold else "At Risk")
    m3.metric("B vs A (Delta)", delta_str,
              delta_color="normal" if delta_ab >= 0 else "inverse")

    # Gauges side by side
    ga_col, gb_col = st.columns(2)
    with ga_col:
        st.caption("Scenario A")
        make_gauge(pred_a, threshold, key="gauge_a")
    with gb_col:
        st.caption("Scenario B")
        make_gauge(pred_b, threshold, key="gauge_b")

    # Confidence bands
    st.caption(
        f"Confidence bands — "
        f"A: {low_a:.1%}–{up_a:.1%}  ·  "
        f"B: {low_b:.1%}–{up_b:.1%}"
    )

    # Input diff table
    with st.expander(" Input Differences (A vs B)", expanded=False):
        NUM_COLS_DIFF = [
            'targeted_productivity','over_time','incentive',
            'idle_time','idle_men','no_of_style_change',
            'no_of_workers','wip','smv'
        ]
        diff_rows = []
        for col in NUM_COLS_DIFF:
            va, vb = raw_a.get(col, 0), raw_b.get(col, 0)
            diff = vb - va
            diff_rows.append({
                'Feature': col.replace('_', ' ').title(),
                'Scenario A': round(float(va), 2),
                'Scenario B': round(float(vb), 2),
                'Change': f"{diff:+.2f}" if diff != 0 else "—"
            })
        diff_df = pd.DataFrame(diff_rows)
        st.dataframe(diff_df, use_container_width=True, hide_index=True)

# ════════════════════════════════════════════════════════════
# TAB 3 — Batch Shift Analysis (4E)
# ════════════════════════════════════════════════════════════
with tab3:
    st.subheader("Batch Shift Analysis")
    st.caption(
        "Upload a CSV of shift schedules to predict productivity "
        "for an entire roster at once."
    )

    # Template download
    template_df = pd.DataFrame([{
        'targeted_productivity': 0.75, 'smv': 20.0, 'wip': 500.0,
        'over_time': 0, 'incentive': 60, 'idle_time': 0.0,
        'idle_men': 0, 'no_of_style_change': 0, 'no_of_workers': 30.0,
        'quarter': 'Quarter1', 'department': 'Finishing', 'day': 'Monday'
    }])
    template_csv = template_df.to_csv(index=False).encode('utf-8')
    st.download_button(
        " Download CSV Template",
        data=template_csv,
        file_name="shift_template.csv",
        mime="text/csv",
        help="Download a correctly formatted template to fill in."
    )

    st.divider()
    uploaded = st.file_uploader(
        "Upload Shift Schedule CSV",
        type="csv",
        help="Must contain the same columns as the template above."
    )

    if uploaded is not None:
        try:
            batch_df = pd.read_csv(uploaded)
        except Exception as e:
            st.error(f"Could not read file: {e}")
            st.stop()

        # Column validation
        missing_cols = [c for c in REQUIRED_BATCH_COLS
                        if c not in batch_df.columns]
        if missing_cols:
            st.error(
                f" Missing required columns: **{', '.join(missing_cols)}**\n\n"
                "Download the template above to check the expected format."
            )
            st.stop()

        # Run predictions
        with st.spinner(f"Running predictions on {len(batch_df)} rows..."):
            preds = []
            errors = []
            for idx, row in batch_df.iterrows():
                try:
                    raw_row = {col: row[col] for col in REQUIRED_BATCH_COLS}
                    fv   = build_feature_vector(raw_row)
                    pred = float(np.clip(model.predict(fv)[0], 0, 1))
                    preds.append(pred)
                except Exception as e:
                    preds.append(None)
                    errors.append(f"Row {idx+2}: {e}")

        batch_df['Predicted_Productivity'] = preds
        batch_df['Status'] = batch_df['Predicted_Productivity'].apply(
            lambda p: "On Target" if p is not None and p >= threshold
            else ("At Risk" if p is not None else "Error")
        )

        # Summary
        n_total   = len(batch_df)
        n_atrisk  = (batch_df['Status'] == "At Risk").sum()
        n_ontgt   = (batch_df['Status'] == "On Target").sum()
        n_err     = (batch_df['Status'] == "Error").sum()

        s1, s2, s3 = st.columns(3)
        s1.metric("Total Shifts",  n_total)
        s2.metric("On Target",   n_ontgt)
        s3.metric("At Risk",     n_atrisk,
                  delta=f"{n_atrisk/n_total:.0%} of shifts",
                  delta_color="inverse" if n_atrisk > 0 else "normal")

        if errors:
            with st.expander(f" {len(errors)} row(s) had errors"):
                for e in errors:
                    st.caption(e)

        st.divider()
        st.subheader("Results")

        # Gauge grid — 3 per row
        cols_per_row = 3
        rows = [batch_df.iloc[i:i+cols_per_row] for i in range(0, len(batch_df), cols_per_row)]
        for row_df in rows:
            cols = st.columns(cols_per_row)
            for col_idx, (_, shift_row) in enumerate(row_df.iterrows()):
                pred = shift_row.get('Predicted_Productivity')
                status = shift_row.get('Status', '')
                shift_label = f"Shift {_ + 1}"
                if pred is not None:
                    gauge_color = "#22c55e" if pred >= threshold else "#ef4444"
                    fig_g = go.Figure(go.Indicator(
                        mode="gauge+number",
                        value=round(pred * 100, 1),
                        number={'suffix': '%', 'font': {'size': 28, 'color': '#ffffff'}},
                        title={'text': f"{shift_label}<br><span style='font-size:12px'>{status}</span>", 'font': {'size': 13, 'color': '#ffffff'}},
                        gauge={
                            'axis': {'range': [0, 100], 'tickcolor': '#ffffff', 'tickfont': {'color': '#ffffff'}},
                            'bar': {'color': gauge_color, 'thickness': 0.25},
                            'steps': [
                                {'range': [0, threshold * 100 * 0.85], 'color': '#FEE2E2'},
                                {'range': [threshold * 100 * 0.85, threshold * 100], 'color': '#FEF9C3'},
                                {'range': [threshold * 100, 100], 'color': '#DCFCE7'},
                            ],
                            'threshold': {'line': {'color': '#EF4444', 'width': 3}, 'thickness': 0.75, 'value': threshold * 100}
                        }
                    ))
                    fig_g.update_layout(
                        height=220,
                        margin=dict(t=60, b=10, l=10, r=10),
                        paper_bgcolor='rgba(0,0,0,0)',
                        font={'color': '#ffffff'}
                    )
                    cols[col_idx].plotly_chart(fig_g, use_container_width=True)
                else:
                    cols[col_idx].warning(f"{shift_label}: Error")

        # Highlight at-risk rows
        def highlight_risk(row):
            if row.get('Status') == "At Risk":
                return ["background-color: #FEE2E2; color: #000000"] * len(row)
            elif row.get('Status') == "On Target":
                return ["background-color: #DCFCE7; color: #000000"] * len(row)
            return [''] * len(row)

        display_cols = REQUIRED_BATCH_COLS + ['Predicted_Productivity','Status']
        st.dataframe(
            batch_df[display_cols].style.apply(highlight_risk, axis=1),
            use_container_width=True
        )

        # Download results
        result_csv = batch_df[display_cols].to_csv(index=False).encode('utf-8')
        st.download_button(
            "Download Results CSV",
            data=result_csv,
            file_name="shift_predictions.csv",
            mime="text/csv"
        )
