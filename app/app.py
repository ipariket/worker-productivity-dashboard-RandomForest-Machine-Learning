# ============================================================
#  Worker Productivity Dashboard — Phase 5
#  Added: Session-based login + Heatmaps tab
# ============================================================

import streamlit as st
import pandas as pd
import numpy as np
import joblib
import os
import io
import plotly.graph_objects as go
import plotly.express as px

# ── Page config ──────────────────────────────────────────────
st.set_page_config(
    page_title="Productivity Dashboard",
    page_icon="",
    layout="wide"
)

# ════════════════════════════════════════════════════════════
# AUTH — reads from st.secrets
# ════════════════════════════════════════════════════════════

def get_users():
    return st.secrets.get("users", {})

def login_page():
    st.markdown("""
    <style>
    .login-title {
        text-align: center;
        font-size: 1.5rem;
        font-weight: 700;
        color: #F1F5F9;
        margin-bottom: 0.25rem;
    }
    .login-sub {
        text-align: center;
        color: #94A3B8;
        font-size: 0.9rem;
        margin-bottom: 1.5rem;
    }
    </style>
    """, unsafe_allow_html=True)

    col_l, col_c, col_r = st.columns([1, 2, 1])
    with col_c:
        st.markdown('<div class="login-title">Worker Productivity Dashboard</div>', unsafe_allow_html=True)
        st.markdown('<div class="login-sub">Supervisor Portal — CSC-492 Senior Design</div>', unsafe_allow_html=True)
        st.divider()

        users = get_users()
        ROLES = ["Floor Supervisor", "Senior Supervisor", "Admin"]

        username = st.text_input("Username", placeholder="Enter your username", key="login_user")
        password = st.text_input("Password", type="password", placeholder="Enter your password", key="login_pass")
        role_selected = st.selectbox("Role", ROLES, key="login_role")

        if st.button("Sign In", type="primary", use_container_width=True):
            uname = username.strip()
            user = users.get(uname)
            if user and user["password"] == password and user["role"] == role_selected:
                st.session_state["authenticated"] = True
                st.session_state["username"]      = uname
                st.session_state["display_name"]  = user["name"]
                st.session_state["role"]          = user["role"]
                st.rerun()
            else:
                st.error("Invalid username, password, or role.")

# Restore session from query params on reload
params = st.query_params
if "authenticated" not in st.session_state:
    if params.get("auth") == "1":
        users = get_users()
        uname = params.get("u", "")
        user = users.get(uname)
        if user:
            st.session_state["authenticated"] = True
            st.session_state["username"]      = uname
            st.session_state["display_name"]  = user["name"]
            st.session_state["role"]          = user["role"]
        else:
            st.session_state["authenticated"] = False
    else:
        st.session_state["authenticated"] = False

if not st.session_state["authenticated"]:
    login_page()
    st.stop()

# Persist login in query params
st.query_params["auth"] = "1"
st.query_params["u"] = st.session_state["username"]

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

# ── Helper: predict with confidence ──────────────────────────
def predict_with_confidence(input_df):
    tree_preds = np.array([t.predict(input_df)[0] for t in model.estimators_])
    mean  = float(np.clip(np.mean(tree_preds), 0, 1))
    std   = float(np.std(tree_preds))
    lower = float(np.clip(mean - std, 0, 1))
    upper = float(np.clip(mean + std, 0, 1))
    return mean, std, lower, upper

# ── Helper: gauge chart ───────────────────────────────────────
def make_gauge(value, threshold, key):
    pct = round(value * 100, 1)
    thr = round(threshold * 100, 1)
    fig = go.Figure(go.Indicator(
        mode="gauge+number",
        value=pct,
        number={'suffix': '%', 'font': {'size': 38, 'color': '#1E2761'}},
        gauge={
            'axis': {'range': [0, 100], 'ticksuffix': '%', 'tickfont': {'size': 11}},
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

# ── Helper: importance chart ─────────────────────────────────
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

# ── Helper: composite score ───────────────────────────────────
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
    gap    = raw['targeted_productivity'] - pred
    delta  = f"{-gap:+.1%} vs target"
    total_w = w_model + w_overtime + w_incentive or 1

    if pred >= threshold:
        status, colour = "On Target",    "normal"
    elif pred >= threshold * 0.85:
        status, colour = "Near Target",  "off"
    else:
        status, colour = "At Risk",      "inverse"

    comp = composite_score(pred, raw['over_time'], raw['incentive'],
                           w_model, w_overtime, w_incentive)

    g_col, m_col = st.columns([1, 1.3])
    with g_col:
        make_gauge(pred, threshold, key=f"{kp}_gauge")
    with m_col:
        st.metric("Predicted Productivity", f"{pred:.1%}",
                  delta=delta, delta_color=colour)
        st.metric("Status", status)
        st.metric("Confidence Band", f"{lower:.1%} – {upper:.1%}",
                  help=f"±1 std dev across {len(model.estimators_)} trees.")

    st.divider()

    with st.expander("Composite Manager Score", expanded=True):
        cs_col, info_col = st.columns([1, 2])
        cs_col.metric("Composite Score", f"{comp:.1%}")
        info_col.caption(
            f"Weighted blend based on your sidebar settings:\n\n"
            f"- Model Prediction: **{w_model/total_w:.0%}**\n"
            f"- Overtime Minimization: **{w_overtime/total_w:.0%}**\n"
            f"- Incentive Efficiency: **{w_incentive/total_w:.0%}**\n\n"
            "_Adjust weights in the sidebar to match your priorities._"
        )

    with st.expander("Financial Impact Estimate", expanded=True):
        if gap > 0:
            labour_loss = gap * raw['no_of_workers'] * cost_per_idle_hour
            units_lost  = gap * raw['no_of_workers'] * value_per_unit * 8
            r1, r2 = st.columns(2)
            r1.metric("Estimated Labour Cost Gap", f"${labour_loss:,.2f}",
                      delta=f"-{gap:.1%} shortfall", delta_color="inverse")
            r2.metric("Est. Units Lost", f"{units_lost:,.0f}",
                      delta=f"@ ${value_per_unit}/unit", delta_color="inverse")
            st.caption("_Based on constants set in the sidebar. Assumes an 8-hour shift._")
        else:
            st.success("On or above target — no projected financial loss for this shift.")

    with st.expander("Prediction Heatmap — Sensitivity Analysis", expanded=True):
        st.caption("How predicted productivity changes as Day and Incentive vary, holding all other inputs fixed.")
        days = ["Monday","Tuesday","Wednesday","Thursday","Saturday","Sunday"]
        incentive_range = [0, 25, 50, 75, 100, 125]
        heat_data = []
        for d in days:
            row_vals = []
            for inc in incentive_range:
                test_raw = dict(raw)
                test_raw["day"] = d
                test_raw["incentive"] = inc
                fv = build_feature_vector(test_raw)
                p = float(np.clip(model.predict(fv)[0], 0, 1))
                row_vals.append(round(p, 3))
            heat_data.append(row_vals)
        heat_df = pd.DataFrame(heat_data, index=days, columns=[f"BDT {i}" for i in incentive_range])
        fig_heat = px.imshow(
            heat_df,
            labels=dict(x="Incentive (BDT)", y="Day", color="Predicted Productivity"),
            color_continuous_scale="Viridis",
            zmin=0, zmax=1,
            text_auto=".2f",
            aspect="auto"
        )
        fig_heat.update_layout(
            height=350,
            margin=dict(l=10,r=10,t=30,b=10),
            paper_bgcolor="rgba(0,0,0,0)",
            font=dict(size=12)
        )
        st.plotly_chart(fig_heat, use_container_width=True, key=f"{kp}_heatmap")
        st.caption("_All other inputs (team size, overtime, WIP, etc.) are held at your entered values._")

    with st.expander("Why This Score? — Feature Importance Breakdown", expanded=False):
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
        st.dataframe(pd.DataFrame(rows), use_container_width=True, hide_index=True)

# ── Helper: shared input panel ───────────────────────────────
def input_panel(key_suffix=""):
    col1, col2 = st.columns(2)
    with col1:
        tp  = st.slider("Targeted Productivity",   0.0,  1.0, 0.75, 0.01, key=f"tp{key_suffix}")
        ot  = st.slider("Overtime (minutes)",        0, 15240,    0, 60,   key=f"ot{key_suffix}")
        inc = st.slider("Incentive (BDT)",            0,   125,    0,  5,   key=f"inc{key_suffix}")
        it  = st.slider("Idle Time (minutes)",       0.0, 300.0,  0.0, 5.0, key=f"it{key_suffix}")
        im  = st.slider("Idle Workers",               0,    45,    0,  1,   key=f"im{key_suffix}")
    with col2:
        nw  = st.slider("Team Size",                 1.0, 90.0,  30.0, 1.0, key=f"nw{key_suffix}")
        wip = st.slider("Work In Progress (WIP)",   0.0, 1252.0, 500.0, 10.0, key=f"wip{key_suffix}")
        smv = st.slider("SMV (Task Complexity)",    0.0,  55.0,  20.0,  0.5, key=f"smv{key_suffix}")
        sc  = st.slider("Style Changes",              0,    10,    0,   1,   key=f"sc{key_suffix}")

    c3, c4, c5 = st.columns(3)
    with c3:
        quarter = st.selectbox("Quarter", QUARTER_OPTIONS, key=f"q{key_suffix}")
    with c4:
        dept_display = st.selectbox("Department", DEPT_OPTIONS, key=f"dept{key_suffix}")
    with c5:
        day = st.selectbox("Day of Week", DAY_OPTIONS, key=f"day{key_suffix}")

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
    st.markdown(f"**{st.session_state['display_name']}**")
    st.caption(st.session_state['role'])
    if st.button("Sign Out", use_container_width=True):
        for key in ["authenticated","username","display_name","role"]:
            st.session_state.pop(key, None)
        st.query_params.clear()
        st.rerun()

    st.divider()
    st.title("Dashboard Controls")
    st.divider()

    st.subheader("Evaluation Threshold")
    threshold = st.slider(
        "Minimum Acceptable Productivity",
        min_value=0.50, max_value=0.90, value=0.70, step=0.01,
        help="Shifts predicted below this are flagged At Risk."
    )
    st.caption(f"Current threshold: **{threshold:.0%}**")
    st.divider()

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

    st.subheader("ROI Constants")
    cost_per_idle_hour = st.number_input(
        "Cost per Worker/Hour ($)", min_value=1, max_value=500, value=25
    )
    value_per_unit = st.number_input(
        "Value per Finished Unit ($)", min_value=1, max_value=100, value=5
    )
    st.divider()
    st.caption("_Model: Random Forest · 200 trees · R² = 0.52_")

    st.divider()
    with st.expander("About This Tool"):
        st.markdown("""
**Worker Productivity Dashboard**
Built for CSC-492 Senior Design (Summer 2026)
*Brent Uyguangco & Pariket Koirala*
California State University, Dominguez Hills

**Model:** Random Forest (200 trees) trained on 1,197 real-world shift records. R² = 0.52.

**Ethical Guardrails:**
- Team-level only — no individual worker surveillance
- Decision-support, not decision-maker
- Scores must never be used for disciplinary actions
        """)

    with st.expander("Data Source"):
        st.markdown("""
**Dataset:** Productivity Prediction of Garment Employees
**Source:** [Kaggle](https://www.kaggle.com/datasets/ishadss/productivity-prediction-of-garment-employees)
**Authors:** Imran et al. (2021) · 1,197 shift records
        """)

    with st.expander("Privacy Policy"):
        st.markdown("""
- No real employee data is collected, stored, or transmitted
- Data entered is processed locally and not saved
- For academic use only — not for commercial workforce systems
        """)

# ════════════════════════════════════════════════════════════
# MAIN
# ════════════════════════════════════════════════════════════
st.title("Worker Productivity Dashboard")
st.caption(
    f"Welcome, **{st.session_state['display_name']}** · "
    "Predict shift productivity for garment factory teams. "
    "Adjust threshold and weights in the sidebar."
)

tab1, tab2, tab3, tab4 = st.tabs([
    "Single Shift Evaluation",
    "What-If Simulator",
    "Batch Shift Analysis",
    "Heatmaps & Analytics"
])

# ════════════════════════════════════════════════════════════
# TAB 1 — Single Shift Evaluation
# ════════════════════════════════════════════════════════════
with tab1:
    st.subheader("Shift Parameters")
    raw = input_panel(key_suffix="_t1")

    st.divider()
    if st.button("Evaluate Productivity", type="primary",
                 use_container_width=True, key="eval_t1"):
        input_df = build_feature_vector(raw)
        pred, std, lower, upper = predict_with_confidence(input_df)
        st.divider()
        st.subheader("Prediction Results")
        render_result(
            pred, lower, upper, raw, threshold,
            w_model, w_overtime, w_incentive,
            cost_per_idle_hour, value_per_unit, kp="t1"
        )
    else:
        st.info(
            "Set the shift parameters above and click "
            "**Evaluate Productivity** to get a prediction."
        )

# ════════════════════════════════════════════════════════════
# TAB 2 — What-If Simulator
# ════════════════════════════════════════════════════════════
with tab2:
    st.subheader("What-If Simulator")
    st.caption(
        "Compare two shift configurations side by side. "
        "Predictions update live as you adjust sliders."
    )

    scen_a, scen_b = st.columns(2)
    with scen_a:
        st.markdown("### Scenario A — Current Shift")
        raw_a = input_panel(key_suffix="_a")
    with scen_b:
        st.markdown("### Scenario B — Adjusted Shift")
        raw_b = input_panel(key_suffix="_b")

    st.divider()
    st.subheader("Side-by-Side Comparison")

    df_a = build_feature_vector(raw_a)
    df_b = build_feature_vector(raw_b)
    pred_a, _, low_a, up_a = predict_with_confidence(df_a)
    pred_b, _, low_b, up_b = predict_with_confidence(df_b)
    delta_ab = pred_b - pred_a

    m1, m2, m3 = st.columns(3)
    m1.metric("Scenario A", f"{pred_a:.1%}", "On Target" if pred_a >= threshold else "At Risk")
    m2.metric("Scenario B", f"{pred_b:.1%}", "On Target" if pred_b >= threshold else "At Risk")
    m3.metric("B vs A (Delta)", f"{delta_ab:+.1%}",
              delta_color="normal" if delta_ab >= 0 else "inverse")

    ga_col, gb_col = st.columns(2)
    with ga_col:
        st.caption("Scenario A")
        make_gauge(pred_a, threshold, key="gauge_a")
    with gb_col:
        st.caption("Scenario B")
        make_gauge(pred_b, threshold, key="gauge_b")

    st.caption(
        f"Confidence bands — "
        f"A: {low_a:.1%}–{up_a:.1%}  ·  "
        f"B: {low_b:.1%}–{up_b:.1%}"
    )

    with st.expander("Input Differences (A vs B)", expanded=False):
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
        st.dataframe(pd.DataFrame(diff_rows), use_container_width=True, hide_index=True)

# ════════════════════════════════════════════════════════════
# TAB 3 — Batch Shift Analysis
# ════════════════════════════════════════════════════════════
with tab3:
    st.subheader("Batch Shift Analysis")
    st.caption(
        "Upload a CSV of shift schedules to predict productivity "
        "for an entire roster at once."
    )

    template_df = pd.DataFrame([{
        'targeted_productivity': 0.75, 'smv': 20.0, 'wip': 500.0,
        'over_time': 0, 'incentive': 60, 'idle_time': 0.0,
        'idle_men': 0, 'no_of_style_change': 0, 'no_of_workers': 30.0,
        'quarter': 'Quarter1', 'department': 'Finishing', 'day': 'Monday'
    }])
    template_csv = template_df.to_csv(index=False).encode('utf-8')
    st.download_button(
        "Download CSV Template",
        data=template_csv,
        file_name="shift_template.csv",
        mime="text/csv"
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

        missing_cols = [c for c in REQUIRED_BATCH_COLS if c not in batch_df.columns]
        if missing_cols:
            st.error(f"Missing required columns: **{', '.join(missing_cols)}**")
            st.stop()

        with st.spinner(f"Running predictions on {len(batch_df)} rows..."):
            preds, errors = [], []
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

        n_total  = len(batch_df)
        n_atrisk = (batch_df['Status'] == "At Risk").sum()
        n_ontgt  = (batch_df['Status'] == "On Target").sum()

        s1, s2, s3 = st.columns(3)
        s1.metric("Total Shifts", n_total)
        s2.metric("On Target",  n_ontgt)
        s3.metric("At Risk",    n_atrisk,
                  delta=f"{n_atrisk/n_total:.0%} of shifts",
                  delta_color="inverse" if n_atrisk > 0 else "normal")

        if errors:
            with st.expander(f"{len(errors)} row(s) had errors"):
                for e in errors:
                    st.caption(e)

        st.divider()
        st.subheader("Results")

        cols_per_row = 3
        rows = [batch_df.iloc[i:i+cols_per_row] for i in range(0, len(batch_df), cols_per_row)]
        for row_df in rows:
            cols = st.columns(cols_per_row)
            for col_idx, (orig_idx, shift_row) in enumerate(row_df.iterrows()):
                pred = shift_row.get('Predicted_Productivity')
                status = shift_row.get('Status', '')
                shift_label = f"Shift {orig_idx + 1}"
                if pred is not None:
                    gauge_color = "#22c55e" if pred >= threshold else "#ef4444"
                    fig_g = go.Figure(go.Indicator(
                        mode="gauge+number",
                        value=round(pred * 100, 1),
                        number={'suffix': '%', 'font': {'size': 28, 'color': '#ffffff'}},
                        title={'text': f"{shift_label}<br><span style='font-size:12px'>{status}</span>",
                               'font': {'size': 13, 'color': '#ffffff'}},
                        gauge={
                            'axis': {'range': [0, 100], 'tickcolor': '#ffffff',
                                     'tickfont': {'color': '#ffffff'}},
                            'bar': {'color': gauge_color, 'thickness': 0.25},
                            'steps': [
                                {'range': [0, threshold * 100 * 0.85], 'color': '#FEE2E2'},
                                {'range': [threshold * 100 * 0.85, threshold * 100], 'color': '#FEF9C3'},
                                {'range': [threshold * 100, 100], 'color': '#DCFCE7'},
                            ],
                            'threshold': {'line': {'color': '#EF4444', 'width': 3},
                                          'thickness': 0.75, 'value': threshold * 100}
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

        st.divider()
        st.subheader("Shift Details")
        for idx, (orig_idx, shift_row) in enumerate(batch_df.iterrows()):
            pred = shift_row.get('Predicted_Productivity')
            status = shift_row.get('Status', '')
            if pred is None:
                continue
            with st.expander(f"Shift {idx + 1} — {round(pred*100,1)}% — {status}"):
                d1, d2 = st.columns(2)
                with d1:
                    st.markdown("**Why This Score?**")
                    st.caption("Global feature importance from the trained Random Forest.")
                    fig_imp = go.Figure(go.Bar(
                        x=importances[:10],
                        y=feature_cols[:10],
                        orientation='h',
                        marker_color='#2196F3',
                        marker_line_width=0
                    ))
                    fig_imp.update_layout(
                        height=250,
                        margin=dict(t=10, b=10, l=10, r=10),
                        paper_bgcolor='rgba(0,0,0,0)',
                        plot_bgcolor='rgba(0,0,0,0)',
                        xaxis={'gridcolor': '#E2E8F0'},
                        yaxis={'gridcolor': 'rgba(0,0,0,0)'},
                        font={'color': '#ffffff', 'size': 10}
                    )
                    st.plotly_chart(fig_imp, use_container_width=True, key=f"imp_{idx}")
                with d2:
                    st.markdown("**What-If Scenario**")
                    st.caption("Adjust incentive and overtime to see how the prediction changes.")
                    wi_incentive = st.slider(
                        f"Incentive (BDT) — Shift {idx+1}", 0, 200,
                        int(shift_row.get('incentive', 0)), key=f"wi_inc_{idx}"
                    )
                    wi_overtime = st.slider(
                        f"Overtime (min) — Shift {idx+1}", 0, 14880,
                        int(shift_row.get('over_time', 0)), key=f"wi_ot_{idx}"
                    )
                    wi_row = {col: shift_row[col] for col in REQUIRED_BATCH_COLS}
                    wi_row['incentive']  = wi_incentive
                    wi_row['over_time'] = wi_overtime
                    try:
                        wi_fv   = build_feature_vector(wi_row)
                        wi_pred = float(np.clip(model.predict(wi_fv)[0], 0, 1))
                        wi_delta = wi_pred - pred
                        st.metric(
                            "Adjusted Prediction", f"{wi_pred:.1%}",
                            delta=f"{wi_delta:+.1%} vs original",
                            delta_color="normal" if wi_delta >= 0 else "inverse"
                        )
                        st.markdown(f"**Status:** {'On Target' if wi_pred >= threshold else 'At Risk'}")
                    except Exception as e:
                        st.warning(f"Could not compute: {e}")

                comp_b = composite_score(pred, shift_row.get('over_time', 0),
                                         shift_row.get('incentive', 0),
                                         w_model, w_overtime, w_incentive)
                st.metric("Composite Manager Score", f"{comp_b:.1%}")

                st.markdown("**Prediction Heatmap — Day vs Incentive**")
                st.caption("How productivity changes across days and incentive levels for this shift.")
                days = ["Monday","Tuesday","Wednesday","Thursday","Saturday","Sunday"]
                incentive_range = [0, 25, 50, 75, 100, 125]
                heat_data = []
                for d in days:
                    row_vals = []
                    for inc in incentive_range:
                        test_raw = {col: shift_row[col] for col in REQUIRED_BATCH_COLS}
                        test_raw["day"] = d
                        test_raw["incentive"] = inc
                        fv = build_feature_vector(test_raw)
                        p = float(np.clip(model.predict(fv)[0], 0, 1))
                        row_vals.append(round(p, 3))
                    heat_data.append(row_vals)
                heat_df = pd.DataFrame(heat_data, index=days, columns=[f"BDT {i}" for i in incentive_range])
                fig_heat = px.imshow(
                    heat_df,
                    labels=dict(x="Incentive (BDT)", y="Day", color="Predicted Productivity"),
                    color_continuous_scale="Viridis",
                    zmin=0, zmax=1,
                    text_auto=".2f",
                    aspect="auto"
                )
                fig_heat.update_layout(
                    height=350,
                    margin=dict(l=10,r=10,t=30,b=10),
                    paper_bgcolor="rgba(0,0,0,0)",
                    font=dict(size=12)
                )
                st.plotly_chart(fig_heat, use_container_width=True, key=f"batch_heatmap_{idx}")

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
        result_csv = batch_df[display_cols].to_csv(index=False).encode('utf-8')
        st.download_button(
            "Download Results CSV",
            data=result_csv,
            file_name="shift_predictions.csv",
            mime="text/csv"
        )

# ════════════════════════════════════════════════════════════
# TAB 4 — Heatmaps & Analytics
# ════════════════════════════════════════════════════════════
with tab4:
    st.subheader("Heatmaps & Analytics")
    st.caption(
        "Visual analysis of the garment dataset — productivity patterns, "
        "correlations, and shift-level insights."
    )

    df = df_ref.copy()

    # Reconstruct department from one-hot columns
    df['department'] = 'Finishing'
    df.loc[df['department_sweing'] == 1, 'department'] = 'Sewing'

    # Reconstruct day from one-hot columns (Monday is baseline)
    day_cols = {'day_Tuesday': 'Tuesday', 'day_Wednesday': 'Wednesday', 'day_Thursday': 'Thursday', 'day_Saturday': 'Saturday', 'day_Sunday': 'Sunday'}
    df['day'] = 'Monday'
    for col, label in day_cols.items():
        if col in df.columns:
            df.loc[df[col] == 1, 'day'] = label

    # Reconstruct quarter (Quarter1 is baseline)
    quarter_cols = {'quarter_Quarter2': 'Quarter2', 'quarter_Quarter3': 'Quarter3', 'quarter_Quarter4': 'Quarter4', 'quarter_Quarter5': 'Quarter5'}
    df['quarter'] = 'Quarter1'
    for col, label in quarter_cols.items():
        if col in df.columns:
            df.loc[df[col] == 1, 'quarter'] = label

    prod_col = 'actual_productivity' if 'actual_productivity' in df.columns else 'targeted_productivity'

    st.markdown("### Productivity by Day & Department")
    st.caption("Average actual productivity per day of the week, split by department.")
    day_order = ["Monday","Tuesday","Wednesday","Thursday","Saturday","Sunday"]

    if 'day' in df.columns and 'department' in df.columns:
        pivot_day_dept = (
            df.groupby(['day', 'department'])[prod_col]
            .mean().reset_index()
            .pivot(index='day', columns='department', values=prod_col)
            .reindex(day_order)
        )
        fig_dd = px.imshow(
            pivot_day_dept,
            labels=dict(x="Department", y="Day", color="Avg Productivity"),
            color_continuous_scale="Viridis", aspect="auto", text_auto=".2f"
        )
        fig_dd.update_layout(height=380, margin=dict(l=10,r=10,t=30,b=10),
                             paper_bgcolor='rgba(0,0,0,0)', font=dict(size=12))
        st.plotly_chart(fig_dd, use_container_width=True, key="hm_day_dept")
    else:
        st.info("Day or department column not found in dataset.")

    st.divider()

    st.markdown("### Feature Correlation Matrix")
    st.caption("Pearson correlation between numeric features. Blue = positive, Red = negative.")
    CORR_COLS = [prod_col,'targeted_productivity','smv','wip',
                 'over_time','incentive','idle_time','idle_men',
                 'no_of_style_change','no_of_workers']
    available_corr = [c for c in CORR_COLS if c in df.columns]
    corr_matrix = df[available_corr].corr()
    pretty_labels = [c.replace('_',' ').title() for c in available_corr]
    fig_corr = px.imshow(
        corr_matrix, x=pretty_labels, y=pretty_labels,
        color_continuous_scale="RdBu", zmin=-1, zmax=1,
        text_auto=".2f", aspect="auto"
    )
    fig_corr.update_layout(height=500, margin=dict(l=10,r=10,t=30,b=10),
                           paper_bgcolor='rgba(0,0,0,0)', font=dict(size=11))
    st.plotly_chart(fig_corr, use_container_width=True, key="hm_corr")

    st.divider()

    st.markdown("### Productivity by Quarter & Department")
    st.caption("Average productivity across production quarters per department.")
    if 'quarter' in df.columns and 'department' in df.columns:
        quarter_order = ["Quarter1","Quarter2","Quarter3","Quarter4","Quarter5"]
        pivot_q_dept = (
            df.groupby(['quarter','department'])[prod_col]
            .mean().reset_index()
            .pivot(index='quarter', columns='department', values=prod_col)
            .reindex(quarter_order)
        )
        fig_qd = px.imshow(
            pivot_q_dept,
            labels=dict(x="Department", y="Quarter", color="Avg Productivity"),
            color_continuous_scale="Greens", aspect="auto", text_auto=".2f"
        )
        fig_qd.update_layout(height=380, margin=dict(l=10,r=10,t=30,b=10),
                             paper_bgcolor='rgba(0,0,0,0)', font=dict(size=12))
        st.plotly_chart(fig_qd, use_container_width=True, key="hm_quarter_dept")
    else:
        st.info("Quarter or department column not found in dataset.")

    st.divider()

    st.markdown("### Idle Time & Overtime Patterns by Day")
    st.caption("Normalized averages — higher value = more idle time or overtime on that day.")
    if 'day' in df.columns and 'idle_time' in df.columns and 'over_time' in df.columns:
        idle_ot = (
            df.groupby('day')[['idle_time','over_time']]
            .mean().reindex(day_order).reset_index()
        )
        idle_ot_norm = idle_ot[['idle_time','over_time']].copy()
        for c in ['idle_time','over_time']:
            col_max = idle_ot_norm[c].max()
            idle_ot_norm[c] = idle_ot_norm[c] / col_max if col_max > 0 else 0
        idle_ot_norm.index = idle_ot['day']
        idle_ot_norm.columns = ['Idle Time (norm)','Overtime (norm)']
        fig_io = px.imshow(
            idle_ot_norm.T,
            labels=dict(x="Day", y="Metric", color="Normalized Value"),
            color_continuous_scale="Oranges", text_auto=".2f", aspect="auto"
        )
        fig_io.update_layout(height=260, margin=dict(l=10,r=10,t=30,b=10),
                             paper_bgcolor='rgba(0,0,0,0)', font=dict(size=12))
        st.plotly_chart(fig_io, use_container_width=True, key="hm_idle_ot")

        with st.expander("Raw averages by day"):
            idle_ot_display = idle_ot.copy()
            idle_ot_display.columns = ['Day','Avg Idle Time (min)','Avg Overtime (min)']
            idle_ot_display['Avg Idle Time (min)'] = idle_ot_display['Avg Idle Time (min)'].round(1)
            idle_ot_display['Avg Overtime (min)']  = idle_ot_display['Avg Overtime (min)'].round(0).astype(int)
            st.dataframe(idle_ot_display, use_container_width=True, hide_index=True)
    else:
        st.info("Idle time or overtime column not found in dataset.")
