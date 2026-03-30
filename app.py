"""
app.py – Streamlit Web Dashboard
Adversarial Attacks in Healthcare AI: Vulnerabilities and Detection Mechanisms

Run with:
    streamlit run app.py
"""

import streamlit as st

# ── Page config (MUST be first Streamlit call) ────────────────────────────────
st.set_page_config(
    page_title="Adversarial Healthcare AI",
    page_icon="🧬",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ── Inject global CSS ─────────────────────────────────────────────────────────
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700;800&display=swap');

/* Root overrides */
html, body, [class*="css"] {
    font-family: 'Inter', sans-serif;
}

/* Dark sidebar */
[data-testid="stSidebar"] {
    background: linear-gradient(180deg, #0f1117 0%, #1a1d2e 100%);
    border-right: 1px solid #2d3561;
}
[data-testid="stSidebar"] * { color: #e2e8f0 !important; }
[data-testid="stSidebar"] .stRadio label { color: #94a3b8 !important; }
[data-testid="stSidebar"] .stRadio [data-baseweb="radio"] { color: #818cf8 !important; }

/* Main background */
.main .block-container {
    background: #0d1117;
    padding-top: 1.5rem;
}

/* Metric cards */
.metric-card {
    background: linear-gradient(135deg, #1e2235 0%, #252840 100%);
    border: 1px solid #2d3561;
    border-radius: 16px;
    padding: 1.4rem 1.6rem;
    text-align: center;
    transition: transform .2s ease, box-shadow .2s ease;
    cursor: default;
}
.metric-card:hover {
    transform: translateY(-4px);
    box-shadow: 0 8px 30px rgba(99,102,241,.35);
}
.metric-card .label {
    font-size: .75rem;
    font-weight: 600;
    letter-spacing: .08em;
    text-transform: uppercase;
    color: #94a3b8;
    margin-bottom: .35rem;
}
.metric-card .value {
    font-size: 2rem;
    font-weight: 800;
    background: linear-gradient(135deg, #818cf8, #38bdf8);
    -webkit-background-clip: text;
    -webkit-text-fill-color: transparent;
}
.metric-card .sub {
    font-size: .72rem;
    color: #64748b;
    margin-top: .2rem;
}

/* Section headings */
.section-title {
    font-size: 1.35rem;
    font-weight: 700;
    color: #e2e8f0;
    margin-bottom: 1rem;
    padding-bottom: .4rem;
    border-bottom: 2px solid #2d3561;
}

/* Tag badges */
.badge {
    display: inline-block;
    padding: .25rem .7rem;
    border-radius: 999px;
    font-size: .72rem;
    font-weight: 600;
    letter-spacing: .04em;
    margin: .15rem;
}
.badge-blue   { background: #1e3a5f; color: #60a5fa; border: 1px solid #2563eb55; }
.badge-green  { background: #14532d; color: #4ade80; border: 1px solid #16a34a55; }
.badge-red    { background: #450a0a; color: #f87171; border: 1px solid #dc262655; }
.badge-purple { background: #2e1065; color: #c084fc; border: 1px solid #7c3aed55; }
.badge-yellow { background: #422006; color: #fbbf24; border: 1px solid #d9770655; }

/* Result boxes */
.result-box {
    border-radius: 12px;
    padding: 1.1rem 1.4rem;
    margin: .6rem 0;
    border-left: 4px solid;
}
.result-safe    { background: #052e1655; border-color: #22c55e; color: #4ade80; }
.result-danger  { background: #450a0a55; border-color: #ef4444; color: #f87171; }
.result-warning { background: #42200655; border-color: #f59e0b; color: #fbbf24; }
.result-info    { background: #1e3a5f55; border-color: #3b82f6; color: #60a5fa; }

/* Sidebar logo area */
.sidebar-header {
    text-align: center;
    padding: 1rem 0 1.5rem;
    border-bottom: 1px solid #2d3561;
    margin-bottom: 1.2rem;
}
.sidebar-header h2 {
    font-size: 1.05rem;
    font-weight: 700;
    color: #818cf8 !important;
}
.sidebar-header p {
    font-size: .72rem;
    color: #64748b !important;
}

/* Pill tabs workaround */
.tab-container { display: flex; gap: .5rem; margin-bottom: 1.5rem; flex-wrap: wrap; }

/* Override Streamlit default dark theme conflicts */
.stSelectbox label, .stSlider label, .stFileUploader label {
    color: #94a3b8 !important;
    font-size: .85rem !important;
    font-weight: 500 !important;
}
div[data-baseweb="select"] { border-radius: 10px !important; }
.stButton > button {
    border-radius: 10px !important;
    font-weight: 600 !important;
    border: none !important;
    background: linear-gradient(135deg, #6366f1, #38bdf8) !important;
    color: white !important;
    padding: .5rem 1.4rem !important;
    transition: opacity .2s !important;
}
.stButton > button:hover { opacity: .85 !important; }
</style>
""", unsafe_allow_html=True)

# ── Sidebar navigation ────────────────────────────────────────────────────────
with st.sidebar:
    st.markdown("""
    <div class="sidebar-header">
        <div style="font-size:2.5rem">🧬</div>
        <h2>Adversarial Healthcare AI</h2>
        <p>HAM10000 · ResNet-18 · Adversarial Robustness</p>
    </div>
    """, unsafe_allow_html=True)

    page = st.radio(
        "Navigate",
        options=[
            "🏠  Dashboard",
            "🔬  Image Classifier",
            "⚔️  Attack Generator",
            "🛡️  Detection",
            "📊  Results & Report",
        ],
        label_visibility="collapsed",
    )
    st.markdown("---")
    st.markdown("""
    <div style="color:#475569;font-size:.72rem;line-height:1.6">
    <b style="color:#64748b">Pipeline Phases</b><br>
    Phase 1 · Baseline ResNet-18<br>
    Phase 2 · FGSM / PGD / C&W<br>
    Phase 3 · LID / Mahal / AE<br>
    Phase 4 · Adv Training + Denoise<br>
    Phase 5 · Metrics & Report<br><br>
    <span style="color:#334155">⚠️ Research use only<br>
    Seed = 42 · HAM10000 dataset</span>
    </div>
    """, unsafe_allow_html=True)

# ── Route to pages ────────────────────────────────────────────────────────────
if   "Dashboard"   in page:
    from pages import pg_dashboard;  pg_dashboard.show()
elif "Classifier"  in page:
    from pages import pg_classifier; pg_classifier.show()
elif "Attack"      in page:
    from pages import pg_attack;     pg_attack.show()
elif "Detection"   in page:
    from pages import pg_detect;     pg_detect.show()
elif "Results"     in page:
    from pages import pg_results;    pg_results.show()
