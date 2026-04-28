import streamlit as st
import pandas as pd
import shap
import matplotlib.pyplot as plt
from src.model.train_model import train_model
from src.bias.detect_bias import detect_bias
from src.bias.mitigate_bias import mitigate_bias
from src.visualization.plots import plot_bias, plot_comparison
from src.utils.report import generate_report
import os
import base64

# ------------------ PAGE CONFIG ------------------
st.set_page_config(page_title="Healthcare AI Bias Inspector", layout="wide")

# ------------------ BACKGROUND VIDEO ------------------
def set_bg_video(video_path):
    if os.path.exists(video_path):
        with open(video_path, 'rb') as f:
            data = f.read()
        video_base64 = base64.b64encode(data).decode()
        ext = video_path.split('.')[-1]
        video_html = f'''
            <style>
            #myVideo {{
                position: fixed;
                right: 0;
                bottom: 0;
                min-width: 100vw;
                min-height: 100vh;
                z-index: -9999;
                object-fit: cover;
                opacity: 0.70;
                filter: brightness(0.85) contrast(1.1) saturate(1.1);
                pointer-events: none;
            }}
            html, body, #root, #root > div, .stApp, [data-testid="stAppViewContainer"], [data-testid="stApp"] {{
                background: transparent !important;
                background-color: transparent !important;
            }}
            </style>
            <video autoplay muted loop playsinline id="myVideo">
                <source src="data:video/{ext};base64,{video_base64}" type="video/{ext}">
            </video>
        '''
        st.markdown(video_html, unsafe_allow_html=True)

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
mp4_path = os.path.join(BASE_DIR, "HEALTHCARE.mp4")
if os.path.exists(mp4_path):
    set_bg_video(mp4_path)

# ------------------ GLOBAL UI STYLE ------------------
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Outfit:wght@300;400;500;600;700;800&display=swap');

/* Global Font & Reset */
* {
    font-family: 'Outfit', sans-serif !important;
}

/* Background Overlay */
.stApp::before {
    content: "";
    position: fixed;
    top: 0;
    left: 0;
    width: 100%;
    height: 100%;
    background: rgba(255, 255, 255, 0.5);
    z-index: -1;
}

/* Scrollbar */
::-webkit-scrollbar {
    width: 8px;
}
::-webkit-scrollbar-track {
    background: transparent;
}
::-webkit-scrollbar-thumb {
    background: rgba(37, 99, 235, 0.2);
    border-radius: 10px;
}
::-webkit-scrollbar-thumb:hover {
    background: rgba(37, 99, 235, 0.4);
}

/* Transparent Containers */
.stApp, [data-testid="stApp"], [data-testid="stAppViewContainer"], [data-testid="stHeader"] {
    background: transparent !important;
}

/* Sidebar Styling */
[data-testid="stSidebar"] {
    background: rgba(255, 255, 255, 0.8) !important;
    backdrop-filter: blur(20px) !important;
    border-right: 1px solid rgba(255, 255, 255, 0.3) !important;
    box-shadow: 10px 0 30px rgba(0, 0, 0, 0.05) !important;
}

/* Main Content Container */
.main .block-container {
    padding: 2rem 5rem !important;
    max-width: 1200px !important;
}

/* Typography */
h1 {
    color: #0F172A !important;
    font-size: 3.2rem !important;
    font-weight: 800 !important;
    letter-spacing: -0.04em !important;
    margin-bottom: 0.5rem !important;
    text-align: center !important;
}

h2 {
    color: #1E293B !important;
    font-size: 1.8rem !important;
    font-weight: 700 !important;
    letter-spacing: -0.02em !important;
}

.hero-subtitle {
    color: #475569 !important;
    font-size: 1.25rem !important;
    font-weight: 400 !important;
    text-align: center !important;
    margin-bottom: 3.5rem !important;
}

/* Glass Card */
.glass-card {
    background: rgba(255, 255, 255, 0.8) !important;
    backdrop-filter: blur(12px) !important;
    border: 1px solid rgba(255, 255, 255, 0.6) !important;
    border-radius: 20px !important;
    padding: 2rem !important;
    box-shadow: 0 10px 30px rgba(0, 0, 0, 0.05) !important;
    transition: all 0.4s cubic-bezier(0.4, 0, 0.2, 1) !important;
    margin-bottom: 2rem !important;
    width: 100% !important;
    display: block !important;
    overflow: hidden !important;
}

/* Solid White Chart Box */
.chart-box {
    background: #ffffff !important;
    border-radius: 16px !important;
    padding: 1.5rem !important;
    border: 1px solid rgba(0, 0, 0, 0.05) !important;
    box-shadow: inset 0 2px 4px rgba(0, 0, 0, 0.02) !important;
    margin-top: 1rem !important;
}

.glass-card:hover {
    transform: translateY(-8px) !important;
    box-shadow: 0 20px 40px rgba(37, 99, 235, 0.1) !important;
    border-color: rgba(37, 99, 235, 0.2) !important;
}

/* Pill Tabs */
[data-baseweb="tab-list"] {
    background: rgba(255, 255, 255, 0.5) !important;
    border-radius: 50px !important;
    padding: 6px !important;
    gap: 8px !important;
    border: 1px solid rgba(255, 255, 255, 0.3) !important;
    margin-bottom: 2.5rem !important;
    width: fit-content !important;
    margin-left: auto !important;
    margin-right: auto !important;
}

[data-baseweb="tab"] {
    border-radius: 40px !important;
    padding: 10px 28px !important;
    background: transparent !important;
    transition: all 0.3s ease !important;
    color: #64748B !important;
    font-weight: 500 !important;
}

[data-baseweb="tab"]:hover {
    color: #2563EB !important;
    background: rgba(37, 99, 235, 0.05) !important;
}

[data-baseweb="tab"][aria-selected="true"] {
    background: #2563EB !important;
    color: white !important;
    box-shadow: 0 4px 15px rgba(37, 99, 235, 0.3) !important;
}

/* Premium Buttons */
.stButton > button {
    width: 100% !important;
    border-radius: 12px !important;
    border: none !important;
    background: linear-gradient(135deg, #2563EB 0%, #38BDF8 100%) !important;
    color: white !important;
    font-weight: 600 !important;
    padding: 0.8rem 1.5rem !important;
    transition: all 0.3s cubic-bezier(0.4, 0, 0.2, 1) !important;
    box-shadow: 0 4px 12px rgba(37, 99, 235, 0.2) !important;
}

.stButton > button:hover {
    transform: scale(1.02) !important;
    box-shadow: 0 8px 20px rgba(37, 99, 235, 0.4) !important;
    filter: brightness(1.1) !important;
}

/* Dropdowns & Inputs */
.stSelectbox div[data-baseweb="select"], .stTextInput input, .stFileUploader > div > div {
    background: rgba(255, 255, 255, 0.8) !important;
    border-radius: 12px !important;
    border: 1px solid rgba(0, 0, 0, 0.05) !important;
}

/* Metric Box Styling */
[data-testid="stMetric"] {
    background: rgba(255, 255, 255, 0.7) !important;
    padding: 1.5rem !important;
    border-radius: 16px !important;
    border: 1px solid rgba(255, 255, 255, 0.5) !important;
    display: flex !important;
    flex-direction: column !important;
    align-items: center !important;
    text-align: center !important;
}

/* Chart alignment */
.js-plotly-plot, .plotly {
    margin: 0 auto !important;
}

/* Animations */
@keyframes fadeInUp {
    from { opacity: 0; transform: translateY(30px); }
    to { opacity: 1; transform: translateY(0); }
}

.fade-in {
    animation: fadeInUp 0.8s ease-out forwards;
}

/* Labels */
label p {
    font-weight: 600 !important;
    color: #1E293B !important;
}

hr {
    border-color: rgba(0,0,0,0.05) !important;
}
</style>
""", unsafe_allow_html=True)


# ============================================================
# SIDEBAR — Control Panel
# ============================================================
with st.sidebar:
    st.markdown("""
        <div style="text-align: center; margin-top: 1rem; margin-bottom: 2rem;">
            <h2 style="font-weight: 800; color: #0F172A; font-size: 1.6rem; margin-bottom: 0;">⚕️ Bias Inspector</h2>
            <p style="color: #64748B; font-size: 0.9rem; font-weight: 500;">AI Fairness Dashboard</p>
        </div>
        <hr style="margin-bottom: 2rem; opacity: 0.1;">
    """, unsafe_allow_html=True)

    dataset_choice = st.selectbox(
        "📁 Choose Dataset",
        ["Sample Dataset", "Heart Dataset", "Upload Your Own"]
    )

    df = None
    if dataset_choice == "Sample Dataset":
        try:
            df = pd.read_csv("data/sample_data.csv")
        except Exception:
            st.error("Sample dataset not found.")
    elif dataset_choice == "Heart Dataset":
        try:
            df = pd.read_csv("data/processed_heart.csv")
        except Exception:
            st.error("Heart dataset not found.")
    else:
        file = st.file_uploader("Upload CSV", type=["csv"])
        if file:
            df = pd.read_csv(file)

    target = None
    sensitive = None
    analyze_btn = False

    if df is not None:
        st.markdown("<br>", unsafe_allow_html=True)
        target = st.selectbox("🎯 Target Column", df.columns, key="target_sel")
        sensitive = st.selectbox("🛡️ Sensitive Attribute", df.columns, key="sens_sel")
        st.markdown("<br>", unsafe_allow_html=True)
        analyze_btn = st.button("🔍 Analyze Data", use_container_width=True, type="primary")
    


# ============================================================
# MAIN CONTENT
# ============================================================
if df is None:
    # LANDING HERO
    st.markdown("""
        <div class="fade-in" style="display: flex; flex-direction: column; align-items: center; justify-content: center; min-height: 70vh;">
            <h1>Healthcare AI Bias Inspector</h1>
            <p class="hero-subtitle">Ensure fairness and transparency in your clinical AI models with state-of-the-art bias detection and mitigation tools.</p>
            <div style="background: linear-gradient(90deg, #2563EB, #38BDF8); height: 5px; width: 100px; border-radius: 10px;"></div>
        </div>
    """, unsafe_allow_html=True)

else:
    # HERO HEADER
    st.markdown("""
        <div class="fade-in">
            <h1>Healthcare AI Bias Inspector</h1>
            <p class="hero-subtitle">Fair AI for Better Healthcare Decisions</p>
        </div>
    """, unsafe_allow_html=True)

    # TABS
    tab_bias, tab_mitigate = st.tabs(["📊 Bias Analysis", "🛠️ Mitigation Strategy"])

    # Run analysis on button click
    if analyze_btn:
        st.session_state["analyzed"] = True
        model, X_test, y_test, sens_test, y_pred, overall_acc = train_model(df, target, sensitive)
        overall_acc, acc_dict, bias_gap = detect_bias(y_test, y_pred, sens_test)

        st.session_state["model"] = model
        st.session_state["X_test"] = X_test
        st.session_state["overall_acc"] = overall_acc
        st.session_state["acc_dict"] = acc_dict
        st.session_state["bias_gap"] = bias_gap

    analyzed = st.session_state.get("analyzed", False)

    # ------------------ TAB 1: BIAS ANALYSIS ------------------
    with tab_bias:
        if not analyzed:
            st.info("Select your dataset and click **Analyze Data** in the sidebar to begin.")
        else:
            overall_acc = st.session_state["overall_acc"]
            acc_dict = st.session_state["acc_dict"]
            bias_gap = st.session_state["bias_gap"]

            # Metrics Card
            st.markdown("""
                <div class="glass-card" style="padding: 24px !important; margin-bottom: 24px;">
                    <h3 style="margin-top: 0; font-size: 1.15rem; color: #0f172a; font-weight: 700;">Model Performance & Bias Metrics</h3>
                </div>
            """, unsafe_allow_html=True)
            m1, m2, m3 = st.columns(3)
            m1.metric("Overall Accuracy", f"{overall_acc:.2%}")
            m2.metric("Bias Gap", f"{bias_gap:.2%}", delta="High Bias" if bias_gap > 0.1 else "Acceptable", delta_color="inverse")
            m3.metric("Sensitive Groups", len(acc_dict))

            # Metrics Card
            st.markdown("""
                <div class="glass-card">
                    <h2 style="margin-top: 0; font-size: 1.4rem; color: #0F172A; margin-bottom: 1.5rem;">Model Performance & Bias Metrics</h2>
                    <p style="color: #64748B; font-size: 0.95rem; margin-bottom: 2rem;">Overview of accuracy and identified bias across sensitive groups.</p>
                </div>
            """, unsafe_allow_html=True)
            
            m1, m2, m3 = st.columns(3)
            m1.metric("Overall Accuracy", f"{overall_acc:.2%}")
            m2.metric("Bias Gap", f"{bias_gap:.2%}", delta="High Bias" if bias_gap > 0.1 else "Acceptable", delta_color="inverse")
            m3.metric("Sensitive Groups", len(acc_dict))

            # Bias Chart Card
            st.markdown("<br>", unsafe_allow_html=True)
            st.markdown("""
                <div class="glass-card">
                    <h2 style="margin-top: 0; font-size: 1.4rem; color: #0F172A; margin-bottom: 1rem;">Accuracy by Group</h2>
                    <div class="chart-box">
            """, unsafe_allow_html=True)
            st.plotly_chart(plot_bias(acc_dict, "Accuracy by Group (Before Mitigation)"), use_container_width=True)
            st.markdown("</div></div>", unsafe_allow_html=True)

            # SHAP Card
            st.markdown("<br>", unsafe_allow_html=True)
            st.markdown("""
                <div class="glass-card">
                    <h2 style="margin-top: 0; font-size: 1.4rem; color: #0F172A; margin-bottom: 0.5rem;">Model Explainability</h2>
                    <p style="color: #64748B; font-size: 0.95rem; margin-bottom: 1.5rem;">SHAP values showing feature importance and their impact on predictions.</p>
                    <div class="chart-box">
            """, unsafe_allow_html=True)
            model = st.session_state["model"]
            X_test = st.session_state["X_test"]
            explainer = shap.Explainer(model, X_test)
            shap_values = explainer(X_test)
            fig_shap, ax_shap = plt.subplots(figsize=(10, 5), facecolor='white')
            shap.summary_plot(shap_values, X_test, show=False)
            plt.tight_layout()
            st.pyplot(fig_shap, use_container_width=True)
            st.markdown("</div></div>", unsafe_allow_html=True)

    # ------------------ TAB 2: MITIGATION ------------------
    with tab_mitigate:
        if not analyzed:
            st.info("Select your dataset and click **Analyze Data** in the sidebar to begin.")
        else:
            overall_acc = st.session_state["overall_acc"]
            acc_dict = st.session_state["acc_dict"]
            bias_gap = st.session_state["bias_gap"]

            # Mitigation Trigger Card
            st.markdown("""
                <div class="glass-card">
                    <h2 style="margin-top: 0; font-size: 1.4rem; color: #0F172A; margin-bottom: 1rem;">Bias Mitigation</h2>
                    <p style="color: #64748B; font-size: 0.95rem; margin-bottom: 2rem;">Apply automated mitigation strategies to balance predictions and reduce disparities across groups.</p>
                </div>
            """, unsafe_allow_html=True)

            if st.button("🚀 Apply Mitigation Fix", type="primary"):
                with st.spinner("Mitigating bias using re-weighting techniques..."):
                    acc_new, new_gap = mitigate_bias(df, target, sensitive)
                    st.session_state["acc_new"] = acc_new
                    st.session_state["new_gap"] = new_gap
                    st.session_state["mitigated"] = True

            if st.session_state.get("mitigated", False):
                acc_new = st.session_state["acc_new"]
                new_gap = st.session_state["new_gap"]

                st.markdown("<br>", unsafe_allow_html=True)
                st.markdown("""
                    <div class="glass-card">
                        <h2 style="margin-top: 0; font-size: 1.4rem; color: #0F172A; margin-bottom: 1.5rem;">Post-Mitigation Results</h2>
                    </div>
                """, unsafe_allow_html=True)
                c1, c2 = st.columns(2)
                c1.metric("New Bias Gap", f"{new_gap:.2%}", f"{(new_gap - bias_gap):.2%}", delta_color="inverse")
                c2.metric("Avg Group Accuracy", f"{(sum(acc_new.values())/len(acc_new)):.2%}")

                st.markdown("<br>", unsafe_allow_html=True)
                st.markdown("""
                    <div class="glass-card">
                        <h2 style="margin-top: 0; font-size: 1.4rem; color: #0F172A; margin-bottom: 1rem;">Performance Comparison</h2>
                        <div class="chart-box">
                """, unsafe_allow_html=True)
                st.plotly_chart(plot_comparison(acc_dict, acc_new), use_container_width=True)
                st.markdown("</div></div>", unsafe_allow_html=True)

                st.markdown("<br>", unsafe_allow_html=True)
                st.markdown("""
                    <div class="glass-card">
                        <h2 style="margin-top: 0; font-size: 1.4rem; color: #0F172A; margin-bottom: 0.5rem;">Export Results</h2>
                        <p style="color: #64748B; font-size: 0.95rem; margin-bottom: 1.5rem;">Generate a comprehensive PDF report of the findings and mitigation impact.</p>
                """, unsafe_allow_html=True)
                if st.button("📄 Generate PDF Report"):
                    generate_report("report.pdf", acc_new, new_gap)
                    with open("report.pdf", "rb") as f:
                        st.download_button("📥 Download PDF", f, file_name="bias_report.pdf", type="primary")
                st.markdown("</div>", unsafe_allow_html=True)

    # ------------------ AI ASSISTANT ------------------
    st.markdown("<br><br>", unsafe_allow_html=True)
    with st.container():
        st.markdown("""
            <div class="glass-card">
                <h2 style="margin-top: 0; font-size: 1.4rem; color: #0F172A; margin-bottom: 0.5rem;">🤖 AI Insights Assistant</h2>
                <p style="color: #64748B; font-size: 0.95rem; margin-bottom: 1.5rem;">Ask questions about your model's fairness and performance results.</p>
        """, unsafe_allow_html=True)
        
        user_input = st.text_input("How can I help you understand the results?", placeholder="e.g., Explain the bias gap...")
        if user_input:
            if analyzed:
                with st.chat_message("assistant", avatar="🤖"):
                    if "bias" in user_input.lower():
                        st.write(f"The initial bias gap is **{st.session_state['bias_gap']:.2%}**.")
                        if "new_gap" in st.session_state:
                            st.write(f"After mitigation, it was reduced to **{st.session_state['new_gap']:.2%}**.")
                    elif "accuracy" in user_input.lower():
                        st.write(f"The initial overall accuracy is **{st.session_state['overall_acc']:.2%}**.")
                    else:
                        st.write("I can help answer questions about the bias gap or accuracy of your model.")
            else:
                st.warning("Please run 'Analyze Data' first!")
        
        st.markdown("</div>", unsafe_allow_html=True)

