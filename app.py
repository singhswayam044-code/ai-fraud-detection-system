import streamlit as st
import pandas as pd
import os
import matplotlib.pyplot as plt

from src.main_pipeline import run_pipeline

# =========================
# PAGE CONFIG
# =========================
st.set_page_config(
    page_title="Fraud Risk Dashboard",
    layout="wide",
    initial_sidebar_state="collapsed"
)

# =========================
# CUSTOM STYLING (VINTAGE LOOK)
# =========================
st.markdown("""
    <style>
    body {
        background-color: #0e1117;
    }
    .stApp {
        background: linear-gradient(145deg, #0e1117, #1c1f26);
        color: white;
    }
    h1, h2, h3 {
        color: #f5c542;
    }
    .stButton>button {
        background-color: #1c1f26;
        color: white;
        border-radius: 10px;
        border: 1px solid #444;
        padding: 10px;
    }
    .stButton>button:hover {
        background-color: #f5c542;
        color: black;
    }
    </style>
""", unsafe_allow_html=True)

# =========================
# TITLE
# =========================
st.title("🚨 Fraud Risk Analysis Dashboard")
st.markdown("### Risk-based Fraud Detection System")

# =========================
# SESSION STATE
# =========================
if "selected_risk" not in st.session_state:
    st.session_state.selected_risk = "All"

# =========================
# RUN PIPELINE BUTTON
# =========================
if st.button("🚀 Run Fraud Detection"):
    with st.spinner("Processing fraud detection..."):
        run_pipeline()
    st.success("✅ Fraud Detection Completed")

# =========================
# LOAD DATA SAFELY
# =========================
df = None

if os.path.exists("fraud_report.csv"):
    df = pd.read_csv("fraud_report.csv")

# =========================
# RISK SUMMARY + PIE CHART
# =========================
st.divider()
st.subheader("📊 Risk Summary")

if df is not None and "Risk_Level" in df.columns:

    low = len(df[df["Risk_Level"] == "Low Risk"])
    med = len(df[df["Risk_Level"] == "Medium Risk"])
    high = len(df[df["Risk_Level"] == "High Risk"])

    summary = {
        "Low Risk": low,
        "Medium Risk": med,
        "High Risk": high
    }

    col1, col2 = st.columns([1, 1])

    # SUMMARY TEXT
    with col1:
        st.write(summary)

    # PIE CHART
    with col2:
        fig, ax = plt.subplots()

        labels = ["Low", "Medium", "High"]
        sizes = [low, med, high]
        colors = ["#4CAF50", "#FFC107", "#F44336"]

        ax.pie(
            sizes,
            labels=labels,
            autopct='%1.1f%%',
            colors=colors,
            startangle=140
        )

        ax.set_title("Risk Distribution")
        st.pyplot(fig)

else:
    st.write({
        "Low Risk": "NA",
        "Medium Risk": "NA",
        "High Risk": "NA"
    })

# =========================
# RISK CATEGORY BUTTONS
# =========================
st.divider()
st.subheader("📊 Fraud Risk Categories")

col1, col2, col3, col4 = st.columns(4)

with col1:
    if st.button("🟢 Low Risk"):
        st.session_state.selected_risk = "Low Risk"

with col2:
    if st.button("🟡 Medium Risk"):
        st.session_state.selected_risk = "Medium Risk"

with col3:
    if st.button("🔴 High Risk"):
        st.session_state.selected_risk = "High Risk"

with col4:
    if st.button("🚨 All Fraud"):
        st.session_state.selected_risk = "All"

# =========================
# DISPLAY TABLE (FULL WIDTH)
# =========================
if df is not None:

    if st.session_state.selected_risk == "All":
        filtered_df = df
    else:
        filtered_df = df[df["Risk_Level"] == st.session_state.selected_risk]

    st.dataframe(filtered_df, use_container_width=True)

    # DOWNLOAD BUTTON
    st.download_button(
        label="⬇ Download Report",
        data=df.to_csv(index=False),
        file_name="fraud_report.csv",
        mime="text/csv"
    )

# =========================
# GRAPH VISUALIZATION
# =========================
st.divider()
st.subheader("🕸 Fraud Network Visualization")

if st.button("📊 Generate Fraud Graph"):

    import networkx as nx
    from src.preprocessing import load_and_preprocess
    from src.models.gnn_model import create_graph

    with st.spinner("Building graph..."):

        df_graph = load_and_preprocess().sample(400)

        X = df_graph.drop(['Class', 'Transaction_ID'], axis=1)
        y = df_graph['Class']
        ids = df_graph['Transaction_ID']

        data = create_graph(X, y, ids)

        G = nx.Graph()
        edge_index = data.edge_index.numpy()

        for i in range(edge_index.shape[1]):
            G.add_edge(int(edge_index[0][i]), int(edge_index[1][i]))

        colors = ['red' if y.iloc[i] == 1 else 'lightblue' for i in range(len(y))]

        fig, ax = plt.subplots(figsize=(10, 7))

        pos = nx.spring_layout(G, k=0.15)

        nx.draw(
            G,
            pos,
            node_color=colors,
            node_size=30,
            edge_color='gray',
            width=0.3,
            alpha=0.7
        )

        plt.title("Fraud Network (Red = Fraud)")

        st.pyplot(fig)

st.info("🔴 Red = Fraud | 🔵 Blue = Normal")