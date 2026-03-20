import streamlit as st
import pandas as pd
import os
import matplotlib.pyplot as plt
import networkx as nx

from src.main_pipeline import run_pipeline

# ==============================
# PAGE CONFIG
# ==============================
st.set_page_config(page_title="Fraud Dashboard", layout="wide")

st.title("Fraud Risk Analysis Dashboard")
st.write("Risk-based fraud detection system")

# ==============================
# SESSION STATE
# ==============================
if "fraud_df" not in st.session_state:
    st.session_state.fraud_df = None

# ==============================
# RUN PIPELINE BUTTON
# ==============================
if st.button("Run Fraud Detection"):
    with st.spinner("Processing..."):
        run_pipeline()

        # Load generated CSV
        if os.path.exists("fraud_report.csv"):
            st.session_state.fraud_df = pd.read_csv("fraud_report.csv")

# ==============================
# LOAD DATA (SAFE)
# ==============================
fraud_df = st.session_state.fraud_df

# ==============================
# KPI SECTION
# ==============================
st.subheader("Risk Summary")

if fraud_df is not None:

    total_txn = len(fraud_df)
    high_risk = len(fraud_df[fraud_df["Risk_Level"] == "High Risk"])
    medium_risk = len(fraud_df[fraud_df["Risk_Level"] == "Medium Risk"])
    low_risk = len(fraud_df[fraud_df["Risk_Level"] == "Low Risk"])

    col1, col2, col3, col4 = st.columns(4)

    col1.metric("Total Transactions", total_txn)
    col2.metric("High Risk", high_risk)
    col3.metric("Medium Risk", medium_risk)
    col4.metric("Low Risk", low_risk)

else:
    st.info("Run the model to see results")

# ==============================
# PIE CHART
# ==============================
st.subheader("Risk Distribution")

if fraud_df is not None:

    counts = fraud_df["Risk_Level"].value_counts()

    col1, col2, col3 = st.columns([1, 2, 1])  # center alignment

    with col2:
        fig, ax = plt.subplots(figsize=(4, 4))  # 👈 reduced size
        ax.pie(counts, labels=counts.index, autopct="%1.1f%%")
        ax.set_title("Risk Distribution")

        st.pyplot(fig)

# ==============================
# FILTER BUTTONS
# ==============================
st.subheader("Fraud Categories")

col1, col2, col3, col4 = st.columns(4)

selected = None

if col1.button("Low Risk"):
    selected = "Low Risk"

if col2.button("Medium Risk"):
    selected = "Medium Risk"

if col3.button("High Risk"):
    selected = "High Risk"

if col4.button("All"):
    selected = "All"

# ==============================
# DISPLAY TABLE (FULL WIDTH)
# ==============================
if fraud_df is not None:

    if selected == "Low Risk":
        df_show = fraud_df[fraud_df["Risk_Level"] == "Low Risk"]

    elif selected == "Medium Risk":
        df_show = fraud_df[fraud_df["Risk_Level"] == "Medium Risk"]

    elif selected == "High Risk":
        df_show = fraud_df[fraud_df["Risk_Level"] == "High Risk"]

    else:
        df_show = fraud_df

    st.dataframe(df_show, use_container_width=True)

# ==============================
# GRAPH VISUALIZATION
# ==============================
st.subheader("Fraud Network Graph")

if fraud_df is not None:

    sample_df = fraud_df.sample(min(150, len(fraud_df)))  # 👈 reduce nodes

    G = nx.Graph()

    for i in range(len(sample_df) - 1):
        G.add_edge(sample_df.iloc[i]["Transaction_ID"],
                   sample_df.iloc[i + 1]["Transaction_ID"])

    col1, col2, col3 = st.columns([1, 2, 1])  # center

    with col2:
        fig, ax = plt.subplots(figsize=(5, 4))  # 👈 smaller graph

        nx.draw(
            G,
            node_size=15,      # 👈 smaller nodes
            width=0.5,         # 👈 thinner edges
            ax=ax
        )

        ax.set_title("Fraud Network")

        st.pyplot(fig)

else:
    st.info("Graph will appear after running model")

# ==============================
# DOWNLOAD REPORT
# ==============================
st.subheader("Download Report")

if fraud_df is not None:
    st.download_button(
        label="Download CSV",
        data=fraud_df.to_csv(index=False),
        file_name="fraud_report.csv",
        mime="text/csv"
    )
else:
    st.info("Run model to generate report")