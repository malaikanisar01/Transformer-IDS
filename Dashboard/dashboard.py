
import streamlit as st
import requests
import pandas as pd
import plotly.express as px


# ============================================================
# CONFIGURATION
# ============================================================

API_URL = "http://127.0.0.1:8000"


# ============================================================
# PAGE CONFIGURATION
# ============================================================

st.set_page_config(
    page_title="Transformer IDS Dashboard",
    page_icon="🛡️",
    layout="wide"
)


# ============================================================
# REFRESH CONTROLS
# ============================================================

refresh_col1, refresh_col2 = st.columns([1, 5])

with refresh_col1:

    if st.button("🔄 Refresh Now"):

        st.rerun()

with refresh_col2:

    st.caption(
        "Click Refresh Now to fetch the latest predictions."
    )


# ============================================================
# TITLE
# ============================================================

st.title("🛡️ Transformer-Based IDS")

st.subheader(
    "Network Intrusion Detection Dashboard"
)

st.write(
    "Real-time monitoring dashboard connected to the "
    "Transformer IDS FastAPI backend."
)


# ============================================================
# API HEALTH CHECK
# ============================================================

try:

    health_response = requests.get(
        f"{API_URL}/health",
        timeout=5
    )

    if health_response.status_code == 200:

        health = health_response.json()

        st.success("🟢 API is connected")

        col1, col2 = st.columns(2)

        with col1:

            st.metric(
                "API Status",
                health["status"]
            )

        with col2:

            st.metric(
                "Model Loaded",
                str(health["model_loaded"])
            )

    else:

        st.error(
            "🔴 API returned an error."
        )

except requests.exceptions.RequestException:

    st.error(
        "🔴 Cannot connect to FastAPI. "
        "Make sure the backend server is running."
    )

    st.stop()


# ============================================================
# LOAD PREDICTIONS
# ============================================================

try:

    predictions_response = requests.get(
        f"{API_URL}/predictions?limit=100",
        timeout=5
    )

    if predictions_response.status_code == 200:

        prediction_data = predictions_response.json()

        predictions = prediction_data.get(
            "predictions",
            []
        )

    else:

        st.error(
            "Could not retrieve prediction history."
        )

        predictions = []

except requests.exceptions.RequestException:

    st.error(
        "Could not connect to prediction history endpoint."
    )

    predictions = []


# ============================================================
# PREDICTION STATISTICS
# ============================================================

if predictions:

    df = pd.DataFrame(predictions)

    total_predictions = len(df)

    benign_predictions = len(
        df[df["predicted_class_id"] == 0]
    )

    attack_predictions = len(
        df[df["predicted_class_id"] != 0]
    )

    average_confidence = (
        df["confidence"].mean() * 100
    )


    # ========================================================
    # DETECTION OVERVIEW
    # ========================================================

    st.markdown(
        "## 📊 Detection Overview"
    )

    col1, col2, col3, col4 = st.columns(4)

    with col1:

        st.metric(
            "Total Predictions",
            total_predictions
        )

    with col2:

        st.metric(
            "🚨 Attack Events",
            attack_predictions
        )

    with col3:

        st.metric(
            "🟢 Benign Traffic",
            benign_predictions
        )

    with col4:

        st.metric(
            "🎯 Avg Confidence",
            f"{average_confidence:.2f}%"
        )


    # ========================================================
    # SECURITY STATUS
    # ========================================================

    st.markdown(
        "## 🚨 Security Status"
    )

    if attack_predictions > 0:

        st.error(
            f"🚨 ATTACK DETECTED — "
            f"{attack_predictions} attack prediction(s) found."
        )

    else:

        st.success(
            "🟢 NO ATTACKS DETECTED — "
            "All recorded traffic is benign."
        )


    # ========================================================
    # LATEST ATTACK ALERT
    # ========================================================

    if attack_predictions > 0:

        attack_df = df[
            df["predicted_class_id"] != 0
        ].copy()

        latest_attack = attack_df.iloc[0]

        st.markdown(
            "## 🚨 Latest Attack Alert"
        )

        alert_col1, alert_col2, alert_col3 = st.columns(3)

        with alert_col1:

            st.metric(
                "Attack Type",
                latest_attack["predicted_class_name"]
            )

        with alert_col2:

            st.metric(
                "Confidence",
                f"{latest_attack['confidence'] * 100:.2f}%"
            )

        with alert_col3:

            st.metric(
                "Detection ID",
                int(latest_attack["id"])
            )

        st.warning(
            f"⚠️ Latest detected attack: "
            f"{latest_attack['predicted_class_name']} "
            f"at {latest_attack['timestamp']}"
        )

            # ====================================================
        # ATTACK SEVERITY
        # ====================================================

        attack_name = latest_attack[
            "predicted_class_name"
        ]

        critical_attacks = [
            "DDoS",
            "DoS Hulk",
            "DoS GoldenEye"
        ]

        high_attacks = [
            "Bot",
            "PortScan",
            "FTP-Patator",
            "SSH-Patator"
        ]

        medium_attacks = [
            "DoS Slowhttptest",
            "DoS slowloris",
            "Web Attack - Brute Force",
            "Web Attack - Sql Injection",
            "Web Attack - XSS"
        ]

        if attack_name in critical_attacks:

            severity = "CRITICAL"

        elif attack_name in high_attacks:

            severity = "HIGH"

        elif attack_name in medium_attacks:

            severity = "MEDIUM"

        else:

            severity = "LOW"


        st.markdown(
            "### ⚠️ Rule-Based Attack Severity"
        )

        st.metric(
            "Risk Level",
            severity
        )

    # ========================================================
    # BENIGN VS ATTACK
    # ========================================================

    st.markdown(
        "## 🛡️ Traffic Classification"
    )

    traffic_data = pd.DataFrame({
        "Traffic Type": [
            "Benign Traffic",
            "Attack Traffic"
        ],
        "Count": [
            benign_predictions,
            attack_predictions
        ]
    })

    fig_traffic = px.pie(
        traffic_data,
        names="Traffic Type",
        values="Count",
        title="Benign vs Attack Traffic",
        hole=0.45
    )

    fig_traffic.update_traces(
        textinfo="label+percent",
        hovertemplate=(
            "<b>%{label}</b><br>"
            "Count: %{value}<br>"
            "Percentage: %{percent}"
            "<extra></extra>"
        )
    )

    fig_traffic.update_layout(
        height=400
    )

    st.plotly_chart(
        fig_traffic,
        use_container_width=True
    )


    # ========================================================
    # ATTACK DISTRIBUTION
    # ========================================================

    st.markdown(
        "## 📈 Detection Distribution"
    )

    class_counts = (
        df["predicted_class_name"]
        .value_counts()
        .reset_index()
    )

    class_counts.columns = [
        "Attack Type",
        "Count"
    ]

    fig = px.bar(
        class_counts,
        x="Attack Type",
        y="Count",
        title="Network Attack Distribution",
        labels={
            "Attack Type": "Attack Type",
            "Count": "Number of Predictions"
        }
    )

    fig.update_layout(
        xaxis_tickangle=-45,
        height=450
    )

    st.plotly_chart(
        fig,
        use_container_width=True
    )


    # ========================================================
    # CONFIDENCE ANALYSIS
    # ========================================================

    st.markdown(
        "## 🎯 Prediction Confidence Analysis"
    )

    confidence_df = df[
        [
            "predicted_class_name",
            "confidence"
        ]
    ].copy()

    confidence_df["confidence"] = (
        confidence_df["confidence"] * 100
    )

    confidence_df = confidence_df.rename(
        columns={
            "predicted_class_name": "Prediction",
            "confidence": "Confidence (%)"
        }
    )

    fig_confidence = px.bar(
        confidence_df,
        x="Prediction",
        y="Confidence (%)",
        title="Prediction Confidence",
        labels={
            "Prediction": "Predicted Class",
            "Confidence (%)": "Confidence"
        }
    )

    fig_confidence.update_layout(
        xaxis_tickangle=-45,
        yaxis_range=[0, 100],
        height=450
    )

    st.plotly_chart(
        fig_confidence,
        use_container_width=True
    )


    # ========================================================
    # PREDICTION HISTORY
    # ========================================================

    st.markdown(
        "## 🕒 Recent Prediction History"
    )

    display_df = df[
        [
            "id",
            "timestamp",
            "predicted_class_name",
            "confidence",
            "time_steps",
            "features_per_step"
        ]
    ].copy()

    display_df["confidence"] = (
        display_df["confidence"] * 100
    ).round(2)

    display_df = display_df.rename(
        columns={
            "id": "ID",
            "timestamp": "Timestamp",
            "predicted_class_name": "Prediction",
            "confidence": "Confidence (%)",
            "time_steps": "Time Steps",
            "features_per_step": "Features / Step"
        }
    )

    st.dataframe(
        display_df,
        use_container_width=True,
        hide_index=True
    )


# ============================================================
# NO PREDICTIONS
# ============================================================

else:

    st.info(
        "No predictions are currently available "
        "in the database."
    )

