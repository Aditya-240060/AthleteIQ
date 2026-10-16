"""
Telemetry and Workload Visualizer Component for AthleteIQ.
Renders interactive charts, ACWR threshold zones, GPS metrics, and soreness trends.
"""

import os
import streamlit as st
import pandas as pd


def render_telemetry_view(csv_path: str = None):
    """Renders the player telemetry and sports science workload dashboard."""
    csv_file = csv_path or os.path.join("data", "private_telemetry", "raw_csvs", "player_daily_metrics.csv")
    
    if not os.path.exists(csv_file):
        st.warning(f"Telemetry dataset not found at `{csv_file}`. Please run data ingestion first.")
        return

    df = pd.read_csv(csv_file)
    df["date"] = pd.to_datetime(df["date"])

    players = df["player_id"].unique().tolist()
    
    col_sel, col_info = st.columns([1, 2])
    with col_sel:
        selected_player = st.selectbox("Select Player for Workload Deep-Dive", options=players, index=0)

    player_df = df[df["player_id"] == selected_player].sort_values("date")
    player_name = player_df["player_name"].iloc[0] if not player_df.empty else selected_player
    player_role = player_df["role"].iloc[0] if not player_df.empty else ""

    with col_info:
        st.markdown(
            f"""
            <div style="background: rgba(30, 41, 59, 0.5); padding: 10px 16px; border-radius: 8px; border: 1px solid rgba(56, 189, 248, 0.2); margin-top: 4px;">
                <span style="font-size: 14px; font-weight: 700; color: #38bdf8;">{player_name} ({selected_player})</span>
                <span style="font-size: 12px; color: #94a3b8; margin-left: 10px;">Role: {player_role}</span>
            </div>
            """,
            unsafe_allow_html=True
        )

    st.markdown("---")

    # High-level KPI cards
    max_acwr = player_df["acwr"].max()
    latest_acwr = player_df["acwr"].iloc[-1]
    total_overs = player_df["overs_bowled"].sum()
    peak_lumbar = player_df["lumbar_soreness"].max()
    latest_lumbar = player_df["lumbar_soreness"].iloc[-1]

    kpi1, kpi2, kpi3, kpi4 = st.columns(4)

    with kpi1:
        acwr_class = "kpi-danger" if max_acwr >= 1.5 else ("kpi-warning" if max_acwr > 1.3 else "kpi-success")
        st.markdown(
            f"""
            <div class="kpi-card">
                <div class="kpi-label">Peak 30-Day ACWR</div>
                <div class="kpi-value {acwr_class}">{max_acwr:.2f}</div>
                <div class="kpi-subtext" style="color: #94a3b8;">Threshold: Danger >= 1.50</div>
            </div>
            """,
            unsafe_allow_html=True
        )

    with kpi2:
        st.markdown(
            f"""
            <div class="kpi-card">
                <div class="kpi-label">Current Workload Ratio</div>
                <div class="kpi-value">{latest_acwr:.2f}</div>
                <div class="kpi-subtext" style="color: #94a3b8;">Sweet Spot: 0.80 - 1.30</div>
            </div>
            """,
            unsafe_allow_html=True
        )

    with kpi3:
        st.markdown(
            f"""
            <div class="kpi-card">
                <div class="kpi-label">Total Overs Bowled</div>
                <div class="kpi-value">{total_overs:.1f}</div>
                <div class="kpi-subtext" style="color: #94a3b8;">Past 30-day sum</div>
            </div>
            """,
            unsafe_allow_html=True
        )

    with kpi4:
        soreness_class = "kpi-danger" if peak_lumbar >= 7 else ("kpi-warning" if peak_lumbar >= 4 else "kpi-success")
        st.markdown(
            f"""
            <div class="kpi-card">
                <div class="kpi-label">Max Lumbar Soreness</div>
                <div class="kpi-value {soreness_class}">{peak_lumbar}/10</div>
                <div class="kpi-subtext" style="color: #94a3b8;">Current: {latest_lumbar}/10</div>
            </div>
            """,
            unsafe_allow_html=True
        )

    st.markdown("### 📈 Workload & ACWR Dynamics (Tim Gabbett Model)")
    
    chart_tab1, chart_tab2, chart_tab3 = st.tabs([
        "Acute vs Chronic Workload",
        "ACWR Risk Trajectory",
        "Bowling Volume & Soreness"
    ])

    with chart_tab1:
        st.caption("Comparison between 7-day fatigue (Acute Load) and 28-day conditioning (Chronic Load).")
        chart_df = player_df.set_index("date")[["acute_workload", "chronic_workload"]]
        chart_df.columns = ["Acute Workload (7-Day)", "Chronic Workload (28-Day)"]
        st.line_chart(chart_df, color=["#f43f5e", "#0ea5e9"])

    with chart_tab2:
        st.caption("ACWR curve. Values > 1.50 fall in the Critical Danger Zone with severe bone stress risk.")
        acwr_chart_df = player_df.set_index("date")[["acwr"]].copy()
        acwr_chart_df["Danger Threshold (1.50)"] = 1.50
        acwr_chart_df["Sweet Spot Upper (1.30)"] = 1.30
        acwr_chart_df["Sweet Spot Lower (0.80)"] = 0.80
        st.line_chart(
            acwr_chart_df,
            color=["#38bdf8", "#f43f5e", "#10b981", "#10b981"]
        )

    with chart_tab3:
        st.caption("Daily overs bowled alongside athlete subjective lumbar spine soreness.")
        c1, c2 = st.columns(2)
        with c1:
            st.bar_chart(player_df.set_index("date")["overs_bowled"], color="#6366f1")
        with c2:
            st.line_chart(player_df.set_index("date")["lumbar_soreness"], color="#f59e0b")

    st.markdown("### 📋 Daily Telemetry Records")
    display_cols = [
        "date", "session_type", "overs_bowled", "total_distance_km",
        "acute_workload", "chronic_workload", "acwr", "lumbar_soreness", "rpe"
    ]
    st.dataframe(player_df[display_cols].sort_values("date", ascending=False), use_container_width=True)
