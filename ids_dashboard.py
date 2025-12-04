import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import joblib
from datetime import datetime, timedelta

# Page configuration
st.set_page_config(
    page_title="AI IDS Dashboard",
    page_icon="🛡️",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS
st.markdown("""
    <style>
    .main-header {font-size: 3rem; color: #1f77b4; text-align: center; margin-bottom: 2rem;}
    .metric-card {background-color: #f0f2f6; padding: 1rem; border-radius: 10px; text-align: center;}
    </style>
""", unsafe_allow_html=True)

# Simulated data (replace with your real preprocessed data)
@st.cache_data
def load_sample_data():
    np.random.seed(42)
    n_samples = 1000
    data = {
        'timestamp': pd.date_range(start='2025-01-01', periods=n_samples, freq='S'),
        'src_ip': np.random.choice(['192.168.1.' + str(i) for i in range(1, 255)], n_samples),
        'dst_ip': np.random.choice(['10.0.0.' + str(i) for i in range(1, 255)], n_samples),
        'src_port': np.random.randint(1024, 65535, n_samples),
        'dst_port': np.random.choice([80, 443, 22, 3389, 445], n_samples),
        'protocol': np.random.choice(['TCP', 'UDP', 'ICMP'], n_samples),
        'bytes_in': np.random.randint(100, 10000, n_samples),
        'bytes_out': np.random.randint(50, 5000, n_samples),
        'prediction': np.random.choice(['Normal', 'Anomaly', 'DoS', 'Probe', 'R2L', 'U2R'], n_samples, p=[0.85, 0.05, 0.04, 0.03, 0.02, 0.01]),
        'confidence': np.random.uniform(0.7, 1.0, n_samples),
        'model_used': np.random.choice(['Isolation Forest', 'Random Forest', 'XGBoost', 'LSTM Autoencoder'], n_samples)
    }
    return pd.DataFrame(data)

# Load data
df = load_sample_data()

# Sidebar controls
st.sidebar.header("🛠️ Controls")
model_filter = st.sidebar.multiselect("Select Models", 
                                      options=df['model_used'].unique(), 
                                      default=df['model_used'].unique())
attack_filter = st.sidebar.multiselect("Filter Attacks", 
                                      options=['Normal', 'Anomaly', 'DoS', 'Probe', 'R2L', 'U2R'], 
                                      default=['Normal', 'Anomaly'])
time_range = st.sidebar.slider("Time Range (hours)", 1, 24, 6)

# Filter data with safety check
filtered_df = df[
    (df['model_used'].isin(model_filter)) & 
    (df['prediction'].isin(attack_filter)) &
    (df['timestamp'] >= datetime.now() - timedelta(hours=time_range))
].copy()

# Main Header
st.markdown('<h1 class="main-header">🛡️ AI Intrusion Detection System</h1>', unsafe_allow_html=True)
st.markdown("---")

# KPI Metrics
col1, col2, col3, col4 = st.columns(4)
total_events = len(filtered_df)
normal_traffic = len(filtered_df[filtered_df['prediction'] == 'Normal']) if not filtered_df.empty else 0
alerts = total_events - normal_traffic
detection_rate = (alerts / total_events * 100) if total_events > 0 else 0

with col1:
    st.markdown(f"""
    <div class="metric-card">
        <h3>Total Events</h3>
        <h2 style="color: #2ca02c;">{total_events:,}</h2>
    </div>
    """, unsafe_allow_html=True)

with col2:
    st.markdown(f"""
    <div class="metric-card">
        <h3>Normal Traffic</h3>
        <h2 style="color: #1f77b4;">{normal_traffic:,}</h2>
    </div>
    """, unsafe_allow_html=True)

with col3:
    st.markdown(f"""
    <div class="metric-card">
        <h3>Alerts</h3>
        <h2 style="color: #d62728;">{alerts:,}</h2>
    </div>
    """, unsafe_allow_html=True)

with col4:
    st.markdown(f"""
    <div class="metric-card">
        <h3>Detection Rate</h3>
        <h2 style="color: #ff7f0e;">{detection_rate:.1f}%</h2>
    </div>
    """, unsafe_allow_html=True)

st.markdown("---")

# Charts Row 1
col1, col2 = st.columns(2)

with col1:
    st.subheader("📊 Attack Distribution")
    if not filtered_df.empty:
        attack_counts = filtered_df['prediction'].value_counts()
        fig_pie = px.pie(values=attack_counts.values, names=attack_counts.index, 
                        color_discrete_sequence=px.colors.qualitative.Set2)
        fig_pie.update_traces(textposition='inside', textinfo='percent+label')
    else:
        fig_pie = px.pie(values=[100], names=['No Data'], title="No Data Available")
    st.plotly_chart(fig_pie, use_container_width=True)

with col2:
    st.subheader("⏰ Alerts Over Time")
    if not filtered_df.empty:
        # FIXED: Proper time aggregation using DataFrame
        time_agg = filtered_df.set_index('timestamp').resample('5T').size().reset_index(name='count')
        time_agg.columns = ['timestamp', 'alerts_count']  # Rename for clarity
        
        fig_line = px.line(time_agg, 
                          x='timestamp', 
                          y='alerts_count',
                          labels={'timestamp': 'Time', 'alerts_count': 'Alerts'},
                          title="Alerts Over Time")
    else:
        # Empty data fallback
        empty_df = pd.DataFrame({'timestamp': [pd.Timestamp.now()], 'alerts_count': [0]})
        fig_line = px.line(empty_df, x='timestamp', y='alerts_count', title="No Data Available")
    
    st.plotly_chart(fig_line, use_container_width=True)

# Charts Row 2
col1, col2 = st.columns(2)

with col1:
    st.subheader("🎯 Model Performance")
    if not filtered_df.empty:
        model_perf = filtered_df.groupby('model_used')['prediction'].apply(
            lambda x: 100 * (len(x[x != 'Normal']) / len(x))
        ).reset_index(name='detection_rate')
        
        fig_bar = px.bar(model_perf, 
                        x='model_used', 
                        y='detection_rate',
                        title="Detection Rate by Model (%)")
    else:
        empty_df = pd.DataFrame({'model_used': ['No Data'], 'detection_rate': [0]})
        fig_bar = px.bar(empty_df, x='model_used', y='detection_rate', title="No Data Available")
    
    st.plotly_chart(fig_bar, use_container_width=True)

with col2:
    st.subheader("🌐 Top Destination Ports")
    if not filtered_df.empty:
        port_counts = filtered_df['dst_port'].value_counts().head(10).reset_index()
        port_counts.columns = ['dst_port', 'count']
        
        fig_port = px.bar(port_counts,
                         x='dst_port',
                         y='count',
                         title="Traffic by Destination Port")
    else:
        empty_df = pd.DataFrame({'dst_port': ['No Data'], 'count': [0]})
        fig_port = px.bar(empty_df, x='dst_port', y='count', title="No Data Available")
    
    st.plotly_chart(fig_port, use_container_width=True)

# Recent Alerts Table
st.markdown("---")
st.subheader("🚨 Recent Alerts (Last 100)")
alerts_df = filtered_df[filtered_df['prediction'] != 'Normal'].tail(100)

if not alerts_df.empty:
    st.dataframe(alerts_df[['timestamp', 'src_ip', 'dst_ip', 'dst_port', 
                           'prediction', 'confidence', 'model_used']].style
                .format({'confidence': '{:.2%}'}),
                use_container_width=True)
else:
    st.info("✅ No recent alerts detected.")

# Status Indicator
st.markdown("---")
status_col1, status_col2 = st.columns([3, 1])
with status_col1:
    st.subheader("System Status")
with status_col2:
    if alerts > 10:
        st.error("🔴 HIGH ALERT ACTIVITY")
    elif alerts > 0:
        st.warning("🟡 ALERTS DETECTED")
    else:
        st.success("🟢 ALL CLEAR")

# Footer
st.markdown("---")
st.markdown("*AI-based Intrusion Detection System | Powered by Streamlit, Scikit-learn & XGBoost*")
