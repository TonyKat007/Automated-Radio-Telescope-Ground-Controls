import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from scipy.signal import find_peaks
from scipy.stats import linregress

# --- App Configuration ---
st.set_page_config(
    page_title="Radio Astronomy & Horn Antenna Dashboard",
    layout="wide",
    initial_sidebar_state="expanded"
)

st.title("📡 Radio Astronomy & Horn Antenna Dashboard")
st.markdown("""
Welcome to the data processing pipeline for our 21-cm Horn Antenna project. 
Navigate through the tabs to analyze raw hydrogen spectra, map galactic kinematics, 
measure solar beamwidths, and calibrate atmospheric noise.
""")

# --- Main Feature Tabs ---
tabs = st.tabs([
    "1. 21-cm Spectrum Analyzer", 
    "2. Galactic Spiral Arm Mapper", 
    "3. Solar Drift Scan (Beamwidth)", 
    "4. Atmospheric Skydip Calibration"
])

# ==========================================
# TAB 1: 21-cm Spectrum Analyzer
# ==========================================
with tabs[0]:
    st.subheader("Data Reduction: 21-cm Neutral Hydrogen Line")
    
    # File Uploader
    uploaded_file = st.file_uploader("Upload raw SDR spectrum (.csv)", type=["csv"], key="spec_upload")
    
    if uploaded_file is not None:
        # If user uploads real data
        df = pd.read_csv(uploaded_file)
    else:
        # Generate mock data for demonstration if no file is uploaded
        st.info("No file uploaded. Displaying mock 21-cm data.")
        freqs = np.linspace(1419.0, 1421.0, 1000)
        # Baseline + RFI Spike + Hydrogen Peak
        baseline = np.zeros_like(freqs) 
        rfi = np.where(np.abs(freqs - 1419.9) < 0.01, 15, 0)
        h_line = 5 * np.exp(-0.5 * ((freqs - 1420.55) / 0.05)**2)
        noise = np.random.normal(0, 0.5, len(freqs))
        
        df = pd.DataFrame({"Frequency (MHz)": freqs, "Relative Power (dB)": baseline + rfi + h_line + noise})

    freq_col = df.columns[0]
    power_col = df.columns[1]

    # Automated Peak Finding using SciPy
    # Filter frequencies around the expected hydrogen line (ignoring RFI below 1420)
    search_region = df[(df[freq_col] > 1420.1) & (df[freq_col] < 1421.0)]
    peaks, properties = find_peaks(search_region[power_col], prominence=2, width=5)
    
    # Plotting
    fig = px.line(df, x=freq_col, y=power_col, title="Calibrated Radio Spectrum")
    fig.add_vline(x=1420.405, line_dash="dash", line_color="red", annotation_text="Rest Frequency (1420.405 MHz)")
    
    if len(peaks) > 0:
        best_peak_idx = peaks[np.argmax(properties['prominences'])]
        peak_freq = search_region.iloc[best_peak_idx][freq_col]
        peak_power = search_region.iloc[best_peak_idx][power_col]
        
        fig.add_trace(go.Scatter(
            x=[peak_freq], y=[peak_power],
            mode='markers', marker=dict(color='crimson', size=10),
            name=f'Detected Peak: {peak_freq:.4f} MHz'
        ))
        
        col1, col2 = st.columns(2)
        col1.metric("Detected H-Line Peak", f"{peak_freq:.4f} MHz")
        
        # Calculate Doppler Velocity
        c = 299792.458 # km/s
        v_lsr = c * ((1420.405 - peak_freq) / 1420.405)
        col2.metric("Line-of-Sight Velocity (V_lsr)", f"{v_lsr:.2f} km/s")

    st.plotly_chart(fig, use_container_width=True)

# ==========================================
# TAB 2: Galactic Spiral Arm Mapper
# ==========================================
with tabs[1]:
    st.subheader("Kinematics: Mapping the Milky Way")
    
    col1, col2 = st.columns(2)
    with col1:
        v_obs = st.number_input("Observed Velocity (km/s)", value=-30.5)
    with col2:
        l_deg = st.number_input("Galactic Longitude (l) in degrees", value=45.0)

    # Flat Rotation Model Parameters
    R_0 = 8.5 # kpc (Distance to Galactic Center)
    V_0 = 220 # km/s (Sun's orbital speed)
    l_rad = np.radians(l_deg)
    
    # Calculate R (Distance from Galactic Center)
    # Derived from: V_lsr = V_0 * (R_0 / R - 1) * sin(l)
    try:
        ratio = (v_obs / (V_0 * np.sin(l_rad))) + 1
        R_calc = R_0 / ratio
        
        # Convert to X, Y Cartesian Coordinates
        # Sun is at (0, -8.5)
        d = R_0 * np.cos(l_rad) + np.sqrt(R_calc**2 - (R_0 * np.sin(l_rad))**2)
        X = d * np.sin(l_rad)
        Y = d * np.cos(l_rad) - R_0
        
        st.success(f"Calculated Distance from Galactic Center (R): {R_calc:.2f} kpc")
        
        # 2D Map Plot
        fig_map = go.Figure()
        
        # Plot Galactic Center
        fig_map.add_trace(go.Scatter(x=[0], y=[0], mode='markers', marker=dict(symbol='star', size=15, color='black'), name="Galactic Center"))
        # Plot Sun
        fig_map.add_trace(go.Scatter(x=[0], y=[-R_0], mode='markers', marker=dict(size=12, color='gold'), name="Our Sun"))
        # Plot Cloud
        fig_map.add_trace(go.Scatter(x=[X], y=[Y], mode='markers', marker=dict(symbol='x', size=12, color='crimson'), name="Hydrogen Cloud"))
        
        # Draw Solar Orbit Ring
        theta = np.linspace(0, 2*np.pi, 100)
        fig_map.add_trace(go.Scatter(x=R_0*np.cos(theta), y=R_0*np.sin(theta), mode='lines', line=dict(dash='dash', color='lightgrey'), name="Solar Orbit"))
        
        fig_map.update_layout(
            title="Bird's-Eye View of the Galaxy",
            xaxis_title="X (kpc)", yaxis_title="Y (kpc)",
            xaxis=dict(range=[-15, 15]), yaxis=dict(range=[-15, 15]),
            width=600, height=600
        )
        st.plotly_chart(fig_map, use_container_width=False)
        
    except Exception as e:
        st.error("Invalid geometry for these parameters. Check your angles.")

# ==========================================
# TAB 3: Solar Drift Scan
# ==========================================
with tabs[2]:
    st.subheader("Solar Transit: Verifying Antenna Beamwidth")
    st.markdown("Calculate your horn's Half-Power Beamwidth (HPBW) using the Earth's rotation (0.25°/min).")
    
    # Mock Time Series Data for the Sun drifting through the beam
    time_mins = np.linspace(-30, 30, 100)
    # Gaussian curve simulating solar transit
    power_watts = 1e-12 + 5e-12 * np.exp(-0.5 * (time_mins / 8)**2) 
    df_sun = pd.DataFrame({"Time (Minutes from Center)": time_mins, "Total Power (W)": power_watts})
    
    fig_sun = px.line(df_sun, x="Time (Minutes from Center)", y="Total Power (W)", title="Solar Drift Scan (Raw Power vs Time)")
    
    # Find Half-Power Points
    max_power = power_watts.max()
    baseline_power = power_watts.min()
    half_power = baseline_power + (max_power - baseline_power) / 2
    
    fig_sun.add_hline(y=half_power, line_dash="dash", line_color="orange", annotation_text="Half-Power Level (-3 dB)")
    st.plotly_chart(fig_sun, use_container_width=True)
    
    # Calculate HPBW
    above_half = time_mins[power_watts >= half_power]
    delta_t = above_half[-1] - above_half[0]
    hpbw = delta_t * 0.25 # Earth rotates 0.25 degrees per minute
    
    st.metric(label="Calculated Time Above Half-Power (Δt)", value=f"{delta_t:.1f} minutes")
    st.success(f"Empirical Half-Power Beamwidth (θ_HPBW) = {hpbw:.1f}°")

# ==========================================
# TAB 4: Atmospheric Skydip
# ==========================================
with tabs[3]:
    st.subheader("System Calibration: The Skydip Method")
    st.markdown("Isolate internal hardware noise ($T_{receiver}$) and atmospheric opacity ($\\tau$) by measuring varying columns of air (Airmass).")
    
    # Editable DataFrame for Skydip Data
    default_data = pd.DataFrame({
        "Elevation Angle θ (deg)": [90, 60, 45, 30, 20],
        "Measured Power (Relative Unit)": [100.0, 101.5, 104.2, 110.0, 120.5]
    })
    
    st.markdown("**Enter your measurements here:**")
    edited_df = st.data_editor(default_data, num_rows="dynamic")
    
    # Calculate Airmass
    angles_rad = np.radians(edited_df["Elevation Angle θ (deg)"])
    edited_df["Airmass (X)"] = 1 / np.sin(angles_rad)
    
    # Linear Regression (Total Power vs Airmass)
    x_data = edited_df["Airmass (X)"].values
    y_data = edited_df["Measured Power (Relative Unit)"].values
    
    if len(x_data) >= 2:
        slope, intercept, r_value, p_value, std_err = linregress(x_data, y_data)
        
        # Plotting the Secant Plot
        fig_dip = px.scatter(edited_df, x="Airmass (X)", y="Measured Power (Relative Unit)", title="Secant Plot (Power vs Airmass)")
        
        # Draw regression line out to X=0 (Zero Atmosphere)
        x_fit = np.linspace(0, max(x_data) * 1.1, 50)
        y_fit = slope * x_fit + intercept
        fig_dip.add_trace(go.Scatter(x=x_fit, y=y_fit, mode='lines', line=dict(dash='dash', color='red'), name=f"Fit (R² = {r_value**2:.3f})"))
        
        fig_dip.add_vline(x=0, line_color="black")
        st.plotly_chart(fig_dip, use_container_width=True)
        
        st.info(f"**Y-Intercept (X=0):** {intercept:.1f} (This represents your hardware's internal $T_{{receiver}}$ without the sky!)")
        st.info(f"**Slope:** {slope:.3f} (This relates to the atmospheric opacity $\\tau$)")
