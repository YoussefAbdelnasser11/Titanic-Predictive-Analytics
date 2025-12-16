import streamlit as st
import joblib
import numpy as np
from PIL import Image
import os
import time

# Set page configuration
st.set_page_config(
    page_title="Titanic Survival Predictor",
    page_icon="🚢",
    layout="wide",
    initial_sidebar_state="expanded"
)

# --- Modern CSS with Ocean/Titanic Theme ---
st.markdown("""
<style>
/* Main app styling */
.stApp {
    background: linear-gradient(135deg, #0a192f 0%, #1a365d 50%, #0f3460 100%);
    color: #e6f1ff;
}

/* Modern container styling */
.main-container {
    background: rgba(10, 25, 47, 0.85);
    backdrop-filter: blur(10px);
    border-radius: 20px;
    padding: 30px;
    margin: 20px 0;
    border: 1px solid rgba(100, 150, 255, 0.2);
    box-shadow: 0 8px 32px rgba(0, 50, 100, 0.3);
}

/* Title styling with ocean theme */
.main-title {
    text-align: center;
    font-size: 3.5rem;
    font-weight: 800;
    background: linear-gradient(90deg, #00b4db, #0083b0, #005f8a);
    -webkit-background-clip: text;
    -webkit-text-fill-color: transparent;
    margin-bottom: 10px;
    text-shadow: 0 0 30px rgba(0, 180, 219, 0.3);
}

.subtitle {
    text-align: center;
    color: #88d3ce;
    font-size: 1.2rem;
    margin-bottom: 40px;
    font-weight: 300;
}

/* Card styling */
.card {
    background: rgba(15, 40, 70, 0.7);
    border-radius: 15px;
    padding: 25px;
    border-left: 5px solid #00b4db;
    box-shadow: 0 5px 15px rgba(0, 0, 0, 0.2);
    transition: transform 0.3s ease;
}

.card:hover {
    transform: translateY(-5px);
}

/* Button styling */
.stButton > button {
    background: linear-gradient(90deg, #00b4db, #0083b0);
    color: white;
    border: none;
    padding: 15px 40px;
    text-align: center;
    text-decoration: none;
    display: inline-block;
    font-size: 18px;
    margin: 10px 2px;
    cursor: pointer;
    border-radius: 30px;
    font-weight: bold;
    transition: all 0.3s ease;
    box-shadow: 0 4px 15px rgba(0, 180, 219, 0.4);
    width: 100%;
}

.stButton > button:hover {
    background: linear-gradient(90deg, #0083b0, #005f8a);
    transform: scale(1.05);
    box-shadow: 0 6px 20px rgba(0, 180, 219, 0.6);
}

/* Sidebar styling */
[data-testid="stSidebar"] {
    background: linear-gradient(180deg, #0a192f 0%, #1a365d 100%);
    border-right: 1px solid rgba(100, 150, 255, 0.2);
}

[data-testid="stSidebar"] .block-container {
    padding-top: 2rem;
}

/* Input field styling */
.stSelectbox, .stSlider {
    background: rgba(15, 40, 70, 0.5);
    border-radius: 10px;
    padding: 15px;
    margin: 10px 0;
}

/* Prediction result styling */
.success-box {
    background: linear-gradient(135deg, rgba(46, 125, 50, 0.9), rgba(56, 142, 60, 0.9));
    padding: 30px;
    border-radius: 15px;
    text-align: center;
    animation: pulse 2s infinite;
    border: 2px solid #4caf50;
    box-shadow: 0 0 30px rgba(76, 175, 80, 0.3);
}

.danger-box {
    background: linear-gradient(135deg, rgba(198, 40, 40, 0.9), rgba(183, 28, 28, 0.9));
    padding: 30px;
    border-radius: 15px;
    text-align: center;
    animation: pulse 2s infinite;
    border: 2px solid #f44336;
    box-shadow: 0 0 30px rgba(244, 67, 54, 0.3);
}

@keyframes pulse {
    0% { transform: scale(1); }
    50% { transform: scale(1.02); }
    100% { transform: scale(1); }
}

/* Progress bar styling */
.stProgress > div > div > div > div {
    background: linear-gradient(90deg, #00b4db, #0083b0);
}

/* Metric styling */
[data-testid="stMetric"] {
    background: rgba(15, 40, 70, 0.5);
    padding: 15px;
    border-radius: 10px;
    border-left: 4px solid #00b4db;
}

/* Responsive design */
@media (max-width: 768px) {
    .main-title {
        font-size: 2.5rem;
    }
}
</style>
""", unsafe_allow_html=True)

# Load the model
try:
    model_path = os.path.join(os.path.dirname(__file__), '..', 'models', 'best_model.pkl')
    model = joblib.load(model_path)
except FileNotFoundError:
    st.error("❌ Model file not found. Please ensure best_model.pkl exists in the models folder.")
    st.stop()

# --- Main Interface ---
st.markdown('<div class="main-container">', unsafe_allow_html=True)

# Header section
col1, col2, col3 = st.columns([1, 2, 1])
with col2:
    st.markdown('<h1 class="main-title">🚢 TITANIC SURVIVAL PREDICTOR</h1>', unsafe_allow_html=True)
    st.markdown('<p class="subtitle">An ML-powered simulation of the fateful night in 1912</p>', unsafe_allow_html=True)

# Main content columns
main_col1, main_col2 = st.columns([1, 1])

with main_col1:
    # Display ship image with modern card
    st.markdown('<div class="card">', unsafe_allow_html=True)
    st.markdown("### RMS Titanic")
    try:
        img_path = os.path.join(os.path.dirname(__file__), 'assets', 'ship.png')
        img = Image.open(img_path)
        st.image(img, use_column_width=True, caption="The Unsinkable Ship - 1912")
    except FileNotFoundError:
        st.warning("⚠️ Ship image not found. Using placeholder.")
        st.image("https://images.unsplash.com/photo-1591193833165-1a1f46c459ad?w=800", 
                 use_column_width=True, caption="Ocean liner representation")
    st.markdown('</div>', unsafe_allow_html=True)
    
    # Statistics card
    st.markdown('<div class="card">', unsafe_allow_html=True)
    st.markdown("### Historical Statistics")
    col_stats1, col_stats2 = st.columns(2)
    with col_stats1:
        st.metric("Total Passengers", "2,224", "April 14, 1912")
        st.metric("Survivors", "710", "31.6%")
    with col_stats2:
        st.metric("Fatalities", "1,514", "68.4%")
        st.metric("Lifeboats", "20", "Capacity: 1,178")
    st.markdown('</div>', unsafe_allow_html=True)

with main_col2:
    # Prediction input card
    st.markdown('<div class="card">', unsafe_allow_html=True)
    st.markdown("### Passenger Information")
    
    # Create two columns for inputs
    input_col1, input_col2 = st.columns(2)
    
    with input_col1:
        pclass = st.selectbox(
            "Ticket Class 🎫",
            [1, 2, 3],
            index=0,
            help="1st = Upper, 2nd = Middle, 3rd = Lower"
        )
        
        sex = st.selectbox(
            "Gender 👤",
            ["Male", "Female"],
            index=0
        )
        
        age = st.slider(
            "Age 🎂",
            min_value=1,
            max_value=80,
            value=25,
            help="Select passenger's age"
        )
    
    with input_col2:
        fare = st.slider(
            "Ticket Fare 💰",
            min_value=0.0,
            max_value=500.0,
            value=50.0,
            step=1.0,
            help="Ticket price in 1912 GBP"
        )
        
        sibsp = st.slider(
            "Siblings/Spouses 👨‍👩‍👧‍👦",
            min_value=0,
            max_value=8,
            value=0,
            help="Number of siblings/spouses aboard"
        )
        
        parch = st.slider(
            "Parents/Children 👨‍👦",
            min_value=0,
            max_value=6,
            value=0,
            help="Number of parents/children aboard"
        )
    
    st.markdown('</div>', unsafe_allow_html=True)
    
    # Prediction button with loading animation
    st.markdown('<div style="margin-top: 30px;">', unsafe_allow_html=True)
    
    if st.button("🚀 PREDICT SURVIVAL", key="predict_button"):
        # Show loading animation
        with st.spinner("Analyzing passenger data..."):
            time.sleep(1.5)
            
            # Encode gender
            sex_encoded = 1 if sex == "Male" else 0
            
            # Prepare input data
            input_data = np.array([[pclass, sex_encoded, age, fare, sibsp, parch]])
            
            # Make prediction
            prediction = model.predict(input_data)[0]
            prob = model.predict_proba(input_data)[0]
            
            # Display result with animation
            st.markdown("---")
            
            if prediction == 1:
                # Survival result
                st.markdown('<div class="success-box">', unsafe_allow_html=True)
                st.markdown("## 🎉 SURVIVAL PREDICTED!")
                st.markdown(f"### Probability: **{prob[1]*100:.1f}%**")
                
                # Progress bar
                st.progress(prob[1])
                
                # Additional information
                st.markdown("---")
                st.markdown("**Factors in favor:**")
                col_fav1, col_fav2 = st.columns(2)
                with col_fav1:
                    st.success("🔹 Higher survival chance")
                    st.success(f"🔹 {'Female' if sex == 'Female' else 'Male'} passenger")
                with col_fav2:
                    st.success(f"🔹 Class {pclass} ticket")
                    st.success(f"🔹 Age: {age} years")
                
                # Confetti effect
                st.balloons()
                st.markdown('</div>', unsafe_allow_html=True)
                
            else:
                # Non-survival result
                st.markdown('<div class="danger-box">', unsafe_allow_html=True)
                st.markdown("## 💔 FATALITY PREDICTED")
                st.markdown(f"### Probability: **{prob[0]*100:.1f}%**")
                
                # Progress bar
                st.progress(prob[0])
                
                # Additional information
                st.markdown("---")
                st.markdown("**Factors against:**")
                col_against1, col_against2 = st.columns(2)
                with col_against1:
                    st.error("🔸 Lower survival chance")
                    st.error(f"🔸 {'Male' if sex == 'Male' else 'Female'} passenger")
                with col_against2:
                    st.error(f"🔸 Class {pclass} ticket")
                    st.error(f"🔸 Age: {age} years")
                st.markdown('</div>', unsafe_allow_html=True)
                
                # Snow effect for icy water
                st.snow()
            
            # Detailed probability breakdown
            st.markdown('<div class="card" style="margin-top: 20px;">', unsafe_allow_html=True)
            st.markdown("### 📊 Detailed Analysis")
            
            prob_col1, prob_col2, prob_col3 = st.columns(3)
            with prob_col1:
                st.metric("Survival Probability", f"{prob[1]*100:.1f}%", 
                         f"{prob[1]*100 - 31.6:.1f}% vs historical avg")
            with prob_col2:
                st.metric("Fatality Probability", f"{prob[0]*100:.1f}%", 
                         f"{prob[0]*100 - 68.4:.1f}% vs historical avg")
            with prob_col3:
                confidence = abs(prob[0] - prob[1]) * 100
                st.metric("Model Confidence", f"{confidence:.1f}%", 
                         "Certainty level")
            st.markdown('</div>', unsafe_allow_html=True)

# Footer
st.markdown("---")
footer_col1, footer_col2, footer_col3 = st.columns(3)
with footer_col2:
    st.markdown("""
    <div style="text-align: center; padding: 20px; color: #88d3ce;">
    <p>🚢 <strong>Titanic Survival Predictor v2.0</strong></p>
    <p>Powered by Machine Learning • Historical Data Analysis</p>
    <p>© 1912 Memorial Simulation • Created for educational purposes</p>
    </div>
    """, unsafe_allow_html=True)

st.markdown('</div>', unsafe_allow_html=True)
