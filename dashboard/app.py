"""
dashboard/app.py
Streamlit Dashboard for KDHS ECD Delay Prediction
==================================================
Simplified version with:
1. 🔮 Individual Prediction (demographic-based)
2. 📝 ECDI2030 Assessment (20 questions)
3. 🗺️ County Risk Map (interactive)
4. 📍 County Profile (drill-down)
Project: KDHS 2022 ECD Analysis
Methodology: UNICEF ECDI2030 (SDG 4.2.1)
Age Range: 36-59 months
"""
import streamlit as st
import requests
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from pathlib import Path
import os
import json
import sys
from datetime import datetime
from math import pi

# ============================================================================
# Page Configuration
# ============================================================================
st.set_page_config(
    page_title="KDHS ECD Delay Prediction",
    page_icon="👶",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ============================================================================
# Custom CSS - Enhanced Styling
# ============================================================================
st.markdown("""
<style>
/* Main container */
.main {
    padding: 1rem 2rem;
}

/* Headers */
h1, h2, h3, h4, h5, h6 {
    margin: 0.5rem 0 !important;
    font-weight: 600;
    color: #1f77b4;
}

h1 { font-size: 2rem !important; }
h2 { font-size: 1.6rem !important; }
h3 { font-size: 1.3rem !important; }

/* Metric cards - White with colored borders */
.metric-card {
    background-color: white;
    padding: 1.2rem;
    border-radius: 10px;
    border-left: 5px solid #28a745;
    box-shadow: 0 2px 8px rgba(0,0,0,0.08);
    margin: 0.5rem 0;
    transition: all 0.3s ease;
}

.metric-card:hover {
    box-shadow: 0 4px 12px rgba(0,0,0,0.12);
    transform: translateY(-2px);
}

.metric-card h3 {
    font-size: 0.9rem !important;
    margin: 0 0 0.5rem 0 !important;
    color: #6c757d;
    font-weight: 600;
}

.metric-card p {
    font-size: 1.8rem !important;
    font-weight: bold !important;
    margin: 0 !important;
    color: #28a745;
}

.metric-card small {
    font-size: 0.8rem !important;
    color: #6c757d;
}

/* Purple metric cards for home page */
.metric-card-purple {
    background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
    padding: 1.2rem;
    border-radius: 10px;
    border-left: 5px solid #fff;
    box-shadow: 0 4px 6px rgba(0,0,0,0.1);
    color: white;
    margin: 0.5rem 0;
}

.metric-card-purple h3 {
    font-size: 0.9rem !important;
    margin: 0 0 0.5rem 0 !important;
    color: rgba(255,255,255,0.9);
}

.metric-card-purple p {
    font-size: 1.8rem !important;
    font-weight: bold !important;
    margin: 0 !important;
    color: white;
}

.metric-card-purple small {
    font-size: 0.8rem !important;
    color: rgba(255,255,255,0.8);
}

/* Sidebar */
.sidebar .sidebar-content {
    padding: 1rem !important;
}

.sidebar-header {
    font-size: 1.5rem !important;
    margin-bottom: 1rem !important;
    color: #1f77b4;
}

/* Form elements */
.stNumberInput, .stSelectbox, .stRadio {
    margin-bottom: 0.5rem !important;
}

.stNumberInput label, .stSelectbox label, .stRadio label {
    font-size: 0.9rem !important;
    font-weight: 600;
    margin-bottom: 0.3rem !important;
    color: #2c3e50;
}

/* Buttons */
.stButton > button {
    padding: 0.6rem 1.5rem !important;
    font-size: 0.95rem !important;
    font-weight: 600;
    margin: 0.5rem 0 !important;
    border-radius: 8px !important;
    border: none !important;
    background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
    color: white !important;
    transition: all 0.3s ease !important;
}

.stButton > button:hover {
    transform: translateY(-2px);
    box-shadow: 0 4px 12px rgba(102, 126, 234, 0.4);
}

/* Tables */
.dataframe {
    font-size: 0.9rem !important;
    border-radius: 8px;
    overflow: hidden;
}

/* Alerts */
.stAlert {
    padding: 0.8rem !important;
    margin: 0.5rem 0 !important;
    font-size: 0.95rem !important;
    border-radius: 8px !important;
}

/* Container spacing */
.block-container {
    padding-top: 1.5rem !important;
    padding-bottom: 1.5rem !important;
}

/* Info boxes */
.info-box {
    background-color: #e7f3ff;
    padding: 1rem;
    border-radius: 8px;
    border-left: 4px solid #2196F3;
    margin: 0.8rem 0;
}

/* Section headers */
.section-header {
    background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
    color: white;
    padding: 0.8rem 1.2rem;
    border-radius: 8px;
    margin: 1.5rem 0 1rem 0;
    font-weight: 600;
}

/* County cards */
.county-card {
    background-color: #e8f4f8;
    padding: 1.2rem;
    border-radius: 10px;
    border-left: 5px solid #28a745;
    margin: 0.8rem 0;
    box-shadow: 0 2px 4px rgba(0,0,0,0.1);
}

.recommendation-card {
    background-color: #fff3cd;
    padding: 1rem;
    border-radius: 8px;
    border-left: 4px solid #ffc107;
    margin: 0.5rem 0;
}
</style>
""", unsafe_allow_html=True)

# API Configuration
API_URL = os.getenv("API_URL", "http://127.0.0.1:5000")

# ============================================================================
# Helper Functions
# ============================================================================
def check_api_health():
    """Check if API is running"""
    try:
        response = requests.get(f"{API_URL}/health", timeout=5)
        return response.status_code == 200
    except:
        return False

def predict_delay(features: dict):
    """Send demographic prediction request to API"""
    try:
        response = requests.post(f"{API_URL}/predict", json=features, timeout=30)
        if response.status_code == 200:
            return response.json()
        else:
            return {"error": response.json().get('message', 'Unknown error')}
    except Exception as e:
        return {"error": str(e)}

def get_feature_importance():
    """Get SHAP feature importance from API"""
    try:
        response = requests.get(f"{API_URL}/features", timeout=10)
        if response.status_code == 200:
            data = response.json()
            if not data.get('feature_importance'):
                st.warning("⚠️  API returned empty feature importance data")
                return None
            return data
        else:
            st.warning(f"⚠️  API returned status {response.status_code} for /features")
            return None
    except Exception as e:
        st.warning(f"⚠️  Error loading feature importance: {str(e)}")
        return None

def get_model_info():
    """Get model info from API - dynamically loads best model"""
    try:
        response = requests.get(f"{API_URL}/model-info", timeout=10)
        if response.status_code == 200:
            data = response.json()
            if not data.get('selected_model'):
                st.warning("⚠️  API returned empty model info data")
                return None
            return data
        else:
            st.warning(f"⚠️  API returned status {response.status_code} for /model-info")
            return None
    except Exception as e:
        st.warning(f"⚠️  Error loading model info: {str(e)}")
        return None

def get_geo_summary():
    """Get geospatial summary from models folder"""
    try:
        geo_path = Path(__file__).parent.parent / 'models' / 'geo_summary.json'
        if geo_path.exists():
            with open(geo_path, 'r') as f:
                return json.load(f)
    except:
        pass
    return None

def get_county_data():
    """Get county data from local tables folder"""
    try:
        county_path = Path(__file__).parent.parent / 'reports' / 'tables' / 'county_dashboard_data.csv'
        if county_path.exists():
            return pd.read_csv(county_path).to_dict('records')
    except:
        pass
    return None

def load_ecd_questions_local():
    """Load ECD questions from local src/ecd_questions.py"""
    try:
        project_root = Path(__file__).parent.parent
        if str(project_root) not in sys.path:
            sys.path.insert(0, str(project_root))
        from src.ecd_questions import ECD_QUESTIONS, calculate_ecd_score, validate_responses, get_all_question_ids
        return True, ECD_QUESTIONS, calculate_ecd_score, validate_responses, get_all_question_ids
    except ImportError as e:
        return False, f"Could not load ECD questions module: {str(e)}"
    except Exception as e:
        return False, f"Error loading ECD questions: {str(e)}"

# ============================================================================
# HELPER: Display ECD Results
# ============================================================================
def display_ecd_results(ecd_result, child_age, assessment_date, caregiver_relation, county, responses, ECD_QUESTIONS):
    """Display ECD assessment results with visualizations and recommendations"""
    
    # Display Results
    st.markdown("---")
    st.subheader("🎯 ECD Assessment Results")
    col1, col2, col3 = st.columns(3)
    
    with col1:
        if ecd_result['on_track']['composite']:
            st.success("✅ ON TRACK")
            st.caption("Child is developmentally on track")
        else:
            st.error("⚠️ AT RISK")
            st.caption("Child may need additional support")
    
    with col2:
        st.metric(
            "Total Score",
            f"{ecd_result['scores']['total']}/{ecd_result['max_scores']['total']}",
            delta=f"{ecd_result['percentages']['total']:.0f}%"
        )
    
    with col3:
        st.metric(
            "Threshold",
            f"{ecd_result['threshold']['total']} milestones",
            delta=f"Age {child_age}mo"
        )
    
    # Domain Scores
    st.markdown("---")
    st.subheader("📊 Domain-Specific Scores")
    domain_icons = {
        'physical': '🏃',
        'language': '💬',
        'literacy': '📚',
        'socio_emotional': '🤝'
    }
    
    cols = st.columns(2)
    for i, (domain_key, domain_data) in enumerate(ECD_QUESTIONS.items()):
        score = ecd_result['scores'][domain_key]
        max_score = ecd_result['max_scores'][domain_key]
        percentage = ecd_result['percentages'][domain_key]
        on_track = ecd_result['on_track'][domain_key]
        threshold = ecd_result['threshold'].get(domain_key, 0)
        
        with cols[i % 2]:
            border_color = '#28a745' if on_track else '#dc3545'
            text_color = '#28a745' if on_track else '#dc3545'
            status_text = '✅ On Track' if on_track else '⚠️ Below Threshold'
            
            html = f"""<div class="metric-card" style="border-left-color: {border_color};">
                <h3>{domain_icons.get(domain_key, '📊')} {domain_data['domain_name']}</h3>
                <p style="font-size: 1.5rem; font-weight: bold; color: {text_color};">
                    {score}/{max_score} ({percentage:.0f}%)
                </p>
                <p>{status_text} (Threshold: {threshold})</p>
            </div>"""
            st.markdown(html, unsafe_allow_html=True)
    
    # Radar Chart
    st.markdown("---")
    st.subheader("🎯 ECDI Domain Profile")
    
    categories = list(ECD_QUESTIONS.keys())
    values = [ecd_result['percentages'][cat] for cat in categories]
    
    N = len(categories)
    angles = [n / float(N) * 2 * pi for n in range(N)]
    angles += angles[:1]
    
    fig, ax = plt.subplots(figsize=(8, 8), subplot_kw=dict(polar=True))
    plt.xticks(angles[:-1], [ECD_QUESTIONS[cat]['domain_name'] for cat in categories], color='grey', size=10)
    ax.set_rlabel_position(0)
    plt.yticks([25, 50, 75, 100], ["25", "50", "75", "100"], color="grey", size=8)
    plt.ylim(0, 100)
    
    values += values[:1]
    ax.plot(angles, values, linewidth=2, linestyle='solid', color='#1f77b4')
    ax.fill(angles, values, '#1f77b4', alpha=0.25)
    ax.plot(angles, [80] * (N + 1), linewidth=1, linestyle='--', color='red', label='On Track Threshold (80%)')
    
    plt.legend(loc='upper right', bbox_to_anchor=(1.3, 1.1))
    plt.title('Child ECDI Domain Profile\n(Percentage of Milestones Achieved)',
              fontsize=14, fontweight='bold', pad=20)
    
    st.pyplot(fig)
    
    # Recommendations
    st.markdown("---")
    st.subheader("💡 Recommendations")
    
    for domain_key, domain_data in ECD_QUESTIONS.items():
        on_track = ecd_result['on_track'][domain_key]
        
        if not on_track:
            border_color = '#ffc107'
            bg_color = '#fff3cd'
            icon = domain_icons.get(domain_key, '📊')
            domain_name = domain_data['domain_name']
            score = ecd_result['scores'][domain_key]
            max_score = ecd_result['max_scores'][domain_key]
            percentage = ecd_result['percentages'][domain_key]
            
            html = f"""<div style="background-color: {bg_color}; padding: 1rem; border-radius: 0.5rem; border-left: 4px solid {border_color}; margin: 0.5rem 0;">
                <h4>⚠️ {icon} {domain_name}</h4>
                <p><strong>Child scored {score}/{max_score} ({percentage:.0f}%)</strong></p>
                <p><strong>Recommendations:</strong></p>
                <ul>"""
            st.markdown(html, unsafe_allow_html=True)
            
            if domain_key == 'physical':
                st.markdown("""<li>Encourage physical play and outdoor activities daily</li>
                <li>Provide opportunities for gross motor skill development</li>
                <li>Practice fine motor skills (buttoning, drawing, stacking)</li>
                <li>Ensure adequate nutrition and regular health check-ups</li>
                </ul></div>""", unsafe_allow_html=True)
            elif domain_key == 'language':
                st.markdown("""<li>Read books and stories to the child daily</li>
                <li>Engage in conversations and ask open-ended questions</li>
                <li>Sing songs and recite rhymes together</li>
                <li>Name objects and describe activities throughout the day</li>
                </ul></div>""", unsafe_allow_html=True)
            elif domain_key == 'literacy':
                st.markdown("""<li>Introduce letters and numbers through play</li>
                <li>Practice counting objects in daily activities</li>
                <li>Provide age-appropriate learning materials</li>
                <li>Enroll in early childhood education programs</li>
                </ul></div>""", unsafe_allow_html=True)
            elif domain_key == 'socio_emotional':
                st.markdown("""<li>Encourage play with other children regularly</li>
                <li>Teach emotion recognition and regulation strategies</li>
                <li>Provide consistent routines and positive reinforcement</li>
                <li>Model appropriate social behavior</li>
                </ul></div>""", unsafe_allow_html=True)
        else:
            icon = domain_icons.get(domain_key, '📊')
            domain_name = domain_data['domain_name']
            score = ecd_result['scores'][domain_key]
            max_score = ecd_result['max_scores'][domain_key]
            percentage = ecd_result['percentages'][domain_key]
            
            html = f"""<div style="background-color: #d4edda; padding: 1rem; border-radius: 0.5rem; border-left: 4px solid #28a745; margin: 0.5rem 0;">
                <h4>✅ {icon} {domain_name}</h4>
                <p><strong>Child scored {score}/{max_score} ({percentage:.0f}%)</strong></p>
                <p><strong>Child is on track!</strong> Continue supporting development through:</p>
                <ul>
                <li>Maintain current stimulation and learning activities</li>
                <li>Monitor progress regularly</li>
                <li>Provide enriched learning environment</li>
                </ul>
                </div>"""
            st.markdown(html, unsafe_allow_html=True)
    
    # Export Results
    st.markdown("---")
    st.subheader("💾 Export Results")
    
    report = {
        'child_info': {
            'age_months': child_age,
            'sex': child_sex,
            'county': county
        },
        'assessment': {
            'date': str(assessment_date),
            'caregiver': caregiver_relation,
            'total_questions': sum(len(d['questions']) for d in ECD_QUESTIONS.values()),
            'answered': len(responses)
        },
        'scores': ecd_result['scores'],
        'percentages': ecd_result['percentages'],
        'on_track': ecd_result['on_track'],
        'threshold': ecd_result['threshold'],
        'responses': responses,
        'timestamp': datetime.now().isoformat()
    }
    
    json_report = json.dumps(report, indent=2)
    
    st.download_button(
        label="📥 Download Assessment Report (JSON)",
        data=json_report,
        file_name=f"ecd_assessment_{child_age}mo_{assessment_date}.json",
        mime="application/json",
        use_container_width=True
    )
    
    st.success("✅ Assessment complete! Download the report for your records.")

# ============================================================================
# Sidebar Navigation (SIMPLIFIED)
# ============================================================================
st.sidebar.title("👶 KDHS ECD")
st.sidebar.markdown("---")

# API Health Check
api_healthy = check_api_health()
if api_healthy:
    st.sidebar.success("✅ API Connected")
else:
    st.sidebar.error("❌ API Offline")
    st.sidebar.info("Start Flask API: `python api/main.py`")

st.sidebar.markdown("---")

# Main Navigation - Simplified (Removed Domain Analysis)
page = st.sidebar.radio(
    "**📋 Navigation**",
    [
        "🏠 Home",
        "📝 ECD Assessment",
        "🔮 Individual Prediction",
        "🗺️ County Risk Map",
        "📍 County Profile",
        "📈 Model Performance",
        "📈 Feature Importance",
        "ℹ️ About"
    ],
    index=0,
    key="main_navigation"
)

st.sidebar.markdown("---")

# Quick Stats in Sidebar - Dynamic from API
st.sidebar.subheader("📊 Quick Stats")

# Fetch model info dynamically
model_info = get_model_info() if api_healthy else None

if model_info and model_info.get('status') == 'success':
    selected_model = model_info.get('selected_model', {})
    model_name = selected_model.get('model_name', 'N/A')
    roc_auc = selected_model.get('roc_auc', 0)
    model_type = selected_model.get('model_type', 'N/A')
    
    st.sidebar.metric("Study Population", "36-59 months")
    st.sidebar.metric("Methodology", "UNICEF ECDI2030")
    st.sidebar.metric("SDG Indicator", "4.2.1")
    st.sidebar.metric("Best Model", model_name)
    st.sidebar.metric("Model Type", model_type)
    st.sidebar.metric("ROC-AUC", f"{roc_auc:.4f}")
else:
    # Fallback to static values if API fails
    st.sidebar.metric("Study Population", "36-59 months")
    st.sidebar.metric("Methodology", "UNICEF ECDI2030")
    st.sidebar.metric("SDG Indicator", "4.2.1")
    st.sidebar.metric("Model", "AdaBoost")
    st.sidebar.metric("ROC-AUC", "0.78")

st.sidebar.markdown("---")
st.sidebar.markdown("""
**Project:** KDHS 2022 ECD Analysis  
**Built with:** Streamlit + Flask + ML  
**License:** Academic Use
""")

# ============================================================================
# Pages
# ============================================================================

# --------------------------
# HOME PAGE (Dynamic Metrics)
# --------------------------
if page == "🏠 Home":
    st.markdown('<p class="main-header">🏠 KDHS ECD Delay Prediction Dashboard</p>', unsafe_allow_html=True)
    st.markdown("---")
    
    # Fetch model info for dynamic metrics
    model_info = get_model_info() if api_healthy else None
    
    # Get dynamic values or use defaults
    if model_info and model_info.get('status') == 'success':
        selected_model = model_info.get('selected_model', {})
        roc_auc = selected_model.get('roc_auc', 0.78)
        f1_score = selected_model.get('f1_score', 0)
        model_name = selected_model.get('model_name', 'AdaBoost')
    else:
        roc_auc = 0.78
        f1_score = 0.56
        model_name = 'AdaBoost'
    
    # Calculate accuracy from confusion matrix if available
    accuracy = 73.3  # Default
    if model_info:
        metrics_path = Path(__file__).parent.parent / 'models' / f"metrics_{model_name.replace(' ', '_')}_composite.json"
        if metrics_path.exists():
            with open(metrics_path, 'r') as f:
                final_metrics = json.load(f)
                accuracy = final_metrics.get('accuracy', 0.733) * 100
    
    # Key Metrics (Dynamic) - Purple cards
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        st.markdown(f"""<div class="metric-card-purple">
            <h3>📈 Model Performance</h3>
            <p>ROC-AUC: {roc_auc:.2f}</p>
            <small>↑ {model_name}</small>
        </div>""", unsafe_allow_html=True)
    
    with col2:
        st.markdown("""<div class="metric-card-purple">
            <h3>👶 Study Population</h3>
            <p>36-59 months</p>
            <small>↑ UNICEF ECDI2030</small>
        </div>""", unsafe_allow_html=True)
    
    with col3:
        st.markdown("""<div class="metric-card-purple">
            <h3>📊 Features</h3>
            <p>16</p>
            <small>↑ Demographic + Household</small>
        </div>""", unsafe_allow_html=True)
    
    with col4:
        st.markdown(f"""<div class="metric-card-purple">
            <h3>🎯 Accuracy</h3>
            <p>{accuracy:.1f}%</p>
            <small>↑ Test Set</small>
        </div>""", unsafe_allow_html=True)
    
    st.markdown("---")
    
    # About Section
    st.markdown('<div class="section-header">📋 About This Dashboard</div>', unsafe_allow_html=True)
    st.markdown(f"""
This dashboard predicts **Early Childhood Development (ECD) Delay** for children aged 36-59 months
using the **UNICEF ECDI2030** methodology aligned with **SDG Indicator 4.2.1**.

**Key Features:**
- 🔮 **Individual Prediction**: Get ECD delay predictions with risk assessment
- 📝 **ECD Assessment**: Answer 20 questions for immediate domain-specific feedback
- 🗺️ **County Risk Map**: Explore geographic patterns across Kenya
- 📍 **County Profile**: Drill down into county-specific insights and recommendations
- 📈 **Model Performance**: View model metrics and comparison
- 📈 **Feature Importance**: Understand key predictors (SHAP analysis)

**Model Information:**
- **Best Model:** {model_name}
- **ROC-AUC:** {roc_auc:.4f}
- **F1-Score:** {f1_score:.4f}
- Trained on KDHS 2022 data (N = 3,599)
- UNICEF ECDI2030 age-specific thresholds
- Explainable AI with SHAP + LIME
""")
    
    st.markdown("---")
    
    # Quick Actions
    st.markdown('<div class="section-header">🚀 Quick Actions</div>', unsafe_allow_html=True)
    col1, col2 = st.columns(2)
    
    with col1:
        if st.button("🔮 Make a Prediction", use_container_width=True):
            st.session_state.page = "🔮 Individual Prediction"
            st.rerun()
    
    with col2:
        if st.button("📝 Start ECD Assessment", use_container_width=True):
            st.session_state.page = "📝 ECD Assessment"
            st.rerun()
    
    st.info("👈 Use the sidebar to navigate to different sections")

# --------------------------
# ECD ASSESSMENT PAGE
# --------------------------
elif page == "📝 ECD Assessment":
    st.markdown('<p class="main-header">📝 UNICEF ECDI2030 Assessment</p>', unsafe_allow_html=True)
    st.markdown("---")
    
    st.markdown("""
Answer these **20 questions** about the child's development to get an immediate ECD assessment.
This is based on the **UNICEF ECDI2030** methodology aligned with **SDG Indicator 4.2.1**.

**Instructions:**
- Answer all questions honestly based on the child's current abilities
- Takes approximately 3-5 minutes to complete
- Results are provided immediately
- All responses are confidential
""")
    
    # Load ECD questions
    local_loaded, *local_data_or_error = load_ecd_questions_local()
    
    if local_loaded:
        ECD_QUESTIONS, calculate_ecd_score, validate_responses, get_all_question_ids = local_data_or_error
        scoring_method = "local"
        st.success("✅ Using local ECD scoring (fast, no API required)")
    else:
        error_msg = local_data_or_error[0] if local_data_or_error else "Unknown error"
        st.warning(f"⚠️  {error_msg}")
        st.info("🔄 Falling back to API-based scoring. Please ensure API is running.")
        scoring_method = "api"
        ECD_QUESTIONS = None
        calculate_ecd_score = None
        validate_responses = None
    
    st.subheader("👶 Child Information")
    col1, col2 = st.columns(2)
    
    with col1:
        child_age = st.number_input("Child Age (months)*", min_value=36, max_value=59, value=48, key="ecd_age")
        child_sex = st.selectbox("Child Sex*", ["Male", "Female"], key="ecd_sex")
        county = st.selectbox("County*", ["Nairobi", "Mombasa", "Kisumu", "Nakuru", "Eldoret", "Other"], key="ecd_county")
    
    with col2:
        caregiver_relation = st.selectbox("Your Relationship to Child*", ["Mother", "Father", "Guardian", "Other"])
        assessment_date = st.date_input("Assessment Date*", value=pd.Timestamp.now().date(), key="ecd_date")
    
    st.markdown("---")
    
    if ECD_QUESTIONS:
        st.subheader("📋 ECD Assessment Questions")
        responses = {}
        
        with st.expander("🏃 Physical Development (4 questions)", expanded=True):
            st.markdown("*Questions about motor skills and physical abilities*")
            for question in ECD_QUESTIONS['physical']['questions']:
                answer = st.radio(
                    f"**{question['id']}.** {question['question']}",
                    question['options'],
                    key=f"q_{question['id']}",
                    horizontal=True,
                    index=None
                )
                if answer:
                    responses[question['id']] = answer
        
        with st.expander("💬 Language Development (5 questions)", expanded=True):
            st.markdown("*Questions about communication and language skills*")
            for question in ECD_QUESTIONS['language']['questions']:
                answer = st.radio(
                    f"**{question['id']}.** {question['question']}",
                    question['options'],
                    key=f"q_{question['id']}",
                    horizontal=True,
                    index=None
                )
                if answer:
                    responses[question['id']] = answer
        
        with st.expander("📚 Literacy-Numeracy (5 questions)", expanded=True):
            st.markdown("*Questions about early learning and cognitive skills*")
            for question in ECD_QUESTIONS['literacy']['questions']:
                answer = st.radio(
                    f"**{question['id']}.** {question['question']}",
                    question['options'],
                    key=f"q_{question['id']}",
                    horizontal=True,
                    index=None
                )
                if answer:
                    responses[question['id']] = answer
        
        with st.expander("🤝 Socio-Emotional Development (6 questions)", expanded=True):
            st.markdown("*Questions about social interaction and emotional regulation*")
            for question in ECD_QUESTIONS['socio_emotional']['questions']:
                answer = st.radio(
                    f"**{question['id']}.** {question['question']}",
                    question['options'],
                    key=f"q_{question['id']}",
                    horizontal=True,
                    index=None
                )
                if answer:
                    responses[question['id']] = answer
        
        st.markdown("---")
        
        col1, col2, col3 = st.columns([2, 1, 2])
        with col2:
            submitted = st.button("📊 Calculate ECD Score", use_container_width=True, type="primary")
        
        if submitted:
            total_questions = sum(len(domain['questions']) for domain in ECD_QUESTIONS.values())
            
            if len(responses) < total_questions:
                st.error(f"⚠️  Please answer all {total_questions} questions before submitting. You've answered {len(responses)}.")
            else:
                with st.spinner("🔄 Calculating ECD scores..."):
                    if scoring_method == "local":
                        is_valid, message = validate_responses(responses, child_age)
                        if not is_valid:
                            st.error(f"❌ Validation error: {message}")
                        else:
                            ecd_result = calculate_ecd_score(responses, child_age)
                            display_ecd_results(ecd_result, child_age, assessment_date, caregiver_relation, county, responses, ECD_QUESTIONS)
                    elif scoring_method == "api":
                        if not api_healthy:
                            st.error("❌ API is not connected. Please start the Flask API first.")
                        else:
                            st.info("🔄 API-based scoring not yet implemented. Using local scoring instead.")
                            is_valid, message = validate_responses(responses, child_age)
                            if is_valid:
                                ecd_result = calculate_ecd_score(responses, child_age)
                                display_ecd_results(ecd_result, child_age, assessment_date, caregiver_relation, county, responses, ECD_QUESTIONS)
    else:
        st.error("❌ Could not load ECD questions. Please ensure src/ecd_questions.py exists and the API is running.")
        st.info("💡 Try restarting the dashboard or checking your project structure.")
    
    st.markdown("---")
    st.subheader("📚 About UNICEF ECDI2030")
    
    with st.expander("What is ECDI2030?"):
        st.markdown("""
The **Early Childhood Development Index 2030 (ECDI2030)** is a population-level measure
used to monitor progress towards **SDG Indicator 4.2.1**:

> *"Proportion of children under 5 years of age who are developmentally on track in health,
learning and psychosocial well-being"*

**Key Features:**
- 20 questions across 4 domains
- Age-specific thresholds (36-59 months)
- Takes 3-5 minutes to administer
- Suitable for household surveys and screenings
- Aligned with UNICEF and WHO guidelines
""")
    
    with st.expander("How is 'On Track' Determined?"):
        st.markdown("""
Children are classified as **"developmentally on track"** if they achieve the minimum
number of milestones for their age group:

| Age Range | Total Milestones Required |
|-----------|--------------------------|
| 36-41 months | ≥11 out of 20 |
| 42-47 months | ≥13 out of 20 |
| 48-59 months | ≥15 out of 20 |

**Domain-Specific Thresholds:**
- Physical: 3-4 milestones
- Language: 3-4 milestones
- Literacy-Numeracy: 3-4 milestones
- Socio-Emotional: 4-5 milestones
""")
    
    with st.expander("Important Notes"):
        st.markdown("""
⚠️ **This assessment is for screening purposes only:**
- Not a diagnostic tool
- Does not replace professional evaluation
- Results should be discussed with healthcare providers
- Early intervention is key if delays are identified

✅ **Best Practices:**
- Answer based on what child CAN do, not what they usually do
- Consider child's best performance
- Complete in a quiet, comfortable environment
- Allow child to demonstrate skills when possible
""")

# --------------------------
# INDIVIDUAL PREDICTION PAGE
# --------------------------
elif page == "🔮 Individual Prediction":
    st.markdown('<p class="main-header">🔮 Individual ECD Delay Prediction</p>', unsafe_allow_html=True)
    st.markdown("---")
    
    st.markdown("""
Enter child and household information to get an ECD delay prediction.
All fields marked with * are required.
""")
    
    with st.form("prediction_form"):
        col1, col2 = st.columns(2)
        
        with col1:
            child_age = st.number_input("Child Age (months)*", min_value=36, max_value=59, value=48)
            child_sex = st.selectbox("Child Sex*", ["Male", "Female"])
            mother_age = st.number_input("Mother Age (years)*", min_value=15, max_value=50, value=30)
            mother_education = st.selectbox("Mother's Education*", [
                "No education", "Primary", "Secondary", "Higher"
            ])
            mother_years_edu = st.number_input("Mother's Years of Education*", min_value=0, max_value=20, value=12)
            mother_working = st.selectbox("Mother Currently Working*", ["Yes", "No"])
        
        with col2:
            wealth_quintile = st.selectbox("Wealth Quintile*", [
                "Poorest", "Poorer", "Middle", "Richer", "Richest"
            ])
            urban_rural = st.selectbox("Urban/Rural*", ["Urban", "Rural"])
            county = st.selectbox("County*", [
                "Nairobi", "Mombasa", "Kisumu", "Nakuru", "Eldoret", "Other"
            ])
            household_members = st.number_input("Household Members*", min_value=1, max_value=20, value=5)
            has_electricity = st.selectbox("Has Electricity*", ["Yes", "No"])
            marital_status = st.selectbox("Marital Status*", [
                "Married", "Single", "Divorced", "Widowed"
            ])
        
        submitted = st.form_submit_button("🔮 Predict ECD Delay", use_container_width=True, type="primary")
        
        if submitted:
            if not api_healthy:
                st.error("❌ API is not connected. Please start the Flask API first.")
            else:
                with st.spinner("🔄 Making prediction..."):
                    features = {
                        "child_age_months": child_age,
                        "child_sex": child_sex,
                        "mother_age": mother_age,
                        "mother_education_level": mother_education,
                        "mother_years_education": mother_years_edu,
                        "mother_currently_working": mother_working,
                        "wealth_quintile": wealth_quintile,
                        "urban_rural": urban_rural,
                        "county": county,
                        "household_members": household_members,
                        "has_electricity": has_electricity,
                        "marital_status": marital_status,
                        "religion": "Christian",
                        "drinking_water_source": "Improved",
                        "toilet_facility": "Improved",
                        "cooking_fuel": "Clean"
                    }
                    
                    result = predict_delay(features)
                    
                    if "error" in result:
                        st.error(f"❌ Error: {result['error']}")
                    else:
                        prediction = result['prediction']
                        
                        col1, col2, col3 = st.columns(3)
                        
                        with col1:
                            if prediction['composite_delay'] == 1:
                                st.error("⚠️ DELAYED")
                            else:
                                st.success("✅ ON TRACK")
                        
                        with col2:
                            st.metric("Probability", f"{prediction['probability']:.1%}")
                        
                        with col3:
                            risk_level = prediction.get('risk_level', 'Unknown')
                            if risk_level == "High":
                                st.error("🔴 High Risk")
                            elif risk_level == "Medium":
                                st.warning("🟡 Medium Risk")
                            else:
                                st.success("🟢 Low Risk")
                        
                        st.markdown("---")
                        
                        if 'domain_predictions' in prediction and prediction['domain_predictions']:
                            st.subheader("📊 Domain-Specific Predictions")
                            domain_data = []
                            for domain, values in prediction['domain_predictions'].items():
                                if isinstance(values, dict) and 'probability' in values:
                                    domain_data.append({
                                        'Domain': domain.title(),
                                        'Delay': 'Yes' if values.get('delay', 0) == 1 else 'No',
                                        'Probability': f"{values.get('probability', 0):.1%}"
                                    })
                            
                            if domain_data:
                                domain_df = pd.DataFrame(domain_data)
                                st.dataframe(domain_df, use_container_width=True)
                        
                        st.caption(f"Model: {result.get('model_info', {}).get('model_name', 'Unknown')} | {result.get('timestamp', '')}")

# --------------------------
# COUNTY PROFILE PAGE (Drill-Down with Interactive Map)
# --------------------------
elif page == "📍 County Profile":
    st.markdown('<p class="main-header">📍 County Profile & Recommendations</p>', unsafe_allow_html=True)
    st.markdown("---")
    
    # Load county profiles
    county_profiles_path = Path(__file__).parent.parent / 'models' / 'county_profiles.json'
    if county_profiles_path.exists():
        with open(county_profiles_path, 'r') as f:
            county_profiles = json.load(f)
    else:
        county_profiles = None
    
    county_data = get_county_data()
    
    if county_profiles is None:
        st.warning("⚠️  County profiles not available. Run Notebook 05 to generate county_profiles.json")
        st.info("💡 Navigate to models/ folder and check if county_profiles.json exists")
    else:
        # County selector
        county_names = sorted([p['county'] for p in county_profiles])
        selected_county = st.selectbox(
            "📍 Select County",
            county_names,
            index=county_names.index('Nairobi') if 'Nairobi' in county_names else 0,
            key="county_profile_selector"
        )
        
        # Get county profile
        county_profile = next((p for p in county_profiles if p['county'] == selected_county), None)
        
        if county_profile:
            # Load county stats for additional metrics
            if county_data and len(county_data) > 0:
                county_df = pd.DataFrame(county_data)
                county_row = next((c for c in county_data if c.get('county') == selected_county), None)
                
                if county_row:
                    # =========================================================================
                    # HEADER SECTION
                    # =========================================================================
                    col1, col2, col3 = st.columns([2, 1, 1])
                    with col1:
                        st.subheader(f"📍 {selected_county} County")
                    with col2:
                        priority_color = {
                            'Critical': '🔴',
                            'High': '🟠',
                            'Medium': '🟡',
                            'Low': '🟢'
                        }.get(county_profile['overall_priority'], '⚪')
                        st.metric("Priority", f"{priority_color} {county_profile['overall_priority']}")
                    with col3:
                        st.metric("Delay Rate", f"{county_profile['delay_rate']:.1f}%")
                    
                    st.markdown("---")
                    
                    # =========================================================================
                    # INTERACTIVE MAP SECTION (NEW)
                    # =========================================================================
                    st.subheader("🗺️ Interactive County Map")
                    
                    # Load and display the interactive map
                    map_path = Path(__file__).parent.parent / 'reports' / 'outputs' / 'ecd_delay_county_map.html'
                    if map_path.exists():
                        with open(map_path, 'r', encoding='utf-8') as f:
                            map_html = f.read()
                        
                        # Display the map with highlighting note
                        st.components.v1.html(map_html, height=500, scrolling=False)
                        
                        st.info(f"💡 **Viewing:** {selected_county} County - The map shows ECD delay risk across all 47 counties. Click on any county to explore.")
                    else:
                        st.info("📍 Interactive map not found. Run Notebook 05 to generate it.")
                    
                    st.markdown("---")
                    
                    # =========================================================================
                    # KEY METRICS
                    # =========================================================================
                    st.subheader("📊 Key Metrics")
                    col1, col2, col3, col4 = st.columns(4)
                    with col1:
                        st.metric("Children Affected", f"{county_profile['estimated_children_affected']:,}")
                    with col2:
                        st.metric("Sample Size", f"{int(county_row.get('sample_size', 0)):,}")
                    with col3:
                        st.metric("Urban Proportion", f"{county_row.get('urban_proportion', 0)*100:.0f}%")
                    with col4:
                        st.metric("Wealth Index", f"{county_row.get('wealth_index', 0):.2f}")
                    
                    st.markdown("---")
                    
                    # =========================================================================
                    # DOMAIN-SPECIFIC DELAYS WITH NATIONAL COMPARISON
                    # =========================================================================
                    st.subheader("📊 Domain-Specific Delay Rates")
                    
                    # Calculate national averages
                    national_avgs = {
                        'physical_delay_pct': county_df['physical_delay_pct'].mean(),
                        'language_delay_pct': county_df['language_delay_pct'].mean(),
                        'literacy_delay_pct': county_df['literacy_delay_pct'].mean(),
                        'socio_emotional_delay_pct': county_df['socio_emotional_delay_pct'].mean()
                    }
                    
                    domain_cols = st.columns(4)
                    domains = [
                        ('🏃 Physical', 'physical_delay_pct', 'Physical Development'),
                        ('💬 Language', 'language_delay_pct', 'Language Development'),
                        ('📚 Literacy-Numeracy', 'literacy_delay_pct', 'Literacy-Numeracy'),
                        ('🤝 Socio-Emotional', 'socio_emotional_delay_pct', 'Socio-Emotional')
                    ]
                    
                    for i, (icon, domain_key, domain_name) in enumerate(domains):
                        with domain_cols[i]:
                            rate = county_row.get(domain_key, 0)
                            national_avg = national_avgs.get(domain_key, 0)
                            delta = rate - national_avg
                            
                            # Determine status with color coding
                            if delta > 5:
                                status = "🔴 Above National"
                                border_color = '#dc3545'
                            elif delta < -5:
                                status = "🟢 Below National"
                                border_color = '#28a745'
                            else:
                                status = "🟡 Near National"
                                border_color = '#ffc107'
                            
                            html = f"""
                            <div class="metric-card" style="border-left-color: {border_color};">
                                <h3>{icon} {domain_name}</h3>
                                <p style="font-size: 1.5rem; font-weight: bold; color: {border_color};">
                                    {rate:.1f}%
                                </p>
                                <p style="font-size: 0.85rem;">
                                    National Avg: {national_avg:.1f}% | 
                                    Delta: <span style="color: {'red' if delta > 0 else 'green'}; font-weight: bold;">
                                        {delta:+.1f}%
                                    </span>
                                </p>
                                <p style="font-size: 0.8rem; margin-top: 0.3rem;">{status}</p>
                            </div>
                            """
                            st.markdown(html, unsafe_allow_html=True)
                    
                    # =========================================================================
                    # VISUALIZATION: County vs National Comparison Chart
                    # =========================================================================
                    st.markdown("---")
                    st.subheader("📈 County vs National Average Comparison")
                    
                    # Prepare data for visualization
                    comparison_data = []
                    for icon, domain_key, domain_name in domains:
                        comparison_data.append({
                            'Domain': domain_name,
                            'County Rate': county_row.get(domain_key, 0),
                            'National Average': national_avgs.get(domain_key, 0)
                        })
                    
                    comp_df = pd.DataFrame(comparison_data)
                    
                    fig, ax = plt.subplots(figsize=(10, 6))
                    x = np.arange(len(comp_df))
                    width = 0.35
                    
                    bars1 = ax.bar(x - width/2, comp_df['County Rate'], width,
                                  label=f'{selected_county}', color='#2E86AB', edgecolor='black')
                    bars2 = ax.bar(x + width/2, comp_df['National Average'], width,
                                  label='National Average', color='#A23B72', edgecolor='black')
                    
                    ax.set_xlabel('ECD Domain', fontsize=11)
                    ax.set_ylabel('Delay Rate (%)', fontsize=11)
                    ax.set_title(f'{selected_county} vs National Average',
                                fontsize=13, fontweight='bold')
                    ax.set_xticks(x)
                    ax.set_xticklabels([d.split()[0] for d in comp_df['Domain']], fontsize=10)
                    ax.legend()
                    ax.grid(axis='y', alpha=0.3, linestyle='--')
                    
                    # Add value labels
                    for bar in bars1:
                        height = bar.get_height()
                        ax.text(bar.get_x() + bar.get_width()/2., height,
                               f'{height:.1f}%', ha='center', va='bottom', fontsize=9)
                    for bar in bars2:
                        height = bar.get_height()
                        ax.text(bar.get_x() + bar.get_width()/2., height,
                               f'{height:.1f}%', ha='center', va='bottom', fontsize=9)
                    
                    plt.tight_layout()
                    st.pyplot(fig)
                    
                    st.markdown("---")
                    
                    # =========================================================================
                    # RECOMMENDATIONS
                    # =========================================================================
                    st.subheader("💡 Policy Recommendations")
                    for i, rec in enumerate(county_profile['recommendations'], 1):
                        with st.expander(f"{i}. {rec['priority']}: {rec['intervention']}", expanded=(i<=2)):
                            st.write(f"**Rationale:** {rec['rationale']}")
                    
                    st.markdown("---")
                    
                    # =========================================================================
                    # KEY RISK FACTORS
                    # =========================================================================
                    st.subheader("⚠️ Key Risk Factors")
                    risk_factors = county_profile['key_risk_factors']
                    col1, col2, col3 = st.columns(3)
                    with col1:
                        st.info(f"**Wealth:** {risk_factors['wealth']}")
                    with col2:
                        st.info(f"**Urbanization:** {risk_factors['urbanization']}")
                    with col3:
                        st.warning(f"**Highest Domain Delay:** {risk_factors['highest_domain_delay'].title()} ({risk_factors['highest_domain_rate']:.1f}%)")
                    
                    st.markdown("---")
                    
                    # =========================================================================
                    # DOWNLOAD COUNTY REPORT
                    # =========================================================================
                    st.subheader("💾 Download County Report")
                    county_report = {
                        'county': selected_county,
                        'overall_priority': county_profile['overall_priority'],
                        'delay_rate': county_profile['delay_rate'],
                        'estimated_children_affected': county_profile['estimated_children_affected'],
                        'sample_size': int(county_row.get('sample_size', 0)),
                        'urban_proportion': float(county_row.get('urban_proportion', 0)),
                        'wealth_index': float(county_row.get('wealth_index', 0)),
                        'domain_delays': {
                            'physical': float(county_row.get('physical_delay_pct', 0)),
                            'language': float(county_row.get('language_delay_pct', 0)),
                            'literacy': float(county_row.get('literacy_delay_pct', 0)),
                            'socio_emotional': float(county_row.get('socio_emotional_delay_pct', 0))
                        },
                        'national_averages': national_avgs,
                        'recommendations': county_profile['recommendations'],
                        'key_risk_factors': risk_factors
                    }
                    county_report_json = json.dumps(county_report, indent=2)
                    st.download_button(
                        label="📥 Download County Profile (JSON)",
                        data=county_report_json,
                        file_name=f"{selected_county.replace(' ', '_')}_ecd_profile.json",
                        mime="application/json",
                        use_container_width=True
                    )
                else:
                    st.warning(f"⚠️  County data not found for {selected_county}")
            else:
                st.info("ℹ️  County data not available. Run Notebook 05 to generate it.")
        else:
            st.warning(f"⚠️  County profile not found for {selected_county}")

# --------------------------
# MODEL PERFORMANCE PAGE
# --------------------------
elif page == "📈 Model Performance":
    st.markdown('<p class="main-header">📈 Model Performance & Information</p>', unsafe_allow_html=True)
    st.markdown("---")
    
    if not api_healthy:
        st.error("❌ API is not connected. Please start the Flask API first.")
        st.info("💡 Run: `python api/main.py` in a separate terminal")
    else:
        model_info = get_model_info()
        
        if model_info is None:
            st.warning("⚠️  Could not load model information from API")
        elif model_info.get('status') != 'success':
            st.error(f"❌ API Error: {model_info.get('message', 'Unknown error')}")
        else:
            selected = model_info.get('selected_model', {})
            
            # KPI Cards
            st.markdown('<div class="section-header">📊 Model Performance Overview</div>', unsafe_allow_html=True)
            col1, col2, col3, col4 = st.columns(4)
            
            with col1:
                st.metric(
                    label="🏆 Best Model",
                    value=selected.get('model_name', 'N/A'),
                    delta=f"Type: {selected.get('model_type', 'N/A')}"
                )
            
            with col2:
                st.metric(
                    label="📈 ROC-AUC Score",
                    value=f"{float(selected.get('roc_auc', 0)):.4f}",
                    delta="Test Set Performance"
                )
            
            with col3:
                st.metric(
                    label="🎯 F1-Score",
                    value=f"{float(selected.get('f1_score', 0)):.4f}",
                    delta="Harmonic Mean"
                )
            
            with col4:
                st.metric(
                    label="📚 Study Scope",
                    value="36-59 months",
                    delta="UNICEF ECDI2030"
                )
            
            st.markdown("---")
            
            # All Models Comparison
            st.markdown('<div class="section-header">📊 All Models Comparison</div>', unsafe_allow_html=True)
            all_models = model_info.get('all_models', {})
            
            if all_models:
                model_data = []
                for name, info in all_models.items():
                    if isinstance(info, dict):
                        model_data.append({
                            'Model': name,
                            'Type': info.get('model_type', 'N/A'),
                            'ROC-AUC': f"{float(info.get('roc_auc', 0)):.4f}" if info.get('roc_auc') else 'N/A',
                            'F1-Score': f"{float(info.get('f1_score', 0)):.4f}" if info.get('f1_score') else 'N/A'
                        })
                
                if model_data:
                    model_df = pd.DataFrame(model_data)
                    model_df['ROC-AUC Numeric'] = pd.to_numeric(model_df['ROC-AUC'], errors='coerce')
                    model_df = model_df.sort_values('ROC-AUC Numeric', ascending=False)
                    
                    # Display table
                    st.dataframe(
                        model_df[['Model', 'Type', 'ROC-AUC', 'F1-Score']],
                        use_container_width=True,
                        height=300
                    )
                    
                    # Visualization
                    fig, ax = plt.subplots(figsize=(10, 6))
                    valid_models = model_df[model_df['ROC-AUC Numeric'].notna()]
                    colors = ['coral' if t == 'Ensemble' else 'steelblue'
                             for t in valid_models['Type']]
                    
                    ax.barh(valid_models['Model'], valid_models['ROC-AUC Numeric'],
                           color=colors, edgecolor='black')
                    ax.set_xlabel('ROC-AUC Score', fontsize=11)
                    ax.set_title('Model Performance Comparison (ROC-AUC)',
                                fontsize=13, fontweight='bold')
                    ax.set_xlim(0.5, 1.0)
                    ax.grid(axis='x', alpha=0.3)
                    
                    for i, v in enumerate(valid_models['ROC-AUC Numeric']):
                        ax.text(v + 0.01, i, f'{v:.3f}', va='center', fontsize=9)
                    
                    plt.tight_layout()
                    st.pyplot(fig)
                    
                    # Download
                    csv = model_df.to_csv(index=False).encode('utf-8')
                    st.download_button(
                        label="📥 Download Model Comparison (CSV)",
                        data=csv,
                        file_name="model_comparison_all.csv",
                        mime="text/csv",
                        use_container_width=True
                    )
            
            st.markdown("---")
            
            # Plain-Language Summary
            st.markdown('<div class="section-header">📖 Model Summary</div>', unsafe_allow_html=True)
            st.markdown(f"""
### Model Performance Summary

Our **{selected.get('model_name', 'AdaBoost')}** model achieves:

- **ROC-AUC of {float(selected.get('roc_auc', 0)):.3f}**: Good ability to distinguish between children who are developmentally on track versus those at risk.
- **F1-Score of {float(selected.get('f1_score', 0)):.3f}**: Balanced performance in identifying both on-track and delayed children.
- **Trained on KDHS 2022 data**: The model learned from 3,599 children aged 36-59 months.

### What This Means

✅ **Good Performance**: The model can reliably predict which children may need additional support  
✅ **Balanced**: The model doesn't favor one class over another  
✅ **Ready for Deployment**: The model has been thoroughly tested and validated

### How to Use

1. Enter child and household information in the **Individual Prediction** page
2. The model will predict whether the child is "On Track" or "Delayed"
3. Review the probability and risk level
4. Use domain-specific predictions to identify areas needing support

⚠️ **Important**: This is a screening tool, not a diagnostic tool. Always consult healthcare professionals for formal assessments.
""")

# --------------------------
# FEATURE IMPORTANCE PAGE
# --------------------------
elif page == "📈 Feature Importance":
    st.markdown('<p class="main-header">📈 Feature Importance (SHAP Analysis)</p>', unsafe_allow_html=True)
    st.markdown("---")
    
    if not api_healthy:
        st.error("❌ API is not connected.")
    else:
        feature_data = get_feature_importance()
        
        if feature_data and feature_data.get('status') == 'success':
            top_features = feature_data.get('top_10_features', [])
            importance = feature_data.get('feature_importance', [])
            
            # Top Features Section
            st.markdown('<div class="section-header">🏆 Top 10 Most Important Features</div>', unsafe_allow_html=True)
            if top_features:
                for i, feature in enumerate(top_features[:10], 1):
                    st.write(f"{i}. **{feature.replace('_', ' ').title()}**")
            
            st.markdown("---")
            
            # Display SHAP Beeswarm Plot
            st.markdown('<div class="section-header">🐝 SHAP Beeswarm Plot</div>', unsafe_allow_html=True)
            st.markdown("""
The beeswarm plot shows how each feature impacts the model's output:

- **X-axis**: SHAP value (impact on model prediction)
- **Color**: Feature value (red = high, blue = low)
- **Position**: Higher SHAP values = more important
- **Right side**: Increases delay risk
- **Left side**: Decreases delay risk
""")
            
            beeswarm_path = Path(__file__).parent.parent / 'reports' / 'figures' / 'shap_beeswarm_plot.png'
            if beeswarm_path.exists():
                st.image(str(beeswarm_path), use_column_width=True)
                st.success("✅ SHAP beeswarm plot loaded successfully")
            else:
                st.info("📍 SHAP beeswarm plot not found. Run Notebook 04 to generate it.")
                st.code("""
To generate SHAP plots:
1. Run Notebook 04: notebooks/04_shap_lime_explainability.ipynb
2. The plot will be saved to: reports/figures/shap_beeswarm_plot.png
""", language="bash")
            
            st.markdown("---")
            
            # SHAP Bar Plot
            st.markdown('<div class="section-header">📊 SHAP Feature Importance (Bar Chart)</div>', unsafe_allow_html=True)
            bar_path = Path(__file__).parent.parent / 'reports' / 'figures' / 'shap_bar_plot.png'
            if bar_path.exists():
                st.image(str(bar_path), use_column_width=True)
            else:
                st.info("📍 SHAP bar plot not found.")
            
            st.markdown("---")
            
            # Detailed Feature Importance Table
            if importance:
                st.markdown('<div class="section-header">📋 Detailed Feature Importance</div>', unsafe_allow_html=True)
                importance_df = pd.DataFrame(importance)
                importance_df = importance_df.sort_values('SHAP_Importance', ascending=False)
                
                # Display top 15 in a better format
                st.dataframe(
                    importance_df.head(15).style.format({
                        'SHAP_Importance': '{:.6f}'
                    }),
                    use_container_width=True,
                    height=400
                )
                
                # Download button
                csv = importance_df.to_csv(index=False).encode('utf-8')
                st.download_button(
                    label="📥 Download Feature Importance (CSV)",
                    data=csv,
                    file_name="shap_feature_importance.csv",
                    mime="text/csv",
                    use_container_width=True
                )
            
            st.markdown("---")
            
            # Interpretation Guide
            st.markdown('<div class="section-header">📖 Interpretation Guide</div>', unsafe_allow_html=True)
            with st.expander("SHAP Interpretation Guide"):
                st.markdown("""
**SHAP (SHapley Additive exPlanations)** measures feature contribution:

**Beeswarm Plot:**
- Each dot represents one prediction
- Red dots = high feature values
- Blue dots = low feature values
- Right side = positive impact (increases delay risk)
- Left side = negative impact (decreases delay risk)

**Key Findings:**
1. **Mother Years Education**: More education → Lower delay risk
2. **Wealth Quintile**: Higher wealth → Lower delay risk
3. **Child Age Months**: Older children → Lower delay risk
4. **County**: Geographic location affects access to services
5. **Mother Education Level**: Formal education matters

**Policy Implications:**
- Target interventions to low-education households
- Prioritize resource allocation to high-risk counties
- Focus on maternal education programs
- Address wealth-based disparities
""")

# --------------------------
# ABOUT PAGE
# --------------------------
elif page == "ℹ️ About":
    st.markdown('<p class="main-header">ℹ️ About This Project</p>', unsafe_allow_html=True)
    st.markdown("---")
    
    st.markdown("""
## KDHS 2022 ECD Delay Prediction

This project predicts **Early Childhood Development (ECD) Delay** for children
aged **36–59 months** using data from the **Kenya Demographic and Health Survey (KDHS) 2022**.

---

### Methodology

- **Target Definition:** UNICEF ECDI2030 with age-specific thresholds
- **Age Groups:**
  - 36–41 months: ≥11 milestones = On Track
  - 42–47 months: ≥13 milestones = On Track
  - 48–59 months: ≥15 milestones = On Track
- **Domains:** Physical, Language, Literacy, Socio-emotional
- **SDG Indicator:** 4.2.1

---

### Model Development

- Multiple models trained (Random Forest, XGBoost, LightGBM, Ensemble)
- Hyperparameter tuning with Optuna
- 5-fold stratified cross-validation
- SHAP + LIME for explainability
- Geospatial analysis for county-level insights

---

### Technology Stack

- **Data Processing:** Pandas, NumPy, PyReadStat
- **Machine Learning:** Scikit-learn, XGBoost, LightGBM
- **Explainability:** SHAP, LIME
- **Geospatial:** GeoPandas, Folium
- **Backend:** Flask API
- **Frontend:** Streamlit Dashboard

---

### Project Structure
""")

# ============================================================================
# Run Dashboard
# ============================================================================
if __name__ == "__main__":
    pass