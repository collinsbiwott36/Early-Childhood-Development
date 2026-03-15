# ============================================================================
# api/main.py
# Flask API for KDHS ECD Delay Prediction
# ============================================================================
"""
Endpoints:
  GET  /                  - Health check & API info
  POST /predict           - Predict ECD delay (demographic input)
  POST /ecd-assess        - Calculate ECDI2030 score (20 questions)
  GET  /features          - Get SHAP feature importance
  GET  /model-info        - Get model information & metrics
  GET  /health            - Detailed health check
  GET  /county-profile    - Get county-specific insights

Project: KDHS 2022 ECD Analysis
Methodology: UNICEF ECDI2030 (SDG 4.2.1)
Age Range: 36-59 months
"""

from flask import Flask, request, jsonify
from flask_cors import CORS
from pathlib import Path
import joblib
import json
import pandas as pd
import numpy as np
from datetime import datetime
import sys
import logging

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# ============================================================================
# Initialize Flask App
# ============================================================================
app = Flask(__name__)
CORS(app)  # Enable cross-origin requests for Streamlit

# ============================================================================
# Configuration & Paths
# ============================================================================
PROJECT_ROOT = Path(__file__).parent.parent
MODELS_DIR = PROJECT_ROOT / "models"
TABLES_DIR = PROJECT_ROOT / "reports" / "tables"

# ============================================================================
# Global Variables (Loaded at Startup)
# ============================================================================
# Model & preprocessing
model = None
scaler = None
feature_cols = None
pipeline = None
encoders = {}

# Explainability & metadata
shap_summary = None
model_registry = None
geo_summary = None
county_data = None
domain_models = {}

# ============================================================================
# Load Resources at Startup
# ============================================================================
def load_resources():
    """Load all models, encoders, and metadata at application startup"""
    global model, scaler, feature_cols, pipeline, encoders
    global shap_summary, model_registry, geo_summary, county_data, domain_models

    logger.info("Loading models and resources...")

    # --- Load composite prediction pipeline ---
    try:
        pipeline_path = MODELS_DIR / "prediction_pipeline_final.pkl"
        if not pipeline_path.exists():
            logger.error(f"Pipeline file not found: {pipeline_path}")
            return False
        pipeline = joblib.load(pipeline_path)
        model = pipeline["model"]
        scaler = pipeline.get("scaler", None)
        feature_cols = pipeline["feature_cols"]
        logger.info(f"Loaded composite model: {pipeline.get('model_name', 'Unknown')}")
    except Exception as e:
        logger.error(f"Error loading composite model: {e}")
        return False

    # --- Load feature encoders ---
    try:
        encoders_path = MODELS_DIR / "feature_encoders.pkl"
        if encoders_path.exists():
            encoders = joblib.load(encoders_path)
            logger.info(f"Loaded {len(encoders)} feature encoders")
        else:
            logger.warning(f"Encoders file not found: {encoders_path}")
    except Exception as e:
        logger.warning(f"Error loading encoders: {e}")

    # --- Load SHAP summary ---
    try:
        shap_path = MODELS_DIR / "shap_summary.json"
        if shap_path.exists():
            with open(shap_path, "r") as f:
                shap_summary = json.load(f)
            n_features = len(shap_summary.get("feature_importance", []))
            logger.info(f"Loaded SHAP summary: {n_features} features")
        else:
            logger.warning(f"SHAP summary not found: {shap_path}")
    except Exception as e:
        logger.warning(f"Error loading SHAP summary: {e}")

    # --- Load model registry ---
    try:
        registry_path = MODELS_DIR / "model_registry.json"
        if registry_path.exists():
            with open(registry_path, "r") as f:
                model_registry = json.load(f)
            logger.info("Loaded model registry")
        else:
            logger.warning(f"Model registry not found: {registry_path}")
    except Exception as e:
        logger.warning(f"Error loading model registry: {e}")

    # --- Load geo summary for county insights ---
    try:
        geo_path = MODELS_DIR / "geo_summary.json"
        if geo_path.exists():
            with open(geo_path, "r") as f:
                geo_summary = json.load(f)
            n_counties = geo_summary.get("counties_with_data", 0)
            logger.info(f"Loaded geo summary: {n_counties} counties")
        else:
            logger.warning(f"Geo summary not found: {geo_path}")
    except Exception as e:
        logger.warning(f"Error loading geo summary: {e}")

    # --- Load domain models ---
    try:
        domain_path = MODELS_DIR / "domain_models.pkl"
        if domain_path.exists():
            domain_models = joblib.load(domain_path)
            logger.info(f"Loaded {len(domain_models)} domain models: {list(domain_models.keys())}")
        else:
            logger.warning(f"Domain models not found: {domain_path}")
    except Exception as e:
        logger.warning(f"Error loading domain models: {e}")

    # --- Load county dashboard data ---
    try:
        county_csv = TABLES_DIR / "county_dashboard_data.csv"
        if county_csv.exists():
            county_data = pd.read_csv(county_csv).to_dict("records")
            logger.info(f"Loaded county dashboard data: {len(county_data)} counties")
        else:
            logger.warning(f"County data not found: {county_csv}")
    except Exception as e:
        logger.warning(f"Error loading county data: {e}")

    logger.info("All resources loaded successfully!\n")
    return True


# Load resources once at startup
if not load_resources():
    logger.error("Failed to load required resources. API may not function correctly.")

# ============================================================================
# Helper Functions
# ============================================================================

def validate_prediction_input(data: dict) -> tuple:
    """
    Validate input data for demographic-based prediction

    Args:
        data: Dictionary containing prediction features

    Returns:
        Tuple of (is_valid: bool, message: str, processed_data: dict or None)
    """
    if not isinstance(data, dict):
        return False, "Payload must be a JSON object", None

    required_features = [
        "child_age_months",
        "child_sex",
        "mother_age",
        "mother_education_level",
        "mother_years_education",
        "wealth_quintile",
        "urban_rural",
        "county",
    ]

    missing = [f for f in required_features if f not in data]
    if missing:
        return False, f"Missing required fields: {missing}", None

    # Validate numeric ranges
    age = data.get("child_age_months", 0)
    if not (36 <= age <= 59):
        return False, "child_age_months must be between 36 and 59", None

    mother_age = data.get("mother_age", 0)
    if not (15 <= mother_age <= 50):
        return False, "mother_age must be between 15 and 50", None

    return True, "Valid", data


def preprocess_prediction_input(data: dict) -> pd.DataFrame:
    """
    Preprocess input data for model prediction

    Args:
        data: Dictionary containing raw input features

    Returns:
        DataFrame with processed features matching model's expected columns
    """
    df = pd.DataFrame([data])

    # Create a copy of columns to avoid modification during iteration
    columns_to_process = list(df.columns)
    
    # Apply encoders for categorical features
    for col in columns_to_process:
        if col in encoders:
            encoder_info = encoders[col]
            mapping = encoder_info.get("mapping", {})
            # Map categorical values to encoded values
            df[col] = df[col].map(mapping).fillna(-1)  # -1 for unknown categories
            # Rename to match model's expected column names
            df.rename(columns={col: f"{col}_encoded"}, inplace=True)

    # Handle numeric features
    numeric_features = [
        "child_age_months",
        "mother_age",
        "mother_years_education",
        "household_members",
    ]
    for col in numeric_features:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce").fillna(0)

    # Ensure all feature columns are present
    if feature_cols:
        for col in feature_cols:
            if col not in df.columns:
                df[col] = 0  # Default value for missing features

    # Ensure correct column order (must match training data)
    if feature_cols:
        df = df[feature_cols]

    # Apply scaling if needed
    if scaler is not None and feature_cols:
        df_values = scaler.transform(df)
        return pd.DataFrame(df_values, columns=feature_cols)

    return df


def get_domain_predictions(X: pd.DataFrame) -> dict:
    """
    Get predictions from all domain-specific models

    Args:
        X: Preprocessed feature DataFrame

    Returns:
        Dictionary with domain-specific predictions
    """
    domain_predictions = {}

    if not domain_models:
        logger.warning("Domain models not loaded")
        return {"error": "Domain models not loaded"}

    for domain_name, domain_model in domain_models.items():
        try:
            pred = domain_model.predict(X)[0]
            if hasattr(domain_model, "predict_proba"):
                proba = float(domain_model.predict_proba(X)[0, 1])
            else:
                proba = float(pred)

            domain_predictions[domain_name] = {
                "delay": int(pred),
                "probability": round(proba, 3),
            }
        except Exception as e:
            logger.error(f"Error predicting domain {domain_name}: {e}")
            domain_predictions[domain_name] = {"error": str(e)}

    return domain_predictions


def fetch_county_profile(county_name: str) -> dict:
    """
    Fetch county-specific insights from geo_summary and county_data

    Args:
        county_name: Name of the county

    Returns:
        Dictionary with county profile information
    """
    if not geo_summary or not county_data:
        return {"error": "County data not available"}

    # Find county in dashboard data (case-insensitive)
    county_info = next(
        (c for c in county_data if c.get("county", "").lower() == county_name.lower()),
        None,
    )

    if not county_info:
        return {"error": f"County '{county_name}' not found in dataset"}

    # Build profile
    profile = {
        "county": county_info.get("county"),
        "risk_category": county_info.get("risk_category", "Unknown"),
        "delay_rate_pct": county_info.get("delay_rate_pct", 0),
        "sample_size": county_info.get("sample_size", 0),
        "n_delayed": county_info.get("n_delayed", 0),
        "recommendations": [],
    }

    # Add recommendations based on risk category
    if profile["risk_category"] == "High Risk":
        profile["recommendations"].append(
            {
                "priority": "CRITICAL",
                "intervention": "Immediate multi-sectoral ECD intervention required",
                "rationale": f"Delay rate ({profile['delay_rate_pct']:.1f}%) exceeds national threshold",
            }
        )
    elif profile["risk_category"] == "Medium Risk":
        profile["recommendations"].append(
            {
                "priority": "HIGH",
                "intervention": "Targeted ECD resource allocation",
                "rationale": f"Delay rate ({profile['delay_rate_pct']:.1f}%) above national average",
            }
        )

    # Add wealth-based recommendation
    if county_info.get("wealth_index", 3) <= 2.5:
        profile["recommendations"].append(
            {
                "priority": "Economic",
                "intervention": "Conditional cash transfers for ECD",
                "rationale": "Low wealth index limits access to ECD services",
            }
        )

    # Add urbanization recommendation
    if county_info.get("urban_proportion", 1) < 0.3:
        profile["recommendations"].append(
            {
                "priority": "Rural Outreach",
                "intervention": "Mobile ECD centres and community health workers",
                "rationale": "Highly rural with limited infrastructure",
            }
        )

    return profile


# ============================================================================
# API Endpoints
# ============================================================================

@app.route("/", methods=["GET"])
def root():
    """Health check & API information endpoint"""
    return jsonify(
        {
            "status": "ok",
            "message": "KDHS ECD Delay Prediction API",
            "version": "1.0.0",
            "endpoints": [
                "GET  /",
                "POST /predict",
                "POST /ecd-assess",
                "GET  /features",
                "GET  /model-info",
                "GET  /health",
                "GET  /county-profile",
            ],
            "timestamp": datetime.now().isoformat(),
        }
    )


@app.route("/predict", methods=["POST"])
def predict_delay():
    """
    Predict ECD delay for a child using demographic features

    Request JSON:
    {
        "child_age_months": 48,
        "child_sex": "Male",
        "mother_age": 32,
        "mother_education_level": "Secondary",
        "mother_years_education": 12,
        "wealth_quintile": "Middle",
        "urban_rural": "Urban",
        "county": "Nairobi"
    }

    Response JSON:
    {
        "status": "success",
        "prediction": {
            "composite_delay": 0,
            "probability": 0.23,
            "risk_level": "Low",
            "domain_predictions": {...},
            "top_shap_features": [...]
        },
        "model_info": {...},
        "timestamp": "..."
    }
    """
    try:
        # Validate input
        is_valid, message, data = validate_prediction_input(request.json)
        if not is_valid:
            logger.warning(f"Invalid prediction input: {message}")
            return jsonify({"status": "error", "message": message}), 400

        # Preprocess input
        X = preprocess_prediction_input(data)

        # Check if model is loaded
        if model is None:
            logger.error("Composite model not loaded")
            return jsonify({"status": "error", "message": "Composite model not loaded"}), 500

        # Make composite prediction
        prediction = int(model.predict(X)[0])
        if hasattr(model, "predict_proba"):
            probability = float(model.predict_proba(X)[0, 1])
        else:
            probability = float(prediction)

        # Determine risk level
        if probability >= 0.7:
            risk_level = "High"
        elif probability >= 0.4:
            risk_level = "Medium"
        else:
            risk_level = "Low"

        # Get domain predictions
        domain_predictions = get_domain_predictions(X)

        # Get top SHAP features
        top_shap_features = []
        if shap_summary and shap_summary.get("top_10_features"):
            top_shap_features = shap_summary["top_10_features"][:5]

        # Prepare response
        response = {
            "status": "success",
            "prediction": {
                "composite_delay": prediction,
                "probability": round(probability, 3),
                "risk_level": risk_level,
                "domain_predictions": domain_predictions,
                "top_shap_features": top_shap_features,
            },
            "input": {k: str(v) for k, v in data.items()},
            "model_info": {
                "model_name": pipeline.get("model_name", "Unknown") if pipeline else "Unknown",
                "study_scope": pipeline.get("study_scope", "36-59 months") if pipeline else "Unknown",
                "methodology": pipeline.get("methodology", "UNICEF ECDI2030") if pipeline else "Unknown",
            },
            "timestamp": datetime.now().isoformat(),
        }

        logger.info(f"Prediction successful: probability={probability:.3f}, risk={risk_level}")
        return jsonify(response), 200

    except Exception as e:
        logger.error(f"Error in predict endpoint: {e}", exc_info=True)
        return jsonify(
            {
                "status": "error",
                "message": str(e),
                "error_type": type(e).__name__,
            }
        ), 500


@app.route("/ecd-assess", methods=["POST"])
def ecd_assessment():
    """
    Calculate ECDI2030 score from 20 assessment questions

    Request JSON:
    {
        "child_age_months": 48,
        "responses": {
            "phys_walk_uneven": "Yes",
            "lang_words_10plus": "Yes",
            ...
        }
    }
    """
    try:
        data = request.json

        if not data or "responses" not in data or "child_age_months" not in data:
            return jsonify(
                {
                    "status": "error",
                    "message": "Missing 'responses' or 'child_age_months'",
                }
            ), 400

        responses = data["responses"]
        age_months = data["child_age_months"]

        # Import ECD questions module
        try:
            from src.ecd_questions import calculate_ecd_score, validate_responses
        except ImportError as e:
            logger.error(f"Could not import ECD questions module: {e}")
            return jsonify(
                {
                    "status": "error",
                    "message": f"Could not import ECD questions module: {str(e)}. Ensure src/ecd_questions.py exists.",
                    "error_type": "ImportError",
                }
            ), 500

        # Validate responses
        is_valid, message = validate_responses(responses, age_months)
        if not is_valid:
            return jsonify({"status": "error", "message": message}), 400

        # Calculate ECD score
        ecd_result = calculate_ecd_score(responses, age_months)

        # Prepare response
        response = {
            "status": "success",
            "assessment": {
                "scores": ecd_result["scores"],
                "percentages": ecd_result["percentages"],
                "on_track": ecd_result["on_track"],
                "threshold": ecd_result["threshold"],
                "age_months": age_months,
            },
            "timestamp": datetime.now().isoformat(),
        }

        return jsonify(response), 200

    except Exception as e:
        logger.error(f"Error in ecd-assess endpoint: {e}", exc_info=True)
        return jsonify(
            {
                "status": "error",
                "message": str(e),
                "error_type": type(e).__name__,
            }
        ), 500


@app.route("/features", methods=["GET"])
def get_feature_importance():
    """Get feature importance from SHAP analysis"""
    if shap_summary is None:
        return jsonify(
            {
                "status": "error",
                "message": "SHAP summary not available",
            }
        ), 404

    return jsonify(
        {
            "status": "success",
            "feature_importance": shap_summary.get("feature_importance", []),
            "top_10_features": shap_summary.get("top_10_features", []),
            "model_name": shap_summary.get("model_name", "Unknown"),
            "generated_at": shap_summary.get("generated_at", "Unknown"),
            "n_features": len(shap_summary.get("feature_importance", [])),
        }
    )


@app.route("/model-info", methods=["GET"])
def get_model_information():
    """Get model information and performance metrics"""
    if model_registry is None:
        return jsonify(
            {
                "status": "error",
                "message": "Model registry not available",
            }
        ), 404

    # Include domain models in response
    domain_models_info = {}
    if model_registry and "domain_models" in model_registry:
        domain_models_info = model_registry["domain_models"]

    # Include SHAP summary info
    shap_info = None
    if shap_summary:
        shap_info = {
            "model_name": shap_summary.get("model_name", "Unknown"),
            "top_10_features": shap_summary.get("top_10_features", []),
            "generated_at": shap_summary.get("generated_at", "Unknown"),
        }

    return jsonify(
        {
            "status": "success",
            "selected_model": model_registry.get("selected_model", {}),
            "all_models": model_registry.get("all_models", {}),
            "domain_models": domain_models_info,
            "shap_summary": shap_info,
        }
    )


@app.route("/county-profile", methods=["GET"])
def get_county_profile_endpoint():
    """
    Get county-specific insights and recommendations

    Query params:
        county: County name (e.g., "Nairobi")

    Response:
    {
        "status": "success",
        "county_profile": {
            "county": "Nairobi",
            "risk_category": "Low Risk",
            "delay_rate_pct": 14.2,
            "sample_size": 125,
            "n_delayed": 18,
            "recommendations": [...]
        }
    }
    """
    try:
        county_name = request.args.get("county", "").strip()

        if not county_name:
            return jsonify(
                {
                    "status": "error",
                    "message": "Missing 'county' query parameter",
                }
            ), 400

        # Fetch county profile
        profile = fetch_county_profile(county_name)

        if "error" in profile:
            return jsonify(
                {
                    "status": "error",
                    "message": profile["error"],
                }
            ), 404

        return jsonify(
            {
                "status": "success",
                "county_profile": profile,
            }
        ), 200

    except Exception as e:
        logger.error(f"Error in county-profile endpoint: {e}", exc_info=True)
        return jsonify(
            {
                "status": "error",
                "message": str(e),
                "error_type": type(e).__name__,
            }
        ), 500


@app.route("/health", methods=["GET"])
def health_check():
    """Detailed health check for all API components"""
    return jsonify(
        {
            "status": "healthy",
            "components": {
                "composite_model": model is not None,
                "domain_models": len(domain_models) > 0,
                "scaler": scaler is not None,
                "encoders": len(encoders) > 0,
                "feature_cols": len(feature_cols) if feature_cols else 0,
                "shap_summary": shap_summary is not None,
                "model_registry": model_registry is not None,
                "geo_summary": geo_summary is not None,
                "county_data": county_data is not None,
            },
            "timestamp": datetime.now().isoformat(),
        }
    )


# ============================================================================
# Run API Server
# ============================================================================
if __name__ == "__main__":
    logger.info("\n" + "=" * 70)
    logger.info("Starting KDHS ECD Delay Prediction API")
    logger.info("=" * 70)
    logger.info(f"Project Root: {PROJECT_ROOT}")
    logger.info(f"Models Dir: {MODELS_DIR}")
    logger.info("API will run on: http://127.0.0.1:5000")
    logger.info("Endpoints:")
    logger.info("   GET  /               - Health check & API info")
    logger.info("   POST /predict        - Demographic-based prediction")
    logger.info("   POST /ecd-assess     - ECDI2030 assessment scoring")
    logger.info("   GET  /features       - SHAP feature importance")
    logger.info("   GET  /model-info     - Model performance metrics")
    logger.info("   GET  /county-profile - County-specific insights")
    logger.info("   GET  /health         - Detailed health check")
    logger.info("=" * 70 + "\n")

    app.run(debug=True, port=5000, host="127.0.0.1")