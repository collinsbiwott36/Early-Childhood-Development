# KDHS 2022 KR — ECD Delay Prediction (36–59 months)

This project predicts early childhood developmental delay using KDHS 2022 (KR file),
with domain-specific outcomes (4 domains) + composite outcome, explainability (SHAP),
and deployment via Streamlit dashboard + Flask API.

## Folder guide
- data/raw/            Put KEKR8CFL.SAV here
- data/interim/        Outputs from Notebook 01 (cleaned EDA)
- data/processed/      Model-ready dataset from Notebook 02
- data/outputs/        Dashboard-ready outputs (county risk tables etc.)
- geo/                 Put kenya_counties.geojson here
- reports/figures/     Thesis plots
- reports/tables/      Thesis tables
- reports/tuning/      Hyperparameter tuning logs/plots
- notebooks/           EDA → FE/Targets → Modeling → SHAP → Geo
- models/              Saved models and pipelines (joblib)
- dashboard/           Streamlit app (Frontend)
- api/                 Flask API service (Backend)
- src/                 Core python modules
- tests/               Unit tests for target scoring
- configs/             YAML/JSON configs for reproducibility

## Start
1) Put KR file: data/raw/KEKR8CFL.SAV
2) Put GeoJSON: geo/kenya_counties.geojson
3) Install: pip install -r requirements.txt

## Run Flask API (Backend)
python api/main.py
# API will run on http://127.0.0.1:5000

## Run Streamlit (Frontend)
streamlit run dashboard/app.py
# Dashboard will run on http://localhost:8501
