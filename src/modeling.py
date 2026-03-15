"""
src/modeling.py
Model training and prediction for ECD delay
"""

import pandas as pd
import numpy as np
from pathlib import Path
import joblib
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.metrics import classification_report, roc_auc_score, confusion_matrix
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import LabelEncoder
import warnings
warnings.filterwarnings('ignore')

from src.data import get_data_paths, DOMAIN_ITEMS, ECD_COLS

# ============================================================================
# PREPROCESSING
# ============================================================================
class ECDPreprocessor:
    """Preprocess data for modeling"""
    
    def __init__(self):
        self.label_encoders = {}
        self.feature_columns = None
    
    def fit_transform(self, df: pd.DataFrame, target_col: str) -> tuple:
        """Fit preprocessors and transform data"""
        df_proc = df.copy()
        
        # Separate features and target
        y = df_proc[target_col].values
        
        # Drop target and ECD items from features
        feature_cols = [c for c in df_proc.columns 
                       if c not in [target_col] + ECD_COLS + ['complete_ecd_strict']]
        
        self.feature_columns = feature_cols
        X = df_proc[feature_cols].copy()
        
        # Encode categorical variables
        for col in X.select_dtypes(include=['object', 'string', 'category']).columns:
            le = LabelEncoder()
            X[col] = X[col].fillna('Missing')
            X[col] = le.fit_transform(X[col].astype(str))
            self.label_encoders[col] = le
        
        # Fill numeric missing values
        for col in X.select_dtypes(include=[np.number]).columns:
            X[col] = X[col].fillna(X[col].median())
        
        return X, y
    
    def transform(self, df: pd.DataFrame) -> pd.DataFrame:
        """Transform new data using fitted preprocessors"""
        df_proc = df.copy()
        X = df_proc[self.feature_columns].copy()
        
        for col, le in self.label_encoders.items():
            if col in X.columns:
                X[col] = X[col].fillna('Missing')
                # Handle unseen categories
                X[col] = X[col].apply(lambda x: le.transform([x])[0] if x in le.classes_ else -1)
        
        for col in X.select_dtypes(include=[np.number]).columns:
            X[col] = X[col].fillna(0)
        
        return X
    
    def save(self, path: Path):
        """Save preprocessor"""
        joblib.dump(self, path)
    
    @staticmethod
    def load(path: Path):
        """Load preprocessor"""
        return joblib.load(path)

# ============================================================================
# MODEL TRAINING
# ============================================================================
def train_model(X: pd.DataFrame, y: np.ndarray, 
                model_type: str = 'random_forest',
                save_path: Path = None) -> tuple:
    """
    Train a classification model
    
    Returns:
        Tuple of (model, preprocessor, metrics)
    """
    # Split data
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )
    
    # Initialize model
    if model_type == 'random_forest':
        model = RandomForestClassifier(
            n_estimators=100, 
            max_depth=10,
            random_state=42,
            class_weight='balanced',
            n_jobs=-1
        )
    elif model_type == 'xgboost':
        from xgboost import XGBClassifier
        model = XGBClassifier(
            n_estimators=100,
            max_depth=6,
            learning_rate=0.1,
            random_state=42,
            use_label_encoder=False,
            eval_metric='logloss'
        )
    else:
        raise ValueError(f"Unknown model type: {model_type}")
    
    # Train
    model.fit(X_train, y_train)
    
    # Evaluate
    y_pred = model.predict(X_test)
    y_proba = model.predict_proba(X_test)[:, 1]
    
    metrics = {
        'roc_auc': roc_auc_score(y_test, y_proba),
        'classification_report': classification_report(y_test, y_pred),
        'confusion_matrix': confusion_matrix(y_test, y_pred).tolist()
    }
    
    print(f"✅ Model trained - ROC AUC: {metrics['roc_auc']:.4f}")
    print(metrics['classification_report'])
    
    # Save if path provided
    if save_path:
        save_path.mkdir(parents=True, exist_ok=True)
        joblib.dump(model, save_path / f"{model_type}_model.pkl")
        print(f"✅ Model saved to: {save_path / f'{model_type}_model.pkl'}")
    
    return model, metrics

# ============================================================================
# PREDICTION
# ============================================================================
def predict_delay(model, preprocessor, df: pd.DataFrame) -> pd.DataFrame:
    """
    Make predictions on new data
    
    Returns:
        DataFrame with predictions and probabilities
    """
    X = preprocessor.transform(df)
    
    df_pred = df.copy()
    df_pred['predicted_delay'] = model.predict(X)
    df_pred['predicted_probability'] = model.predict_proba(X)[:, 1]
    
    return df_pred

# ============================================================================
# FULL PIPELINE
# ============================================================================
def train_and_save_models(data_path: Path = None, 
                          target_col: str = 'ecd_delay_composite') -> dict:
    """
    Train and save all models (composite + 4 domains)
    
    Returns:
        Dict of trained models and metrics
    """
    paths = get_data_paths()
    
    if data_path is None:
        data_path = paths['processed'] / "kdhs_processed_clean.parquet"
    
    # Load data
    df = pd.read_parquet(data_path)
    print(f"✅ Loaded data: {df.shape}")
    
    results = {}
    
    # Train composite model
    print("\n" + "=" * 50)
    print("TRAINING COMPOSITE MODEL")
    print("=" * 50)
    
    prep = ECDPreprocessor()
    X, y = prep.fit_transform(df, target_col)
    model, metrics = train_model(X, y, model_type='random_forest', 
                                 save_path=paths['models'])
    prep.save(paths['models'] / 'preprocessor.pkl')
    
    results['composite'] = {'model': model, 'preprocessor': prep, 'metrics': metrics}
    
    # Train domain models
    for domain in DOMAIN_ITEMS.keys():
        domain_target = f'{domain}_delay'
        if domain_target in df.columns:
            print(f"\n" + "=" * 50)
            print(f"TRAINING {domain.upper()} MODEL")
            print("=" * 50)
            
            prep_domain = ECDPreprocessor()
            X_d, y_d = prep_domain.fit_transform(df, domain_target)
            model_d, metrics_d = train_model(
                X_d, y_d, 
                model_type='random_forest',
                save_path=paths['models']
            )
            prep_domain.save(paths['models'] / f'preprocessor_{domain}.pkl')
            
            results[domain] = {'model': model_d, 'preprocessor': prep_domain, 'metrics': metrics_d}
    
    # Save results summary
    import json
    with open(paths['models'] / 'training_results.json', 'w') as f:
        json.dump({k: {'roc_auc': v['metrics']['roc_auc']} 
                   for k, v in results.items()}, f, indent=2)
    
    print(f"\n✅ All models saved to: {paths['models']}")
    return results