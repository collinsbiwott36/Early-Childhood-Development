"""
src/targets.py
UNICEF ECDI2030 Target Creation for KDHS ECD Delay Prediction
============================================================
FOCUS: Children aged 36-59 months only

UNICEF Age-Specific Thresholds (for 36-59mo range):
- 36-41 months: ≥11 milestones = "On Track"
- 42-47 months: ≥13 milestones = "On Track"  
- 48-59 months: ≥15 milestones = "On Track"

SDG Indicator 4.2.1 Compatible
"""

import pandas as pd
import numpy as np
from src.data import DOMAIN_ITEMS, ECD_COLS

# ============================================================================
# UNICEF ECDI2030 THRESHOLDS (36-59 MONTHS ONLY)
# ============================================================================
UNICEF_THRESHOLDS_36_59 = {
    (36, 41): 11,  # 36-41 months: ≥11 milestones
    (42, 47): 13,  # 42-47 months: ≥13 milestones
    (48, 59): 15,  # 48-59 months: ≥15 milestones
}

def get_unicef_threshold_36_59(age_months: int) -> int:
    """
    Get UNICEF ECDI2030 threshold for age 36-59 months
    
    Args:
        age_months: Child's age in months (must be 36-59)
    
    Returns:
        Minimum milestones needed to be "on track"
    """
    for (min_age, max_age), threshold in UNICEF_THRESHOLDS_36_59.items():
        if min_age <= age_months <= max_age:
            return threshold
    # Handle edge cases
    if age_months < 36:
        return UNICEF_THRESHOLDS_36_59[(36, 41)]
    return UNICEF_THRESHOLDS_36_59[(48, 59)]

def calculate_ecd_count(row, ecd_cols: list = None) -> int:
    """Count achieved ECD milestones (0-20)"""
    if ecd_cols is None:
        ecd_cols = ECD_COLS
    available_cols = [c for c in ecd_cols if c in row.index]
    return int(row[available_cols].sum(skipna=True))

# ============================================================================
# DOMAIN SCORES (0.0 to 1.0)
# ============================================================================
def make_domain_scores(df: pd.DataFrame) -> pd.DataFrame:
    """Calculate domain scores as mean of achieved items"""
    df_score = df.copy()
    
    for domain_name, items in DOMAIN_ITEMS.items():
        available_items = [i for i in items if i in df_score.columns]
        if len(available_items) > 0:
            df_score[f'{domain_name}_score'] = df_score[available_items].mean(axis=1, skipna=True)
    
    print(f"✅ Created {len(DOMAIN_ITEMS)} domain scores")
    return df_score

# ============================================================================
# UNICEF COMPOSITE TARGET (36-59mo)
# ============================================================================
def make_unicef_composite_target(df: pd.DataFrame, 
                                  age_col: str = 'child_age_months',
                                  ecd_cols: list = None) -> pd.DataFrame:
    """
    Create UNICEF ECDI2030 composite target for ages 36-59 months.
    
    Target definition:
    - Count achieved milestones (0-20)
    - Apply age-specific threshold (11/13/15)
    - 1 = Delayed (below threshold), 0 = On Track (at/above)
    
    Aligns with SDG indicator 4.2.1 methodology.
    """
    if ecd_cols is None:
        ecd_cols = ECD_COLS
    
    df_target = df.copy()
    
    # Ensure ECD items are numeric
    for col in ecd_cols:
        if col in df_target.columns and df_target[col].dtype == 'object':
            df_target[col] = pd.to_numeric(df_target[col], errors='coerce').fillna(0)
    
    # Count achieved milestones
    df_target['ecd_milestone_count'] = df_target[ecd_cols].sum(axis=1, skipna=True)
    
    # Apply age-specific threshold
    df_target['ecd_age_threshold'] = df_target[age_col].apply(get_unicef_threshold_36_59)
    
    # Create binary target: 1 = Delayed
    df_target['ecd_delay_composite'] = (
        df_target['ecd_milestone_count'] < df_target['ecd_age_threshold']
    ).astype(int)
    
    # Add inverse for convenience
    df_target['ecd_on_track'] = (df_target['ecd_delay_composite'] == 0).astype(int)
    
    print(f"✅ Created UNICEF ECDI2030 composite target")
    return df_target

# ============================================================================
# DOMAIN TARGETS (60% threshold proxy)
# ============================================================================
def make_domain_targets(df: pd.DataFrame, 
                        domain_threshold_ratio: float = 0.6) -> pd.DataFrame:
    """
    Create domain-specific delay targets.
    
    Child is "on track" in a domain if they achieve ≥60% of items in that domain.
    """
    df_target = df.copy()
    
    for domain_name, items in DOMAIN_ITEMS.items():
        available_items = [i for i in items if i in df_target.columns]
        if len(available_items) == 0:
            continue
        
        domain_count = df_target[available_items].sum(axis=1, skipna=True)
        threshold = np.ceil(len(available_items) * domain_threshold_ratio)
        
        df_target[f'{domain_name}_delay'] = (domain_count < threshold).astype(int)
        df_target[f'{domain_name}_count'] = domain_count
        df_target[f'{domain_name}_threshold'] = int(threshold)
        
        print(f"   {domain_name}: threshold = {threshold} ({domain_threshold_ratio:.0%})")
    
    print(f"✅ Created {len(DOMAIN_ITEMS)} domain targets")
    return df_target

# ============================================================================
# FULL PIPELINE
# ============================================================================
def create_all_targets(df: pd.DataFrame, 
                       age_col: str = 'child_age_months',
                       ecd_cols: list = None) -> pd.DataFrame:
    """Create all UNICEF-aligned targets for 36-59 month cohort"""
    df_targets = df.copy()
    df_targets = make_domain_scores(df_targets)
    df_targets = make_unicef_composite_target(df_targets, age_col=age_col, ecd_cols=ecd_cols)
    df_targets = make_domain_targets(df_targets)
    return df_targets

# ============================================================================
# SUMMARY & VALIDATION
# ============================================================================
def print_target_summary_36_59(df: pd.DataFrame, 
                                age_col: str = 'child_age_months',
                                target_col: str = 'ecd_delay_composite'):
    """Print formatted summary for 36-59 month cohort"""
    
    print("\n" + "="*70)
    print("📊 UNICEF ECDI2030 TARGET SUMMARY (Ages 36-59 Months)")
    print("="*70)
    
    # Overall
    print(f"\n🔹 Overall (36-59 months):")
    print(f"   Total children: {len(df):,}")
    print(f"   On track: {(df[target_col] == 0).sum():,} ({(1-df[target_col].mean()):.1%})")
    print(f"   Delayed:    {(df[target_col] == 1).sum():,} ({df[target_col].mean():.1%})")
    
    # By age group
    print(f"\n🔹 By Age Group (UNICEF thresholds):")
    for (min_age, max_age), threshold in UNICEF_THRESHOLDS_36_59.items():
        mask = (df[age_col] >= min_age) & (df[age_col] <= max_age)
        if mask.sum() > 0:
            subgroup = df.loc[mask]
            on_track = (subgroup[target_col] == 0).sum()
            rate = on_track / mask.sum()
            print(f"   {min_age:2d}-{max_age:2d} mo | Threshold: {threshold:2d} | "
                  f"N={mask.sum():4d} | On Track: {on_track:4d} ({rate:.1%})")
    
    # Domain targets
    domain_targets = [f'{d}_delay' for d in DOMAIN_ITEMS.keys() if f'{d}_delay' in df.columns]
    if domain_targets:
        print(f"\n🔹 Domain Delay Rates:")
        for target in domain_targets:
            rate = df[target].mean()
            domain_name = target.replace('_delay', '').title()
            print(f"   {domain_name:20} | Delayed: {rate:.1%}")
    
    print("\n" + "="*70)

def validate_unicef_targets(df: pd.DataFrame, 
                            age_col: str = 'child_age_months',
                            target_col: str = 'ecd_delay_composite') -> dict:
    """Validate that targets follow UNICEF methodology"""
    results = {}
    
    # Check required columns exist
    required = ['ecd_milestone_count', 'ecd_age_threshold', target_col]
    results['columns_present'] = all(c in df.columns for c in required)
    
    if not results['columns_present']:
        results['error'] = f"Missing required columns: {[c for c in required if c not in df.columns]}"
        return results
    
    # Verify threshold logic
    results['threshold_mismatch'] = False
    for idx, row in df.sample(min(100, len(df)), random_state=42).iterrows():
        age = row[age_col]
        expected_threshold = get_unicef_threshold_36_59(age)
        actual_threshold = row['ecd_age_threshold']
        if expected_threshold != actual_threshold:
            results['threshold_mismatch'] = True
            break
    
    return results