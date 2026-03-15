"""
api/schemas.py
Input validation schemas for Flask API
"""

# Required features for prediction
REQUIRED_FEATURES = [
    'child_age_months',
    'child_sex',
    'mother_age',
    'mother_education_level',
    'mother_years_education',
    'mother_currently_working',
    'marital_status',
    'wealth_quintile',
    'urban_rural',
    'religion',
    'county',
    'household_members',
    'has_electricity',
    'drinking_water_source',
    'toilet_facility',
    'cooking_fuel'
]

# Valid values for categorical features
VALID_VALUES = {
    'child_sex': ['Male', 'Female'],
    'urban_rural': ['Urban', 'Rural'],
    'wealth_quintile': ['Poorest', 'Poorer', 'Middle', 'Richer', 'Richest'],
    'mother_currently_working': ['Yes', 'No'],
    'has_electricity': ['Yes', 'No'],
}

def validate_prediction_request(data: dict):
    """
    Validate prediction request
    
    Returns:
        Tuple of (is_valid, message)
    """
    if not isinstance(data, dict):
        return False, "Payload must be a JSON object"
    
    # Check required fields
    missing = [f for f in REQUIRED_FEATURES if f not in data]
    if missing:
        return False, f"Missing required fields: {missing}"
    
    # Validate numeric ranges
    if not (36 <= data.get('child_age_months', 0) <= 59):
        return False, "child_age_months must be between 36 and 59"
    
    if not (15 <= data.get('mother_age', 0) <= 50):
        return False, "mother_age must be between 15 and 50"
    
    if not (0 <= data.get('mother_years_education', 0) <= 20):
        return False, "mother_years_education must be between 0 and 20"
    
    if not (1 <= data.get('household_members', 0) <= 20):
        return False, "household_members must be between 1 and 20"
    
    return True, "Valid"