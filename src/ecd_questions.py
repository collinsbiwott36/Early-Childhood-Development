"""
src/ecd_questions.py
UNICEF ECDI2030 Assessment Questions & Scoring
===============================================
20 questions across 4 domains for children aged 36-59 months
Aligned with SDG Indicator 4.2.1
"""

# ============================================================================
# ECDI2030 Questions by Domain
# ============================================================================
ECD_QUESTIONS = {
    'physical': {
        'domain_name': 'Physical Development',
        'icon': '🏃',
        'questions': [
            {
                'id': 'ECD21',
                'question': 'Can the child walk on uneven surfaces without falling?',
                'options': ['Yes', 'No'],
                'scoring': {'Yes': 1, 'No': 0}
            },
            {
                'id': 'ECD22',
                'question': 'Can the child jump with both feet off the ground?',
                'options': ['Yes', 'No'],
                'scoring': {'Yes': 1, 'No': 0}
            },
            {
                'id': 'ECD23',
                'question': 'Can the child dress themselves without help (put on clothes)?',
                'options': ['Yes', 'No'],
                'scoring': {'Yes': 1, 'No': 0}
            },
            {
                'id': 'ECD24',
                'question': 'Can the child button or unbutton their clothes?',
                'options': ['Yes', 'No'],
                'scoring': {'Yes': 1, 'No': 0}
            }
        ]
    },
    'language': {
        'domain_name': 'Language Development',
        'icon': '💬',
        'questions': [
            {
                'id': 'ECD25',
                'question': 'Can the child say at least 10 different words?',
                'options': ['Yes', 'No'],
                'scoring': {'Yes': 1, 'No': 0}
            },
            {
                'id': 'ECD26',
                'question': 'Can the child speak in sentences of 3 or more words?',
                'options': ['Yes', 'No'],
                'scoring': {'Yes': 1, 'No': 0}
            },
            {
                'id': 'ECD27',
                'question': 'Can the child speak in sentences of 5 or more words?',
                'options': ['Yes', 'No'],
                'scoring': {'Yes': 1, 'No': 0}
            },
            {
                'id': 'ECD28',
                'question': 'Does the child use pronouns correctly (I, you, he, she)?',
                'options': ['Yes', 'No'],
                'scoring': {'Yes': 1, 'No': 0}
            },
            {
                'id': 'ECD29',
                'question': 'Can the child name common objects when pointed to?',
                'options': ['Yes', 'No'],
                'scoring': {'Yes': 1, 'No': 0}
            }
        ]
    },
    'literacy': {
        'domain_name': 'Literacy-Numeracy',
        'icon': '📚',
        'questions': [
            {
                'id': 'ECD30',
                'question': 'Can the child identify or name at least 5 letters?',
                'options': ['Yes', 'No'],
                'scoring': {'Yes': 1, 'No': 0}
            },
            {
                'id': 'ECD31',
                'question': 'Can the child write their own name?',
                'options': ['Yes', 'No'],
                'scoring': {'Yes': 1, 'No': 0}
            },
            {
                'id': 'ECD32',
                'question': 'Can the child count from 1 to 5?',
                'options': ['Yes', 'No'],
                'scoring': {'Yes': 1, 'No': 0}
            },
            {
                'id': 'ECD33',
                'question': 'Can the child count 3 objects correctly when asked?',
                'options': ['Yes', 'No'],
                'scoring': {'Yes': 1, 'No': 0}
            },
            {
                'id': 'ECD34',
                'question': 'Can the child count up to 10 objects?',
                'options': ['Yes', 'No'],
                'scoring': {'Yes': 1, 'No': 0}
            }
        ]
    },
    'socio_emotional': {
        'domain_name': 'Socio-Emotional Development',
        'icon': '🤝',
        'questions': [
            {
                'id': 'ECD35',
                'question': 'Can the child play independently without constant supervision?',
                'options': ['Yes', 'No'],
                'scoring': {'Yes': 1, 'No': 0}
            },
            {
                'id': 'ECD36',
                'question': 'Can the child name familiar people (family, friends)?',
                'options': ['Yes', 'No'],
                'scoring': {'Yes': 1, 'No': 0}
            },
            {
                'id': 'ECD37',
                'question': 'Does the child try to help others (family, friends)?',
                'options': ['Yes', 'No'],
                'scoring': {'Yes': 1, 'No': 0}
            },
            {
                'id': 'ECD38',
                'question': 'Does the child get along well with other children?',
                'options': ['Yes', 'No'],
                'scoring': {'Yes': 1, 'No': 0}
            },
            {
                'id': 'ECD39',
                'question': 'How often does the child seem sad or unhappy?',
                'options': ['Never', 'A few times a year', 'Monthly', 'Weekly', 'Daily'],
                'scoring': {
                    'Never': 1,
                    'A few times a year': 1,
                    'Monthly': 1,
                    'Weekly': 0,
                    'Daily': 0
                }
            },
            {
                'id': 'ECD40',
                'question': 'How often does the child show aggressive behavior (hitting, biting)?',
                'options': ['Not at all', 'The same or less', 'More', 'A lot more'],
                'scoring': {
                    'Not at all': 1,
                    'The same or less': 1,
                    'More': 0,
                    'A lot more': 0
                }
            }
        ]
    }
}

# ============================================================================
# UNICEF ECDI2030 Age-Specific Thresholds
# ============================================================================
UNICEF_THRESHOLDS = {
    # (min_age, max_age): {total, physical, language, literacy, socio_emotional}
    (36, 41): {'total': 11, 'physical': 3, 'language': 3, 'literacy': 3, 'socio_emotional': 4},
    (42, 47): {'total': 13, 'physical': 3, 'language': 4, 'literacy': 4, 'socio_emotional': 4},
    (48, 59): {'total': 15, 'physical': 4, 'language': 4, 'literacy': 4, 'socio_emotional': 5}
}

def get_threshold_for_age(age_months: int) -> dict:
    """
    Get UNICEF threshold dict for child's age
    
    Args:
        age_months: Child's age in months (36-59)
    
    Returns:
        Dict with threshold values for total and each domain
    """
    for (min_age, max_age), threshold in UNICEF_THRESHOLDS.items():
        if min_age <= age_months <= max_age:
            return threshold
    # Default to oldest group if age is out of range
    return UNICEF_THRESHOLDS[(48, 59)]

def calculate_ecd_score(responses: dict, age_months: int) -> dict:
    """
    Calculate ECDI2030 score from responses
    
    Args:
        responses: Dict of {question_id: answer}, e.g., {"ECD21": "Yes", "ECD22": "No"}
        age_months: Child's age in months (36-59)
    
    Returns:
        Dict with:
        - scores: Raw scores per domain and total
        - max_scores: Maximum possible scores
        - percentages: Percentage achieved per domain
        - on_track: Boolean status per domain and composite
        - threshold: Threshold values used
        - age_months: Child's age
    """
    # Initialize scores
    scores = {
        'physical': 0,
        'language': 0,
        'literacy': 0,
        'socio_emotional': 0,
        'total': 0
    }
    
    max_scores = {
        'physical': 4,
        'language': 5,
        'literacy': 5,
        'socio_emotional': 6,
        'total': 20
    }
    
    # Score each response
    for domain, domain_data in ECD_QUESTIONS.items():
        for question in domain_data['questions']:
            q_id = question['id']
            if q_id in responses:
                answer = responses[q_id]
                # Get score from question's scoring dict, default to 0 if answer not found
                score = question['scoring'].get(answer, 0)
                scores[domain] += score
                scores['total'] += score
    
    # Calculate percentages
    percentages = {}
    for domain in scores.keys():
        if max_scores[domain] > 0:
            percentages[domain] = (scores[domain] / max_scores[domain]) * 100
        else:
            percentages[domain] = 0
    
    # Get threshold for age
    threshold = get_threshold_for_age(age_months)
    
    # Determine on-track status (score >= threshold)
    on_track = {
        'physical': scores['physical'] >= threshold.get('physical', 3),
        'language': scores['language'] >= threshold.get('language', 3),
        'literacy': scores['literacy'] >= threshold.get('literacy', 3),
        'socio_emotional': scores['socio_emotional'] >= threshold.get('socio_emotional', 4),
        'composite': scores['total'] >= threshold.get('total', 11)
    }
    
    return {
        'scores': scores,
        'max_scores': max_scores,
        'percentages': percentages,
        'on_track': on_track,
        'threshold': threshold,
        'age_months': age_months
    }

def get_all_question_ids() -> list:
    """Get list of all 20 ECD question IDs"""
    question_ids = []
    for domain_data in ECD_QUESTIONS.values():
        for question in domain_data['questions']:
            question_ids.append(question['id'])
    return question_ids

def validate_responses(responses: dict, age_months: int) -> tuple:
    """
    Validate that all required responses are provided
    
    Args:
        responses: Dict of {question_id: answer}
        age_months: Child's age
    
    Returns:
        Tuple of (is_valid, message)
    """
    # Check age range
    if not (36 <= age_months <= 59):
        return False, f"Age must be between 36 and 59 months, got {age_months}"
    
    # Check all 20 questions answered
    required_ids = get_all_question_ids()
    missing = [qid for qid in required_ids if qid not in responses]
    
    if missing:
        return False, f"Missing responses for: {missing}"
    
    # Check valid answers
    for domain_data in ECD_QUESTIONS.values():
        for question in domain_data['questions']:
            q_id = question['id']
            if q_id in responses:
                answer = responses[q_id]
                if answer not in question['options']:
                    return False, f"Invalid answer '{answer}' for {q_id}. Valid options: {question['options']}"
    
    return True, "Valid"