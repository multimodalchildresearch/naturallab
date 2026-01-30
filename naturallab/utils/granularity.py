"""
Granularity Utilities for NaturalLab
===================================
Consistent object categorization functions for use across all analysis scripts.

Granularity levels:
- Coarse: Stage 1 detection categories (human, hand, toy, letter, book, background)
- Medium: toyset categories from annotation file (for toys) + original categories for non-toys
- Fine: object categories from annotation file (for toys) + original categories for non-toys
"""


def categorize_object(label, granularity='coarse', annotation_mappings=None):
    """
    Categorize objects based on granularity level using annotation mappings.
    
    Args:
        label: Object label (can be None for background)
        granularity: 'coarse', 'medium', or 'fine'
        annotation_mappings: Dict with object->toyset and object->granularity mappings
    
    Returns:
        Categorized object name
    """
    if label is None:
        return 'background'
    
    # Stage 1 detection categories (always the same for coarse)
    stage1_categories = {
        'human': 'human',
        'hand': 'hand', 
        'book': 'book'
    }
    
    # If it's a stage 1 category (not a toy), use it directly for all granularities
    if label in stage1_categories:
        return stage1_categories[label]
    
    # For toy objects, use annotation mappings if available
    if annotation_mappings and 'object_to_granularity' in annotation_mappings:
        if label in annotation_mappings['object_to_granularity']:
            granularity_info = annotation_mappings['object_to_granularity'][label]
            return granularity_info.get(granularity, label)
    
    # Fallback categorization if no annotation mappings available
    if granularity == 'coarse':
        # Coarse: basic categories from stage 1 detection
        if 'letter' in label.lower():
            return 'letter'
        else:
            return 'toy'
    
    elif granularity == 'medium':
        # Medium: try to infer toyset categories
        if 'letter' in label.lower():
            return 'Letter_set'
        elif 'egg' in label.lower():
            return 'egg_toy'
        elif 'ball' in label.lower():
            return 'ball_toy'
        else:
            return 'other_toy'
    
    elif granularity == 'fine':
        # Fine: use exact labels
        return label
    
    else:
        raise ValueError(f"Unknown granularity: {granularity}")


def get_granularity_mapping(annotation_mappings):
    """
    Get a complete mapping of all objects to their granularity categories.
    
    Args:
        annotation_mappings: Dict with annotation data
    
    Returns:
        Dict with object -> {coarse, medium, fine} mappings
    """
    if not annotation_mappings or 'object_to_granularity' not in annotation_mappings:
        return {}
    
    return annotation_mappings['object_to_granularity']


def validate_granularity_consistency(detection_data):
    """
    Validate that granularity mappings are consistent in detection data.
    
    Args:
        detection_data: Detection results dict
    
    Returns:
        Bool indicating if mappings are consistent
    """
    if 'annotation_mappings' not in detection_data:
        print("Warning: No annotation mappings found in detection data")
        return False
    
    mappings = detection_data['annotation_mappings']
    
    if not mappings.get('object_to_granularity'):
        print("Warning: No granularity mappings found")
        return False
    
    print(f"Found granularity mappings for {len(mappings['object_to_granularity'])} objects")
    return True


def print_granularity_summary(annotation_mappings):
    """
    Print a summary of available granularity categories.
    
    Args:
        annotation_mappings: Dict with annotation data
    """
    if not annotation_mappings or 'object_to_granularity' not in annotation_mappings:
        print("No annotation mappings available")
        return
    
    granularity_data = annotation_mappings['object_to_granularity']
    
    # Collect all categories for each granularity level
    coarse_categories = set()
    medium_categories = set()
    fine_categories = set()
    
    for obj, categories in granularity_data.items():
        coarse_categories.add(categories.get('coarse', 'unknown'))
        medium_categories.add(categories.get('medium', 'unknown'))
        fine_categories.add(categories.get('fine', 'unknown'))
    
    print(f"\nGranularity Summary:")
    print(f"Coarse categories ({len(coarse_categories)}): {sorted(coarse_categories)}")
    print(f"Medium categories ({len(medium_categories)}): {sorted(medium_categories)}")
    print(f"Fine categories ({len(fine_categories)}): {len(fine_categories)} unique objects") 