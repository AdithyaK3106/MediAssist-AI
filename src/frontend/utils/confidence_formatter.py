def normalize_topk_probabilities(predictions, k=3):
    """
    Normalize Top-K probabilities for frontend display only.
    Calculates the relative probability sum among the top-K items and scales them to 100%.
    """
    top_k = predictions[:k]
    
    if not top_k:
        return []
        
    total_prob = sum(pred.get('probability', 0.0) for pred in top_k)
    
    if total_prob == 0:
        return [{"disease": p['disease'], "relative_prob": 0.0, "raw_prob": p.get('probability', 0.0)} for p in top_k]
        
    normalized = []
    for pred in top_k:
        rel_prob = (pred.get('probability', 0.0) / total_prob) * 100.0
        normalized.append({
            "disease": pred['disease'],
            "relative_prob": rel_prob,
            "raw_prob": pred.get('probability', 0.0)
        })
        
    return normalized

def get_confidence_label(relative_prob):
    """Convert relative probability percentage into a human-friendly label."""
    if relative_prob >= 70:
        return "Most Likely Match"
    elif relative_prob >= 50:
        return "Moderate Match"
    elif relative_prob >= 30:
        return "Possible Match"
    elif relative_prob >= 10:
        return "Lower Match"
    else:
        return "Very Uncertain"

def get_confidence_color(relative_prob, severity="medium"):
    """
    Return a hex color based on severity first, then confidence.
    Red is reserved for critical severity.
    """
    if severity.lower() == "critical":
        return "#dc3545" # Red
        
    if relative_prob >= 50:
        return "#28a745" # Green (High confidence)
    elif relative_prob >= 30:
        return "#ffc107" # Yellow (Moderate)
    else:
        return "#fd7e14" # Orange (Low/Uncertain)
