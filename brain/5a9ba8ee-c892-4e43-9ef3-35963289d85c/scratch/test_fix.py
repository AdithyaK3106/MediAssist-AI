
import sys
import os
from pathlib import Path

# Add project root to path
project_root = Path(__file__).resolve().parents[3]
sys.path.append(str(project_root))

from src.safety.medical_consistency import MedicalConsistencyValidator

def test_prediction_fix():
    validator = MedicalConsistencyValidator()
    
    # Simulate "mild fever and cough"
    user_input = "mild fever and cough"
    
    # Simulate bad ML predictions where Fracture has high confidence
    predictions = [
        {"disease": "Fracture Of The Facial Bones", "probability": 0.8},
        {"disease": "Common Cold", "probability": 0.15},
        {"disease": "Flu", "probability": 0.05}
    ]
    
    print(f"User Input: {user_input}")
    print(f"Original Predictions: {predictions}")
    
    result = validator.validate_and_rerank(user_input, predictions)
    reranked = result['reranked_predictions']
    
    print("\n--- Reranked Result ---")
    for i, p in enumerate(reranked):
        print(f"{i+1}. {p['disease']}: {p['probability']:.4f} (Original: {p['original_probability']:.4f})")

    # Verification
    if reranked[0]['disease'] in ["Common Cold", "Flu"]:
        print("\n✅ SUCCESS: Medical Consistency Layer correctly reranked the plausible diseases to the top.")
    else:
        print("\n❌ FAILURE: Inconsistent disease is still at the top.")

if __name__ == "__main__":
    test_prediction_fix()
