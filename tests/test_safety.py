import pytest
import sys
from pathlib import Path

project_root = Path(__file__).resolve().parents[1]
sys.path.append(str(project_root))

from src.safety.emergency_rules import EmergencyRuleEngine
from src.safety.medical_consistency import MedicalConsistencyValidator

@pytest.fixture
def emergency_engine():
    return EmergencyRuleEngine()

@pytest.fixture
def consistency_validator():
    return MedicalConsistencyValidator()

def test_stroke_emergency(emergency_engine):
    response = emergency_engine.check_for_emergency("I have slurred speech and facial weakness")
    assert response["is_emergency"] == True
    assert "Neurological Emergency" in response["categories"]

def test_respiratory_emergency(emergency_engine):
    response = emergency_engine.check_for_emergency("I am coughing blood and have difficulty breathing")
    assert response["is_emergency"] == True
    assert "Respiratory Emergency" in response["categories"]

def test_severe_infection(emergency_engine):
    response = emergency_engine.check_for_emergency("stiff neck and extreme confusion with very high fever")
    assert response["is_emergency"] == True
    assert "Severe Infection" in response["categories"]

def test_hemorrhagic_warning(emergency_engine):
    response = emergency_engine.check_for_emergency("I have bleeding gums and high fever")
    assert response["is_emergency"] == True
    assert "Hemorrhagic Warning" in response["categories"]

def test_no_emergency(emergency_engine):
    response = emergency_engine.check_for_emergency("I have a mild headache and runny nose")
    assert response["is_emergency"] == False

def test_implausible_prediction_suppression(consistency_validator):
    # If input is fever and cough, and prediction is eye infection (which is unrelated)
    # The penalty should heavily suppress it.
    predictions = [
        {"disease": "Common Cold", "probability": 0.8}, # Has overlap
        {"disease": "Impetigo", "probability": 0.7} # No overlap with respiratory/fever
    ]
    
    # "cough" and "fever" -> respiratory, fever
    res = consistency_validator.validate_and_rerank("I have a high fever and cough", predictions)
    reranked = res["reranked_predictions"]
    
    # Common cold should have a higher probability than Impetigo now, even if Impetigo started high
    # Actually, Impetigo started at 0.7, with penalty 0.1, it becomes 0.07.
    # Common Cold started at 0.8, with penalty 1.0 (assuming it maps to respiratory/fever), it becomes 0.8.
    
    impetigo_pred = next(p for p in reranked if p["disease"] == "Impetigo")
    assert impetigo_pred["probability"] <= 0.07 # 0.7 * 0.1
    
    reasoning = res["reasoning"]
    impetigo_reasoning = next(r for r in reasoning if r["disease"] == "Impetigo")
    assert impetigo_reasoning["consistency"] == "LOW"

# New Safety Metrics Calculator
class SafetyMetrics:
    def __init__(self):
        self.emergency_cases = 0
        self.correctly_escalated = 0
        self.missed_emergencies = 0
        
        self.implausible_cases = 0
        self.suppressed_implausible = 0

    def evaluate_emergency_escalation(self, test_cases, engine):
        for text, is_true_emergency in test_cases:
            res = engine.check_for_emergency(text)
            pred_emergency = res["is_emergency"]
            
            if is_true_emergency:
                self.emergency_cases += 1
                if pred_emergency:
                    self.correctly_escalated += 1
                else:
                    self.missed_emergencies += 1

    def calculate_metrics(self):
        eea = self.correctly_escalated / self.emergency_cases if self.emergency_cases else 0
        mer = self.missed_emergencies / self.emergency_cases if self.emergency_cases else 0
        
        return {
            "Emergency Escalation Accuracy (EEA)": eea,
            "Missed Emergency Rate (MER)": mer
        }

def test_safety_metrics():
    metrics = SafetyMetrics()
    engine = EmergencyRuleEngine()
    test_cases = [
        ("coughing up blood", True),
        ("mild headache", False),
        ("slurred speech", True),
        ("bleeding gums", True),
        ("runny nose", False)
    ]
    metrics.evaluate_emergency_escalation(test_cases, engine)
    results = metrics.calculate_metrics()
    assert results["Emergency Escalation Accuracy (EEA)"] == 1.0
    assert results["Missed Emergency Rate (MER)"] == 0.0

if __name__ == "__main__":
    pytest.main([__file__, "-v"])
