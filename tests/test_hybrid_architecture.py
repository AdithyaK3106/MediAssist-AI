import pytest
from src.conversational.triage_orchestrator import MediAssistTriageOrchestrator
from src.safety.emergency_firewall import EmergencySafetyFirewall
from src.conversational.timeline_extractor import SymptomTimelineExtractor
from src.safety.decision_layer import ClinicalDecisionLayer

@pytest.fixture
def orchestrator():
    return MediAssistTriageOrchestrator()

@pytest.fixture
def firewall():
    return EmergencySafetyFirewall()

@pytest.fixture
def extractor():
    return SymptomTimelineExtractor()

@pytest.fixture
def decision_layer():
    return ClinicalDecisionLayer()

# 1. Emergency Safety Firewall Tests (Veto Authority)
def test_stroke_emergency_veto(firewall):
    res = firewall.evaluate("Sudden slurred speech and facial drooping on one side")
    assert res["is_emergency"] is True
    assert "Neurological Emergency" in res["categories"][0]
    assert "🚨 CRITICAL MEDICAL EMERGENCY" in res["guidance_en"]

def test_cardiac_emergency_veto(firewall):
    res = firewall.evaluate("Severe crushing chest pain radiating to left arm")
    assert res["is_emergency"] is True
    assert "Cardiac Emergency" in res["categories"][0]

def test_respiratory_emergency_veto(firewall):
    res = firewall.evaluate("I am coughing up blood and cannot breathe")
    assert res["is_emergency"] is True
    assert "Respiratory Emergency" in res["categories"][0]

def test_hemorrhagic_warning_veto(firewall):
    res = firewall.evaluate("High fever with bleeding gums and blood in vomit")
    assert res["is_emergency"] is True
    assert "Hemorrhagic Warning" in res["categories"][0]

def test_routine_symptoms_pass_firewall(firewall):
    res = firewall.evaluate("I have a mild runny nose and sneezing for 2 days")
    assert res["is_emergency"] is False

# 2. Timeline Extraction Tests
def test_timeline_extraction_duration_and_severity(extractor):
    text = "I have had severe fever and body aches for 3 days and it is getting worse"
    profile = extractor.extract(text)
    assert "fever" in profile["symptoms"]
    assert profile["timeline"]["duration_days"] == 3
    assert profile["timeline"]["severity"] == "severe"
    assert profile["timeline"]["progression"] == "worsening"

def test_hindi_language_detection(extractor):
    text = "मुझे 4 दिन से बुखार और सिर दर्द है"
    profile = extractor.extract(text)
    assert profile["language"] == "hi"

# 3. Decision Layer & Home Care Gating Tests
def test_home_care_strictly_blocked_for_high_severity(decision_layer):
    bert_preds = [{"disease": "Dengue", "probability": 0.85}]
    timeline = {"duration_days": 3, "severity": "moderate", "progression": "worsening"}
    res = decision_layer.evaluate(bert_preds, timeline)
    
    # Dengue is a high-severity condition -> home care must be blocked
    assert res["urgency_level"] == "High"
    assert res["allow_home_care"] is False
    assert len(res["home_care"]) == 0
    assert len(res["hospitals"]) > 0

def test_home_care_allowed_for_low_severity(decision_layer):
    bert_preds = [{"disease": "Common Cold", "probability": 0.80}]
    timeline = {"duration_days": 1, "severity": "mild", "progression": "stable"}
    res = decision_layer.evaluate(bert_preds, timeline)
    
    assert res["urgency_level"] == "Low"
    assert res["allow_home_care"] is True
    assert len(res["home_care"]) > 0

# 4. End-to-End Orchestrator Multi-Turn Tests
def test_orchestrator_turn_progression(orchestrator):
    # Turn 1: Vague -> Expect follow-up question
    t1 = orchestrator.process_message("I have a cough and mild fever")
    assert t1["status"] == "FOLLOWUP_PENDING"
    assert "Is your condition worsening" in t1["message"] or "answer:" in t1["message"]

    # Turn 2: Clarification provided -> Expect complete assessment
    t2 = orchestrator.process_message("It has been 2 days and it is staying about the same")
    assert t2["status"] == "ASSESSMENT_COMPLETE"
    assert "Top Possible Conditions:" in t2["message"]
    assert "Urgency Level:" in t2["message"]
    assert "Medical Disclaimer:" in t2["message"]

def test_orchestrator_emergency_immediate_halt(orchestrator):
    # Emergency should trigger immediately on Turn 1
    t = orchestrator.process_message("I have slurred speech and facial weakness")
    assert t["status"] == "EMERGENCY_OVERRIDE"
    assert t["is_emergency"] is True
    assert "Top Possible Conditions" not in t["message"] # Verify BERT was bypassed
