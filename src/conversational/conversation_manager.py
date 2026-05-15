
from .symptom_memory import SymptomMemory
from .dialogue_state import DialogueState
from .intent_detector import IntentDetector
from .response_formatter import ResponseFormatter
from .followup_engine import FollowupEngine
from .safety_guard import SafetyGuard
from src.safety.medical_consistency import MedicalConsistencyValidator
from src.safety.emergency_rules import EmergencyRuleEngine
from src.recommendation.recommender import HospitalRecommender

class ConversationManager:
    """Research-grade dialogue manager with defense-in-depth safety layers."""
    
    def __init__(self):
        self.memory = SymptomMemory()
        self.state = DialogueState()
        self.intent_detector = IntentDetector()
        self.emergency_rules = EmergencyRuleEngine()
        self.formatter = ResponseFormatter()
        self.followup_engine = FollowupEngine()
        self.consistency_validator = MedicalConsistencyValidator()
        self.safety_guard = SafetyGuard(confidence_threshold=0.15)
        self.hospital_recommender = HospitalRecommender()
        
        self.latest_predictions = []
        self.latest_uncertainty = None
        self.latest_flag_unreliable = False
        self.latest_consistency_data = None
        
    def handle_message(self, message, prediction_pipeline=None):
        # LAYER 1: PRE-ML EMERGENCY TRIAGE
        emergency_check = self.emergency_rules.check_for_emergency(message)
        if emergency_check.get("is_emergency"):
            self.latest_predictions = [] 
            self.latest_flag_unreliable = True
            return emergency_check["escalation_message"]
            
        # Intent Detection
        intent = self.intent_detector.detect_intent(message)
        
        if intent == "GREETING":
            self.state.set_stage("SYMPTOM_INPUT")
            return "Hello! I am MediAssist. Please describe your symptoms as clearly as possible."
            
        elif intent == "SYMPTOM_INPUT" or intent == "FOLLOWUP_ANSWER":
            self.memory.add_symptoms([message])
            
            if prediction_pipeline:
                # LAYER 2: ML INFERENCE
                results = prediction_pipeline.predict([self.memory.get_symptoms_text()])
                res = results[0]
                
                top_preds = res['predictions']
                uncertainty = res['uncertainty']
                
                # LAYER 3: POST-ML MEDICAL CONSISTENCY (Reranking)
                consistency_res = self.consistency_validator.validate_and_rerank(
                    self.memory.get_symptoms_text(), 
                    top_preds
                )
                top_preds = consistency_res['reranked_predictions']
                
                self.latest_consistency_data = consistency_res
                self.latest_predictions = top_preds
                self.latest_uncertainty = uncertainty
                
                # LAYER 4: DIALOGUE MANAGEMENT (Follow-ups)
                questions = self.followup_engine.get_followup_questions(top_preds, uncertainty)
                
                if questions and self.state.stage != "FINAL_RESPONSE":
                    self.state.set_stage("FOLLOWUP")
                    self.state.set_pending_questions(questions)
                    return self.formatter.format_questions(questions)
                else:
                    self.state.set_stage("FINAL_RESPONSE")
                    
                    # Get hospital recommendations (Default to Bangalore for demo)
                    top_cat = None
                    if self.latest_consistency_data and self.latest_consistency_data['detected_symptoms']:
                        top_cat = self.latest_consistency_data['detected_symptoms'][0]
                    
                    hospitals = self.hospital_recommender.get_recommendations(
                        user_lat=12.9716, user_lon=77.5946, # Bangalore Center
                        disease_category=top_cat
                    )
                    
                    response = self.formatter.format_predictions(top_preds, hospitals=hospitals)
                    
                # LAYER 5: FINAL SAFETY GUARD (Confidence Check & Disclaimers)
                return self.safety_guard.check_response(
                    response, 
                    top_predictions=top_preds, 
                    uncertainty=uncertainty
                )
                
        return "I'm not sure how to help with that. Could you describe your symptoms more clearly?"
