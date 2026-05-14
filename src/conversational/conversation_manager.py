from .symptom_memory import SymptomMemory
from .dialogue_state import DialogueState
from .intent_detector import IntentDetector
from .response_formatter import ResponseFormatter
from .followup_engine import FollowupEngine
from src.safety.medical_consistency import MedicalConsistencyValidator
from src.safety.emergency_rules import EmergencyRuleEngine

class ConversationManager:
    def __init__(self):
        self.memory = SymptomMemory()
        self.state = DialogueState()
        self.intent_detector = IntentDetector()
        self.emergency_rules = EmergencyRuleEngine()
        self.formatter = ResponseFormatter()
        self.followup_engine = FollowupEngine()
        self.consistency_validator = MedicalConsistencyValidator()
        self.latest_predictions = []
        self.latest_uncertainty = None
        self.latest_flag_unreliable = False
        self.latest_consistency_data = None
        
    def handle_message(self, message, prediction_pipeline=None):
        # PART 3: PRE-ML TRIAGE PIPELINE
        emergency_check = self.emergency_rules.check_for_emergency(message)
        if emergency_check["is_emergency"]:
            self.latest_predictions = [] # Clear predictions
            self.latest_consistency_data = None
            self.latest_flag_unreliable = True
            return emergency_check["escalation_message"]
            
        intent = self.intent_detector.detect_intent(message)
        
        if intent == "GREETING":
            self.state.set_stage("SYMPTOM_INPUT")
            return "Hello! I am MediAssist. Please describe your symptoms."
            
        elif intent == "SYMPTOM_INPUT" or intent == "FOLLOWUP_ANSWER":
            self.memory.add_symptoms([message])
            
            if prediction_pipeline:
                results = prediction_pipeline.predict([self.memory.get_symptoms_text()])
                res = results[0]
                
                top_preds = res['predictions']
                uncertainty = res['uncertainty']
                
                # PART 6: MEDICAL CONSISTENCY PENALTIES
                consistency_res = self.consistency_validator.validate_and_rerank(self.memory.get_symptoms_text(), top_preds)
                top_preds = consistency_res['reranked_predictions']
                
                self.latest_consistency_data = consistency_res
                self.latest_predictions = top_preds
                self.latest_uncertainty = uncertainty
                self.latest_flag_unreliable = res.get('flag_unreliable', False)
                
                questions = self.followup_engine.get_followup_questions(top_preds, uncertainty)
                
                if questions and self.state.stage != "FINAL_RESPONSE":
                    self.state.set_stage("FOLLOWUP")
                    self.state.set_pending_questions(questions)
                    response = self.formatter.format_questions(questions)
                else:
                    self.state.set_stage("FINAL_RESPONSE")
                    response = self.formatter.format_predictions(top_preds)
                    
                # Basic safety guard disclaimer
                disclaimer = "Disclaimer: This is not a clinical diagnosis. Please consult a doctor."
                if disclaimer not in response:
                    response += "\n\n" + disclaimer
                    
                return response
                
        return "I'm not sure how to help with that. Please describe your symptoms."
