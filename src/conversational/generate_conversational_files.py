import os

files = {
    "symptom_memory.py": '''
class SymptomMemory:
    def __init__(self):
        self.current_symptoms = []
        self.history = []
        
    def add_symptoms(self, symptoms):
        for s in symptoms:
            if s not in self.current_symptoms:
                self.current_symptoms.append(s)
                
    def get_symptoms_text(self):
        return " ".join(self.current_symptoms)
        
    def clear(self):
        self.current_symptoms = []
        self.history = []
''',
    "dialogue_state.py": '''
class DialogueState:
    def __init__(self):
        self.stage = "GREETING" # GREETING, SYMPTOM_INPUT, FOLLOWUP, FINAL_RESPONSE
        self.pending_questions = []
        self.uncertainty_status = False
        
    def set_stage(self, stage):
        self.stage = stage
        
    def set_pending_questions(self, questions):
        self.pending_questions = questions
''',
    "intent_detector.py": '''
class IntentDetector:
    def detect_intent(self, text):
        text = text.lower()
        if any(g in text for g in ["hello", "hi", "hey"]):
            return "GREETING"
        elif any(e in text for e in ["chest pain", "cannot breathe", "stroke"]):
            return "EMERGENCY"
        elif any(y in text for y in ["yes", "no", "maybe"]):
            return "FOLLOWUP_ANSWER"
        else:
            return "SYMPTOM_INPUT"
''',
    "safety_guard.py": '''
class SafetyGuard:
    def check_response(self, response):
        disclaimer = "Disclaimer: This is not a clinical diagnosis. Please consult a doctor."
        if disclaimer not in response:
            response += "\\n\\n" + disclaimer
        return response
        
    def detect_emergency(self, text):
        emergency_keywords = ["chest pain", "shortness of breath", "unconscious", "seizure"]
        return any(k in text.lower() for k in emergency_keywords)
''',
    "response_formatter.py": '''
class ResponseFormatter:
    def format_predictions(self, top_predictions, upr=None):
        response = "### Possible Conditions:\\n"
        for i, pred in enumerate(top_predictions[:3]):
            response += f"{i+1}. {pred['disease']} — {pred['probability']*100:.1f}%\\n"
            
        response += "\\n### Suggested Action:\\n"
        response += "Please consult a healthcare professional for an accurate diagnosis."
        return response
        
    def format_questions(self, questions):
        response = "To help me narrow down the possibilities, could you answer a few questions?\\n\\n"
        for i, q in enumerate(questions):
            response += f"{i+1}. {q}\\n"
        return response
''',
    "conversation_manager.py": '''
from .symptom_memory import SymptomMemory
from .dialogue_state import DialogueState
from .intent_detector import IntentDetector
from .safety_guard import SafetyGuard
from .response_formatter import ResponseFormatter
from .followup_engine import FollowupEngine

class ConversationManager:
    def __init__(self):
        self.memory = SymptomMemory()
        self.state = DialogueState()
        self.intent_detector = IntentDetector()
        self.safety_guard = SafetyGuard()
        self.formatter = ResponseFormatter()
        self.followup_engine = FollowupEngine()
        
    def handle_message(self, message, prediction_pipeline=None):
        if self.safety_guard.detect_emergency(message):
            return "⚠️ EMERGENCY: Please seek immediate medical attention or call emergency services."
            
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
                
                questions = self.followup_engine.get_followup_questions(top_preds, uncertainty)
                
                if questions and self.state.stage != "FINAL_RESPONSE":
                    self.state.set_stage("FOLLOWUP")
                    self.state.set_pending_questions(questions)
                    response = self.formatter.format_questions(questions)
                else:
                    self.state.set_stage("FINAL_RESPONSE")
                    response = self.formatter.format_predictions(top_preds)
                    
                return self.safety_guard.check_response(response)
                
        return "I'm not sure how to help with that. Please describe your symptoms."
'''
}

base_path = "c:/Users/urbra/OneDrive/Desktop/Projects/MediAssist/src/conversational/"

for filename, content in files.items():
    filepath = os.path.join(base_path, filename)
    with open(filepath, "w", encoding="utf-8") as f:
        f.write(content.strip() + "\n")
    print(f"Created {filepath}")
