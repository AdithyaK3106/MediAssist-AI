from src.conversational.timeline_extractor import SymptomTimelineExtractor
from src.safety.emergency_firewall import EmergencySafetyFirewall
from src.models.transformers.predict_bert import MedicalBERTInferencePipeline
from src.safety.decision_layer import ClinicalDecisionLayer
from src.models.llm.qwen_client import QwenLLMClient

class MediAssistTriageOrchestrator:
    """
    Master dialogue and safety orchestrator implementing the 5-layer architecture:
    User -> Parser -> Emergency Firewall -> Medical BERT -> Decision Layer -> Bilingual Generator
    """
    def __init__(self):
        self.llm_client = QwenLLMClient()
        self.extractor = SymptomTimelineExtractor(llm_client=self.llm_client)
        self.emergency_firewall = EmergencySafetyFirewall()
        self.bert_pipeline = MedicalBERTInferencePipeline()
        self.decision_layer = ClinicalDecisionLayer()

        # Session state
        self.accumulated_text = []
        self.pending_followup = False
        self.last_profile = {}
        self.last_decision = {}

    def reset(self):
        self.accumulated_text = []
        self.pending_followup = False
        self.last_profile = {}
        self.last_decision = {}

    def process_message(self, user_message: str, user_lat=12.9716, user_lon=77.5946) -> dict:
        self.accumulated_text.append(user_message)
        combined_text = " ".join(self.accumulated_text)

        # LAYER 1: Parse symptoms & timeline
        profile = self.extractor.extract(combined_text)
        self.last_profile = profile
        language = profile.get("language", "en")

        # LAYER 2: Emergency Safety Firewall (Pre-Inference Veto)
        emergency_result = self.emergency_firewall.evaluate(combined_text, structured_profile=profile)
        if emergency_result.get("is_emergency"):
            # HALT all predictions immediately
            msg = emergency_result["guidance_hi"] if language == "hi" else emergency_result["guidance_en"]
            contacts = emergency_result.get("contacts", {})
            contacts_text = f"\n\n📞 Emergency: {contacts.get('national_emergency', '112')} | Ambulance: {contacts.get('ambulance', '108')}"
            
            return {
                "status": "EMERGENCY_OVERRIDE",
                "is_emergency": True,
                "message": msg + contacts_text,
                "profile": profile,
                "decision": None
            }

        # Clarifying follow-up check (limit to 1 round)
        if not self.pending_followup:
            followup = self.extractor.generate_followup(profile)
            if followup.get("needs_followup"):
                self.pending_followup = True
                questions_formatted = "\n".join([f"• {q}" for q in followup["questions"]])
                prompt_intro = "कृपया मुझे और समझने में मदद करें:" if language == "hi" else "To help assess your symptoms accurately, please answer:"
                
                return {
                    "status": "FOLLOWUP_PENDING",
                    "is_emergency": False,
                    "message": f"{prompt_intro}\n\n{questions_formatted}",
                    "profile": profile,
                    "decision": None
                }

        # LAYER 3: Medical BERT Condition Classification
        symptoms_str = " ".join(profile.get("symptoms", []))
        symptom_query = f"{combined_text} {symptoms_str}".strip() if symptoms_str else combined_text

        bert_res = self.bert_pipeline.predict([symptom_query])[0]
        bert_preds = bert_res.get("predictions", [])

        # LAYER 4: Clinical Decision Layer
        decision_bundle = self.decision_layer.evaluate(
            bert_predictions=bert_preds,
            timeline=profile.get("timeline", {}),
            user_lat=user_lat,
            user_lon=user_lon
        )
        self.last_decision = decision_bundle

        # LAYER 5: Bilingual Response Synthesis
        final_response = self.llm_client.generate_triage_response(
            decision_bundle=decision_bundle,
            timeline=profile.get("timeline", {}),
            language=language
        )

        return {
            "status": "ASSESSMENT_COMPLETE",
            "is_emergency": False,
            "message": final_response,
            "profile": profile,
            "decision": decision_bundle
        }
