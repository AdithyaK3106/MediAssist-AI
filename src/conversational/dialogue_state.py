class DialogueState:
    def __init__(self):
        self.stage = "GREETING" # GREETING, SYMPTOM_INPUT, FOLLOWUP, FINAL_RESPONSE
        self.pending_questions = []
        self.uncertainty_status = False
        
    def set_stage(self, stage):
        self.stage = stage
        
    def set_pending_questions(self, questions):
        self.pending_questions = questions
