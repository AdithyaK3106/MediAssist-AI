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
