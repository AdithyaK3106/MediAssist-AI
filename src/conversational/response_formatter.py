class ResponseFormatter:
    def format_predictions(self, top_predictions, hospitals=None):
        response = "### Possible Conditions:\n"
        for i, pred in enumerate(top_predictions[:3]):
            response += f"{i+1}. {pred['disease']} — {pred['probability']*100:.1f}%\n"
            
        if hospitals:
            response += "\n### 🏥 Recommended Hospitals Nearby:\n"
            for h in hospitals:
                specialties = ", ".join(h['specialties'][:3])
                response += f"- **{h['name']}** ({h['city']}) | Specialties: {specialties}\n"

        response += "\n### Suggested Action:\n"
        response += "Please consult a healthcare professional for an accurate diagnosis."
        return response
        
    def format_questions(self, questions):
        response = "To help me narrow down the possibilities, could you answer a few questions?\n\n"
        for i, q in enumerate(questions):
            response += f"{i+1}. {q}\n"
        return response
