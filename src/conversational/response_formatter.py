class ResponseFormatter:
    def format_predictions(self, top_predictions, upr=None):
        return "I have analyzed your symptoms. Please refer to the insights and guidance panels on the right for supportive recommendations."
        
    def format_questions(self, questions):
        response = "To help me provide better guidance, could you answer a few quick questions?\n\n"
        for i, q in enumerate(questions):
            response += f"{i+1}. {q}\n"
        return response
