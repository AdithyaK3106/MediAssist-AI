import os
import json
from pathlib import Path
from src.models.classical_ml.config import config
from src.models.classical_ml.utils import setup_logging

logger = setup_logging(__name__)

class FollowupEngine:
    def __init__(self):
        self.project_root = Path(config.PROJECT_ROOT)
        self.question_bank_path = self.project_root / "src/conversational/question_bank.json"
        self.question_bank = self._load_question_bank()
        
    def _load_question_bank(self):
        if self.question_bank_path.exists():
            with open(self.question_bank_path, "r") as f:
                return json.load(f)
        else:
            logger.warning(f"Question bank not found at {self.question_bank_path}. Using empty bank.")
            return {}
            
    def get_followup_questions(self, top_predictions, uncertainty, threshold=0.7):
        """
        Determine if follow-up is needed and return questions.
        
        Args:
            top_predictions (list): List of dicts with 'disease' and 'probability'.
            uncertainty (dict): Dict with 'entropy', 'top2_gap', etc.
            threshold (float): Confidence threshold.
            
        Returns:
            list: List of question strings.
        """
        if not top_predictions:
            return []
            
        max_conf = top_predictions[0]['probability']
        entropy = uncertainty.get('entropy', 0) if uncertainty else 0
        gap = uncertainty.get('top2_gap', 1) if uncertainty else 1
        
        # Follow-up decision rule
        if max_conf < threshold or entropy > 0.7 or gap < 0.2:
            logger.info("Follow-up needed based on uncertainty metrics.")
            
            questions = []
            # Get questions for top diseases
            for pred in top_predictions[:2]: # Look at top 2
                disease = pred['disease']
                if disease in self.question_bank:
                    questions.extend(self.question_bank[disease])
                    
            # Remove duplicates while preserving order
            questions = list(dict.fromkeys(questions))
            
            # Return top 3 questions
            return questions[:3]
        else:
            logger.info("Confidence is sufficient. No follow-up needed.")
            return []
