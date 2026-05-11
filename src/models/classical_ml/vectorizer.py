import os
from sklearn.feature_extraction.text import TfidfVectorizer
from .config import config
from .utils import save_vectorizer, load_vectorizer, setup_logging

logger = setup_logging(__name__)

class MedicalVectorizer:
    def __init__(self, model_name="default"):
        self.model_name = model_name
        self.max_features = config.VECTORIZER_CONFIG.get("max_features", 5000)
        self.ngram_range = tuple(config.VECTORIZER_CONFIG.get("ngram_range", [1, 2]))
        self.min_df = config.VECTORIZER_CONFIG.get("min_df", 2)
        
        self.vectorizer = TfidfVectorizer(
            max_features=self.max_features,
            ngram_range=self.ngram_range,
            min_df=self.min_df,
            stop_words='english' # Optional, but good for medical text if not already removed
        )

    def fit_transform(self, texts):
        logger.info(f"Fitting TF-IDF vectorizer for {self.model_name}")
        X = self.vectorizer.fit_transform(texts)
        save_vectorizer(self.vectorizer, self.model_name)
        return X

    def transform(self, texts):
        if not hasattr(self.vectorizer, 'vocabulary_'):
            try:
                self.vectorizer = load_vectorizer(self.model_name)
            except FileNotFoundError:
                logger.warning(f"No saved vectorizer found for {self.model_name}. You must fit first.")
                raise
        return self.vectorizer.transform(texts)

    def get_feature_names(self):
        return self.vectorizer.get_feature_names_out()
