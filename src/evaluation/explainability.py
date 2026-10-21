"""
MediAssist AI — Explainable AI (XAI) & Token Saliency Engine
Computes token-level gradient attribution (Input * Gradient) for fine-tuned PubMedBERT.
Generates clinical feature attribution heatmaps for peer-reviewed paper figures.
"""

import os
import json
import torch
import numpy as np
from pathlib import Path
from typing import Dict, Any, List

from src.models.transformers.predict_bert import MedicalBERTInferencePipeline, DEFAULT_MODEL_DIR, DATA_DIR

class TokenSaliencyExplainer:
    def __init__(self):
        self.pipeline = MedicalBERTInferencePipeline()
        self.model = self.pipeline.model
        self.tokenizer = self.pipeline.tokenizer
        self.device = self.pipeline.device
        self.label_mapping = self.pipeline.label_mapping
        self.inv_label_mapping = {v: k for k, v in self.label_mapping.items()}

    def explain_instance(self, text: str, target_disease: str = None) -> Dict[str, Any]:
        """
        Computes token-level attribution using Input * Gradient on the embedding layer.
        """
        if self.model is None or self.tokenizer is None:
            return {"error": "Transformer model not loaded"}

        self.model.eval()
        self.model.zero_grad()

        # Tokenize input
        inputs = self.tokenizer(
            text,
            return_tensors="pt",
            truncation=True,
            max_length=64
        ).to(self.device)

        input_ids = inputs["input_ids"]
        attention_mask = inputs["attention_mask"]

        # Get embedding layer
        embedding_layer = self.model.bert.embeddings.word_embeddings
        embeddings = embedding_layer(input_ids)
        embeddings.retain_grad()

        # Forward pass with custom embedding hook
        outputs = self.model(
            inputs_embeds=embeddings,
            attention_mask=attention_mask
        )
        logits = outputs.logits
        probs = torch.softmax(logits, dim=1).detach().cpu().numpy()[0]

        pred_idx = int(torch.argmax(logits, dim=1).item())
        predicted_disease = self.label_mapping.get(pred_idx, "Unknown")
        pred_confidence = float(probs[pred_idx])

        # Target class for gradient
        target_idx = self.inv_label_mapping.get(target_disease, pred_idx) if target_disease else pred_idx
        target_logit = logits[0, target_idx]

        # Backward pass to get gradient w.r.t input embeddings
        target_logit.backward()

        # Input * Gradient attribution
        grads = embeddings.grad.detach().cpu().numpy()[0]  # (seq_len, hidden_dim)
        embeds = embeddings.detach().cpu().numpy()[0]     # (seq_len, hidden_dim)

        # Dot product / norm of Input * Grad
        attributions = np.sum(grads * embeds, axis=1)    # (seq_len,)
        attributions = np.maximum(0, attributions)        # ReLU to focus on positive evidence

        # Normalize attributions to [0, 1]
        max_attr = np.max(attributions)
        if max_attr > 0:
            norm_attributions = attributions / max_attr
        else:
            norm_attributions = attributions

        # Decode tokens
        tokens = self.tokenizer.convert_ids_to_tokens(input_ids[0].cpu().numpy())

        # Filter out special tokens [CLS], [SEP], [PAD]
        token_attributions = []
        for tok, score in zip(tokens, norm_attributions):
            if tok in ["[CLS]", "[SEP]", "[PAD]"]:
                continue
            token_attributions.append({
                "token": tok,
                "attribution_score": round(float(score), 4)
            })

        # Identify top salient keywords
        sorted_tokens = sorted(token_attributions, key=lambda x: x["attribution_score"], reverse=True)

        return {
            "input_text": text,
            "predicted_condition": predicted_disease,
            "confidence": round(pred_confidence * 100, 2),
            "tokens": token_attributions,
            "top_salient_tokens": sorted_tokens[:6]
        }

    def run_case_studies(self) -> List[Dict[str, Any]]:
        test_vignettes = [
            {
                "case_id": "Case 1: Outpatient Primary Care",
                "text": "headache since three days with tight band pressure around forehead and scalp stiffness from work stress",
                "condition": "Tension-Type Headache"
            },
            {
                "case_id": "Case 2: Digital Eye Strain / CVS",
                "text": "burning sensation in eyes and tired blurry vision after working on computer monitor for ten hours",
                "condition": "Eye Strain / Screen Fatigue (CVS)"
            },
            {
                "case_id": "Case 3: Acute Rhinosinusitis",
                "text": "facial heaviness and severe sinus pressure behind cheekbones worse when bending forward with nasal congestion",
                "condition": "Acute Sinusitis / Sinus Pressure"
            },
            {
                "case_id": "Case 4: Functional Dyspepsia / GERD",
                "text": "heartburn and sour acidic fluid rising into throat with burning sensation behind breastbone after meals",
                "condition": "Acidity & Indigestion (Functional Dyspepsia)"
            },
            {
                "case_id": "Case 5: Endemic Tropical Polyarthralgia",
                "text": "severe crippling joint pain and swelling in wrists and ankles following viral fever episodes",
                "condition": "Chikungunya"
            }
        ]

        results = []
        print("\n" + "="*80)
        print("XAI TOKEN ATTRIBUTION CASE STUDIES (PubMedBERT)")
        print("="*80)
        
        for case in test_vignettes:
            res = self.explain_instance(case["text"])
            res["case_id"] = case["case_id"]
            res["expected_condition"] = case["condition"]
            results.append(res)
            
            print(f"\n[{case['case_id']}]")
            print(f"  Input: \"{case['text']}\"")
            print(f"  Prediction: {res['predicted_condition']} ({res['confidence']}%)")
            top_toks = ", ".join([f"{t['token']} ({t['attribution_score']})" for t in res['top_salient_tokens'][:4]])
            print(f"  Top Salient Tokens: {top_toks}")

        output_path = "reports/model_benchmarks/xai_attributions.json"
        with open(output_path, "w") as f:
            json.dump(results, f, indent=2)
            
        print(f"\n[Saved XAI Attribution Results to {output_path}]")
        return results

if __name__ == "__main__":
    explainer = TokenSaliencyExplainer()
    explainer.run_case_studies()
