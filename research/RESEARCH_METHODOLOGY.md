# MediAssist AI — Research & Methodology Notes

## 1. Project Title
**MediAssist AI: A Safety-Aware Bilingual Healthcare Triage Framework Combining Pre-Trained Transformers and LLM-Guided Dialogue**

## 2. Core Research Hypotheses & Thesis
1. **Separation of Safety & Inference**: Pure probabilistic models (both classical ML and large autoregressive LLMs) are inherently prone to treating critical, life-threatening symptoms as statistical tokens. Decoupling deterministic emergency rule gates from probabilistic classification reduces the Missed Emergency Rate (MER) to near zero.
2. **Specialized Clinical Representations**: A domain-specific transformer (PubMedBERT) fine-tuned on clinical symptom profiles achieves higher condition classification reliability and calibration compared to general-domain LLMs alone.
3. **Timeline-Aware Urgency Gating**: Incorporating symptom timeline (onset, duration, severity progression) provides the necessary clinical context to prevent over-triage or dangerous under-triage.
4. **Bilingual Equity**: Safe conversational healthcare triage can be effectively provided in both English and Hindi without degrading safety constraints.

## 3. Architecture Overview
- **Conversational Layer**: Qwen 2.5 7B (Bilingual Natural Language Understanding & Generation)
- **Safety Layer 1**: Pre-Inference Deterministic Emergency Firewall (Veto Gate)
- **Classifier**: Fine-Tuned PubMedBERT (`microsoft/BiomedNLP-PubMedBERT-base-uncased-abstract-fulltext`)
- **Safety Layer 2**: Post-Inference Decision & Urgency Layer (Timeline integration, severity gating, hospital routing)
- **Evaluation Metrics**:
  - Emergency Escalation Accuracy (EEA)
  - Missed Emergency Rate (MER)
  - Top-1 and Top-3 Condition Accuracy
  - Calibrated Urgency Alignment
