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

## 4. Synthetic Clinical Data Expansion & Key-Phrase Anchoring (Indian & Tropical Diseases)
To address the severe data scarcity of natural language presentations for underrepresented regional and tropical illnesses in India (such as *Chikungunya*, *Leptospirosis*, *Scrub Typhus*, *Kala-Azar*, and *Snake Envenomation*), the framework incorporates an **LLM-assisted Clinical Data Expansion Protocol**:
1. **Clinical Key-Phrase Anchoring**: Each target condition is bounded by ICMR/CDC pathognomonic anchors (e.g., black eschar scab in Scrub Typhus; calf muscle tenderness post-flood exposure in Leptospirosis; crippling symmetric polyarthralgia in Chikungunya).
2. **Multi-Persona Voice Variations**: Generative prompt engineering expands each clinical profile into 25–50 diverse conversational manifestations across four patient personas (anxious family member, casual/colloquial Indian vernacular with Hinglish loanwords, timeline-explicit reporting, and acute emergency escalation).
3. **Automated Triage Triangulation**: Each generated utterance is annotated with urgency tier, emergency veto eligibility, and typical day-onset trajectories prior to integration into the fine-tuning split.
