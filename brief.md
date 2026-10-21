# 🏥 MediAssist AI — Session Brief & Architectural Roadmap

> **Document Version:** 1.3.0  
> **Date:** October 21, 2026  
> **Project Role:** Senior AI Engineer & Software Manager  
> **Target Deployment:** Laptop with NVIDIA GeForce RTX 4060 (8GB VRAM) via Docker & Ollama  
> **Test Status:** 18 / 18 Tests Passing (100%)  
> **Taxonomy Scale:** 35 Conditions (1,589 Consolidated Samples)

---

## 1. Executive Summary
**MediAssist AI** is a **Safety-Aware Bilingual AI Healthcare Triage Assistant**. The framework strictly operates as a **preliminary assistance and triage tool, NOT a diagnostic replacement**.

Following user feedback on low confidence (5%) for generic symptoms like *"headache since 3 days"*, the system was upgraded with a **high-prevalence Indian Outpatient Synthetic Dataset** and **Relative Likelihood Calibration**, expanding the taxonomy from 27 to **35 conditions**.

The primary technical contribution is the **defense-in-depth separation of responsibilities**:
$$\text{Conversation (Qwen 2.5 7B)} \longrightarrow \text{Emergency Firewall (Deterministic Veto)} \longrightarrow \text{Clinical BERT (PubMedBERT)} \longrightarrow \text{Decision Layer (Relative Likelihood)} \longrightarrow \text{Generator (Qwen 2.5 7B)}$$

---

## 2. Implemented Hybrid Architecture

```mermaid
flowchart TD
    User([User]) <--> UI[WhatsApp-Style Web UI\nBilingual: English / Hindi]
    
    subgraph S1 [1. Conversational Understanding]
        UI --> Orch[Backend Orchestrator\nPython State Machine]
        Orch --> QwenExtract[Qwen 2.5 7B: Conversational Parser\nOllama Native JSON Mode]
        QwenExtract --> JSON[Structured Profile:\nSymptoms + Timeline: Onset, Duration, Trend, Severity]
    end

    subgraph S2 [2. Independent Safety Firewall]
        JSON --> EmergGate{Emergency Safety Firewall\nDeterministic Veto Engine}
        EmergGate -->|🚨 Critical Emergency Detected| EmergEsc[Emergency Escalation Protocol\nImmediate Hospital / 112 Guidance]
    end

    subgraph S3 [3. Clinical Classification]
        EmergGate -->|Safe for Routine Triage| PubMedBERT[Fine-Tuned Medical BERT / Calibrated Baseline\n35 Outpatient & Endemic Conditions]
        PubMedBERT --> Top3[Top-3 Conditions + Probabilities]
    end

    subgraph S4 [4. Safety & Decision Layer]
        Top3 & JSON --> DecisionLayer[Clinical Decision Layer]
        DecisionLayer --> Prior[Epidemiological Prior Weighting]
        DecisionLayer --> RelLikelihood[Relative Likelihood % + Clinical Match Bands]
        DecisionLayer --> Urgency[Urgency Level: Low / Moderate / High]
        DecisionLayer --> CareGate[Home Care Gate: Blocked for High/Emergency]
        DecisionLayer --> HospRoute[Bangalore Hospital Routing: GPS + Specialty]
        DecisionLayer --> Bundle[(Safe Decision Bundle)]
    end

    subgraph S5 [5. Empathetic Synthesis]
        EmergEsc --> QwenGen[Qwen 2.5 7B: Response Generator]
        Bundle --> QwenGen
        QwenGen --> UI
    end
```

---

## 3. Core Modules & Implementation Status

| Module | Implementation File | Status | Description |
| :--- | :--- | :---: | :--- |
| **Synthetic Data Generator** | [`src/preprocessing/synthetic_data_generator.py`](file:///data/Projects/MediAssist/src/preprocessing/synthetic_data_generator.py) | ✅ Done | Generates 406 clinically authentic patient narratives across 8 high-frequency Indian outpatient complaints (Tension Headache, Eye Strain, Sinusitis, Acidity, Sore Throat, Dehydration, Low Back Strain, Chikungunya) in English, Indian English, and Hinglish. |
| **Dataset Consolidator** | [`src/preprocessing/dataset_consolidator.py`](file:///data/Projects/MediAssist/src/preprocessing/dataset_consolidator.py) | ✅ Done | Consolidates benchmark data + synthetic data into 1,589 samples across 35 conditions. Generates stratified train (1,112), val (238), test (239) splits and clinical metadata. |
| **Conversational Parser** | [`src/conversational/timeline_extractor.py`](file:///data/Projects/MediAssist/src/conversational/timeline_extractor.py) | ✅ Done | Native Qwen 2.5 7B JSON extractor via Ollama (`format: "json"`). Extracts canonical symptoms, onset, duration in days, severity, progression, and language. |
| **Emergency Safety Firewall** | [`src/safety/emergency_firewall.py`](file:///data/Projects/MediAssist/src/safety/emergency_firewall.py) | ✅ Done | Deterministic pre-inference veto engine (100% EEA, 0% MER). Intercepts stroke, cardiac, respiratory, hemorrhagic, and severe CNS infections in English and Hindi. |
| **Medical BERT Classifier** | [`src/models/transformers/predict_bert.py`](file:///data/Projects/MediAssist/src/models/transformers/predict_bert.py)<br>[`src/models/classical_ml/train_consolidated.py`](file:///data/Projects/MediAssist/src/models/classical_ml/train_consolidated.py) | ✅ Done | 35-class classifier with Top-3 condition prediction, entropy estimation, and safe GPU/CPU execution. Calibrated baseline achieves 88.28% test accuracy. |
| **Clinical Decision Layer** | [`src/safety/decision_layer.py`](file:///data/Projects/MediAssist/src/safety/decision_layer.py) | ✅ Done | Computes urgency tiers (*Low*, *Moderate*, *High*), gates home care (blocked for high-severity), applies clinical prior weighting for outpatient complaints, and calculates **Relative Likelihood (%)** with Match Bands. |
| **Response Generator** | [`src/models/llm/qwen_client.py`](file:///data/Projects/MediAssist/src/models/llm/qwen_client.py) | ✅ Done | Modular client for Ollama Qwen 2.5 7B (`host.docker.internal:11434`) with native bilingual generator fallback for dev machines. |
| **Master Orchestrator** | [`src/conversational/triage_orchestrator.py`](file:///data/Projects/MediAssist/src/conversational/triage_orchestrator.py) | ✅ Done | Coordinates all 5 layers, multi-turn session memory, 1-round follow-up limit, and emergency re-evaluation. Passes combined conversation text and symptoms to classification. |
| **WhatsApp Web UI** | [`src/frontend/app.py`](file:///data/Projects/MediAssist/src/frontend/app.py)<br>[`src/frontend/styles/whatsapp.css`](file:///data/Projects/MediAssist/src/frontend/styles/whatsapp.css) | ✅ Done | WhatsApp-themed interface with chat bubbles, timestamps, blue checkmarks, language switcher, and a live clinical audit inspector. |
| **Deployment Scripts** | [`run_laptop.sh`](file:///data/Projects/MediAssist/run_laptop.sh)<br>[`run_laptop.bat`](file:///data/Projects/MediAssist/run_laptop.bat) | ✅ Done | One-click launcher scripts for Linux/macOS and Windows that auto-pull `qwen2.5:7b` in Ollama and launch Docker containers. |
| **Research & Presentation** | [`research/PAPER_DRAFT.md`](file:///data/Projects/MediAssist/research/PAPER_DRAFT.md)<br>[`research/LITERATURE_REVIEW.md`](file:///data/Projects/MediAssist/research/LITERATURE_REVIEW.md)<br>[`research/PRESENTATION_SLIDES_28PT.md`](file:///data/Projects/MediAssist/research/PRESENTATION_SLIDES_28PT.md) | ✅ Done | Full IEEE conference paper, 10 verified peer-reviewed papers (2023–2026), and 9-section PPT slides formatted for 28 pt font. |

---

## 4. What Was Tested & Verified

1. **Resolution of Sparse Query Dilution ("headache since 3 days")**:
   * **Earlier**: 5% confidence split across Hypertension, Malaria, and Chicken Pox.
   * **Now**: **Tension-Type Headache** is predicted as **Primary Consideration (90.4% relative likelihood)** with 5 comprehensive supportive home care instructions and explicit red flags.
2. **Indian Outpatient Presentations Tested**:
   * *Gale me kharash aur nigalne me dard (3 days)* $\rightarrow$ **Viral Pharyngitis (Sore Throat)** (57.1% - Primary Consideration).
   * *Severe heartburn & sour burps after dinner* $\rightarrow$ **Acidity & Indigestion (Dyspepsia)** (72.9% - Primary Consideration).
   * *Dhoop me weakness, dry mouth, dark urine* $\rightarrow$ **Dehydration & Heat Fatigue** (81.4% - Primary Consideration).
   * *Post-fever crippling joint pain in wrists/ankles* $\rightarrow$ **Chikungunya** (61.8% - Primary Consideration).
3. **Automated Test Suite**:
   * **18 / 18 Tests Passing (100%)** via `pytest tests/ -v`.
