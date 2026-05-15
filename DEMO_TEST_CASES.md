# 🏥 MediAssist AI — Demo Test Cases

> **Purpose:** This document contains diverse, carefully selected test cases to showcase every major feature and safety layer of the MediAssist AI system during a live demonstration.

---

## 📋 Table of Contents

1. [Feature Overview](#-feature-overview)
2. [Feature 1 — Emergency Rule Engine (Pre-ML Triage)](#-feature-1--emergency-rule-engine-pre-ml-triage)
3. [Feature 2 — Uncertainty-Aware ML Predictions](#-feature-2--uncertainty-aware-ml-predictions)
4. [Feature 3 — Medical Consistency Validation & Reranking](#-feature-3--medical-consistency-validation--reranking)
5. [Feature 4 — Conversational Memory & Follow-up Engine](#-feature-4--conversational-memory--follow-up-engine)
6. [Feature 5 — Home Care Recommendations & Severity Gating](#-feature-5--home-care-recommendations--severity-gating)
7. [Feature 6 — Audit Logging & Explainability](#-feature-6--audit-logging--explainability)
8. [Feature 7 — Unsafe Prediction Rate (Safety Metrics)](#-feature-7--unsafe-prediction-rate-safety-metrics)
9. [Feature 8 — India-Specific Medical Intelligence (Regional Awareness)](#-feature-8--india-specific-medical-intelligence-regional-awareness)
10. [Feature 9 — Real-World Hospital Integration (Proximity & Capability)](#-feature-9--real-world-hospital-integration-proximity--capability)
11. [Edge Cases & Stress Tests](#-edge-cases--stress-tests)
12. [End-to-End Demo Walkthrough Script](#-end-to-end-demo-walkthrough-script)

---

## 🌟 Feature Overview

| # | Feature | Module | Key Outcome |
|---|---------|--------|-------------|
| 1 | Emergency Rule Engine | `src/safety/emergency_rules.py` | Bypasses ML and shows 🚨 Critical Alert |
| 2 | Uncertainty-Aware Predictions | `src/models/classical_ml/predict.py` | Returns Top-K + entropy/gap metrics |
| 3 | Medical Consistency Validation | `src/safety/medical_consistency.py` | Penalizes implausible predictions |
| 4 | Conversational Memory & Follow-ups | `src/conversational/` | Asks clarifying questions when unsure |
| 5 | Home Care Recommendations | `src/recommendation/home_care.py` | Safe remedies only for non-critical cases |
| 6 | Audit Logging | `logs/safety/` | JSON logs of all safety overrides |
| 7 | Safety Metrics (EEA / MER / UPR) | `tests/test_safety.py` | Quantified healthcare safety performance |
| 8 | India-Specific Intelligence | `data/raw/india_medical_dataset_expansion.csv` | High-precision triage for regional diseases |
| 9 | Hospital Integration | `data/raw/hospitals/hospital_directory.csv` | Capability-aware proximity search |

---

## 🚨 Feature 1 — Emergency Rule Engine (Pre-ML Triage)

> **What to demonstrate:** The system intercepts **life-threatening inputs BEFORE the ML model runs** and immediately escalates to a critical alert, bypassing any probabilistic scoring.

### TC-E01 — Stroke Symptoms 🔴
| | |
|---|---|
| **Input** | `"I suddenly have slurred speech and one-sided facial weakness"` |
| **Expected Behaviour** | 🚨 Critical Alert triggered. Escalation message displayed immediately. No disease prediction shown. |
| **Detected Category** | `Neurological Emergency` |
| **Triggered Keywords** | `slurred speech`, `facial weakness` |
| **Key Point to Explain** | The ML model is completely bypassed. The rule engine fires before TF-IDF vectorization. |

---

### TC-E02 — Respiratory Emergency 🔴
| | |
|---|---|
| **Input** | `"I am coughing up blood and can't breathe properly"` |
| **Expected Behaviour** | 🚨 Critical Alert. No probability scores returned. |
| **Detected Category** | `Respiratory Emergency` |
| **Triggered Keywords** | `coughing up blood`, `can't breathe` |
| **Key Point to Explain** | Even if the ML might predict "Asthma" (due to feature overlap), the rule engine intervenes first — showing the value of deterministic safety over probabilistic inference. |

---

### TC-E03 — Cardiac Emergency 🔴
| | |
|---|---|
| **Input** | `"I have crushing chest pain and the pain is radiating to my left arm"` |
| **Expected Behaviour** | 🚨 Critical Alert. Immediate escalation. |
| **Detected Category** | `Cardiac Emergency` |
| **Triggered Keywords** | `crushing chest pain`, `pain radiating to arm` |
| **Key Point to Explain** | Heart attack patterns are hardcoded as non-negotiable safety rules, independent of training data biases. |

---

### TC-E04 — Severe Infection (Meningitis Pattern) 🔴
| | |
|---|---|
| **Input** | `"I have a very high fever with stiff neck and extreme confusion"` |
| **Expected Behaviour** | 🚨 Critical Alert. Multiple categories may fire simultaneously. |
| **Detected Categories** | `Severe Infection` |
| **Triggered Keywords** | `stiff neck`, `extreme confusion`, `very high fever` |
| **Key Point to Explain** | The engine detects **multiple emergency signals** in a single input and reports all. |

---

### TC-E05 — Hemorrhagic Warning 🔴
| | |
|---|---|
| **Input** | `"I have been having bleeding gums and blood in my vomit since morning"` |
| **Expected Behaviour** | 🚨 Critical Alert. Hemorrhagic pattern detected. |
| **Detected Category** | `Hemorrhagic Warning` |
| **Triggered Keywords** | `bleeding gums`, `blood in vomit` |
| **Key Point to Explain** | Dengue hemorrhagic fever and internal bleeding are patterns that ML models often mislabel. The rule engine catches them at zero cost. |

---

### TC-E06 — No Emergency (Baseline) ✅
| | |
|---|---|
| **Input** | `"I have a mild headache and a runny nose since yesterday"` |
| **Expected Behaviour** | No alert triggered. System proceeds normally to ML pipeline. |
| **Detected Category** | None |
| **Key Point to Explain** | Demonstrates **precision** — the engine does NOT over-trigger. Only genuine emergencies are escalated. |

---

## 🔮 Feature 2 — Uncertainty-Aware ML Predictions

> **What to demonstrate:** For non-emergency inputs, the ML pipeline returns **Top-3 predictions with probability scores, entropy, and Top-2 gap** instead of a single confident (and potentially wrong) guess.

### TC-U01 — Clear, High-Confidence Prediction ✅
| | |
|---|---|
| **Input** | `"I have a runny nose, mild cough, sneezing, and slight sore throat"` |
| **Expected Top Prediction** | `Common Cold` (~75–85% confidence) |
| **Expected Uncertainty** | Low entropy, high Top-2 gap (model is confident) |
| **flag_unreliable** | `False` |
| **Key Point to Explain** | System identifies a clean symptom profile. Low entropy means prediction is reliable. |

---

### TC-U02 — Ambiguous Input Triggering High Uncertainty ⚠️
| | |
|---|---|
| **Input** | `"I feel tired and have some mild pain"` |
| **Expected Behaviour** | Top-3 predictions shown with closely spaced probabilities. Follow-up questions triggered. |
| **Expected Uncertainty** | High entropy (> 0.7), small Top-2 gap (< 0.2) |
| **flag_unreliable** | `True` |
| **Key Point to Explain** | Vague symptoms create high uncertainty. Instead of confidently guessing wrong, the system flags the prediction as unreliable and asks follow-up questions. |

---

### TC-U03 — Multi-Symptom Respiratory Case
| | |
|---|---|
| **Input** | `"I have been coughing with yellow mucus, chest tightness, and mild fever for three days"` |
| **Expected Top Predictions** | `Pneumonia`, `Bronchitis`, `Flu` |
| **Expected Uncertainty** | Moderate entropy; follow-up questions may be triggered |
| **Key Point to Explain** | Several respiratory diseases share overlapping features. The **Top-K approach** honestly shows all plausible diagnoses rather than masking uncertainty with false confidence. |

---

### TC-U04 — Digestive Symptoms
| | |
|---|---|
| **Input** | `"I have been vomiting and have diarrhea and severe stomach cramps"` |
| **Expected Top Predictions** | `Gastroenteritis`, `Food Poisoning`, `Irritable Bowel Syndrome` |
| **Expected Uncertainty** | Moderate. Predictions should be within the gastrointestinal category. |
| **Key Point to Explain** | The TF-IDF vectorizer picks up domain-specific gastro terms and clusters them correctly. |

---

### TC-U05 — Skin Condition
| | |
|---|---|
| **Input** | `"I have a red itchy rash with small blisters on my arm and it keeps spreading"` |
| **Expected Top Predictions** | `Contact Dermatitis`, `Chickenpox`, `Impetigo` |
| **Key Point to Explain** | Demonstrates the system's breadth across non-respiratory disease domains. |

---

## 🔍 Feature 3 — Medical Consistency Validation & Reranking

> **What to demonstrate:** After ML prediction, a post-processing safety layer **penalizes predictions that are biologically implausible** given the symptoms described. This corrects for training data biases.

### TC-C01 — Implausible Prediction Suppression 🎯
| | |
|---|---|
| **Input** | `"I have high fever and a persistent cough"` |
| **Initial ML Predictions (simulated)** | `Common Cold (0.80)`, `Conjunctivitis (0.70)` |
| **After Consistency Check** | `Common Cold (0.80)` stays. `Conjunctivitis` penalized to `0.07` (0.70 × 0.1 penalty) |
| **Detected Symptom Categories** | `fever`, `respiratory` |
| **Conjunctivitis Disease Categories** | `eye` — no overlap |
| **Consistency Score** | `LOW → Penalty Applied` |
| **Key Point to Explain** | An eye infection predicted for fever + cough is biologically incoherent. The consistency layer catches this where the ML model fails. |

---

### TC-C02 — Plausible Prediction Preserved ✅
| | |
|---|---|
| **Input** | `"I have stomach pain, nausea, and diarrhea after eating"` |
| **Predictions** | `Food Poisoning`, `Gastroenteritis`, `GERD` |
| **Consistency Score** | `HIGH` for all three (GI overlap confirmed) |
| **Penalty Applied** | `1.0` (no penalty) |
| **Key Point to Explain** | When the ML output matches the symptom profile, the consistency layer passes predictions through untouched. |

---

### TC-C03 — Partial Overlap with Mixed Symptoms
| | |
|---|---|
| **Input** | `"I have a sore throat, runny nose, and joint pain"` |
| **Predictions** | `Flu (0.75)`, `Common Cold (0.65)`, `Arthritis (0.40)` |
| **After Consistency Check** | `Flu` and `Common Cold` preserved (systemic + respiratory overlap). `Arthritis` downranked (musculoskeletal only — no respiratory overlap with input). |
| **Key Point to Explain** | Shows fine-grained reranking, not just binary suppression. The model adjusts confidences based on multi-category symptom analysis. |

---

## 💬 Feature 4 — Conversational Memory & Follow-up Engine

> **What to demonstrate:** When the model is uncertain, it asks **targeted, disease-specific follow-up questions** drawn from a medical question bank. It also remembers symptoms across turns.

### TC-F01 — Follow-up Triggered by Low Confidence
| | |
|---|---|
| **Turn 1 Input** | `"I have been feeling unwell lately"` |
| **System Response** | Follow-up questions triggered (high entropy). E.g., *"Are you experiencing body aches?"*, *"Do you have a fever?"*, *"Do you have a sore throat?"* |
| **Turn 2 Input** | `"Yes, I have body aches and a high fever"` |
| **System Response** | Prediction refined with combined context. Likely `Flu` gains confidence. |
| **Key Point to Explain** | The `SymptomMemory` accumulates symptom context across turns. The `FollowupEngine` uses the `question_bank.json` to generate disease-relevant questions. |

---

### TC-F02 — Follow-up for Asthma Differentiation
| | |
|---|---|
| **Turn 1 Input** | `"I'm having trouble breathing and my chest feels tight"` |
| **System Response** | Questions: *"Do you experience shortness of breath?"*, *"Is there any wheezing when you breathe?"*, *"Do you feel chest tightness?"* |
| **Turn 2 Input** | `"Yes, there's a wheezing sound and it gets worse at night"` |
| **System Response** | `Asthma` confidence increases. GERD and anxiety downranked. |
| **Key Point to Explain** | The follow-up questions are **disease-targeted**, not generic. They come from the curated `question_bank.json` linked to the top predictions. |

---

### TC-F03 — Follow-up Skipped (High Confidence)
| | |
|---|---|
| **Input** | `"I have a fever of 102F, body aches all over, sore throat, runny nose, and I feel exhausted"` |
| **Expected Behaviour** | No follow-up questions. Final prediction returned directly. |
| **Top Prediction** | `Flu` |
| **Key Point to Explain** | When **entropy is low and the Top-2 gap is large**, follow-up is skipped. The model is confident enough to respond. |

---

### TC-F04 — Multi-Turn Memory Accumulation
| | |
|---|---|
| **Turn 1** | `"I've had a headache for two days"` |
| **Turn 2** | `"I also notice I'm very sensitive to light"` |
| **Turn 3** | `"And I feel nauseous when the headache peaks"` |
| **Expected Behaviour** | System accumulates all three inputs. Prediction by Turn 3 should favour `Migraine`. |
| **Key Point to Explain** | Unlike a stateless single-turn system, MediAssist's `SymptomMemory` builds a richer picture over a conversation session. |

---

## 🏠 Feature 5 — Home Care Recommendations & Severity Gating

> **What to demonstrate:** Home care remedies are **only provided for non-critical conditions**. For critical diseases, the system immediately escalates to "seek emergency care" with no home remedy suggestions.

### TC-H01 — Home Care for Non-Critical Condition ✅
| | |
|---|---|
| **Disease Predicted** | `Common Cold` |
| **Expected Home Care** | "Rest adequately", "Stay hydrated", "Use saline nasal spray", "Warm fluids like soup or tea" |
| **Severity** | `low` |
| **Emergency Escalation** | None |
| **Key Point to Explain** | The `HomeCareRecommender` safely provides wellness advice for minor conditions. |

---

### TC-H02 — Home Care Blocked for Critical Disease 🚫
| | |
|---|---|
| **Disease Predicted** | `Pneumonia` |
| **Expected Home Care** | ❌ Empty — NO home remedies provided |
| **Expected Message** | `"Immediately seek emergency medical care"` |
| **Severity** | `critical` |
| **Key Point to Explain** | The `is_critical()` method detects critical disease keywords (`pneumonia`, `stroke`, `tuberculosis`, etc.) and **removes the home_care list entirely** to prevent false safety. |

---

### TC-H03 — Dengue Severe — No Remedies Shown 🚫
| | |
|---|---|
| **Disease Predicted** | `Severe Dengue` |
| **Expected Behaviour** | `home_care: []` — escalation message only. |
| **Key Point to Explain** | "Severe dengue" is in the critical keywords list. Home remedies (like drinking papaya juice — a common folk remedy) are explicitly blocked. |

---

### TC-H04 — Fallback for Unknown Disease ⚠️
| | |
|---|---|
| **Disease Predicted** | `Rare Syndrome X` (not in database) |
| **Expected Home Care** | Generic fallback: *"Stay hydrated"*, *"Get adequate rest"*, *"Monitor symptoms carefully"* |
| **Seek Medical Attention If** | *"Symptoms worsen significantly"*, *"Fever exceeds 102°F"*, *"Breathing difficulty develops"* |
| **Key Point to Explain** | The system gracefully degrades to a safe, generic fallback — never returns an empty or broken response. |

---

## 📝 Feature 6 — Audit Logging & Explainability

> **What to demonstrate:** Every safety decision is recorded in **structured JSON logs** in `logs/safety/` for full clinical auditability.

### TC-L01 — Emergency Override Log
| | |
|---|---|
| **Trigger Input** | `"I have crushing chest pain and irregular heartbeat"` |
| **Log File** | `logs/safety/emergency_overrides.log` |
| **Log Contains** | `timestamp`, `input`, `detected_categories: ["Cardiac Emergency"]`, `triggered_symptoms`, `escalation message` |
| **Key Point to Explain** | Every emergency override is **timestamped and logged** — important for healthcare audits and research. |

---

### TC-L02 — Consistency Penalty Log
| | |
|---|---|
| **Trigger Input** | Any input where a prediction is suppressed (e.g., eye disease for fever + cough) |
| **Log File** | `logs/safety/medical_consistency.log` |
| **Log Contains** | `detected_symptoms`, `reranking_changes`, `suppressed_predictions`, `penalties_applied` |
| **Key Point to Explain** | The system provides a complete **audit trail of why a prediction was downranked** — not just a black box. This is critical for clinical explainability. |

---

## 📊 Feature 7 — Unsafe Prediction Rate (Safety Metrics)

> **What to demonstrate:** MediAssist evaluates itself against healthcare-specific safety metrics that go beyond standard ML accuracy.

### TC-M01 — Emergency Escalation Accuracy (EEA) ✅
| Metric | Target | Achieved |
|--------|--------|----------|
| Emergency Escalation Accuracy (EEA) | > 95% | **100%** (on 5-case suite) |
| Missed Emergency Rate (MER) | ~0% | **0%** |

**Test Cases Used:**
```
"coughing up blood"        → True Emergency  → Correctly Escalated ✅
"mild headache"            → Not Emergency   → Correctly Passed ✅
"slurred speech"           → True Emergency  → Correctly Escalated ✅
"bleeding gums"            → True Emergency  → Correctly Escalated ✅
"runny nose"               → Not Emergency   → Correctly Passed ✅
```

---

### TC-M02 — Consistency Suppression Verification
| | |
|---|---|
| **Input** | `"I have a high fever and cough"` |
| **Original Predictions** | `Common Cold (0.80)`, `Impetigo (0.70)` |
| **After Penalty** | `Common Cold (0.80)`, `Impetigo (0.07)` |
| **Assertion** | `impetigo_probability <= 0.07` ✅ |
| **Key Point to Explain** | The 90% penalty (×0.1 multiplier) is quantifiably verifiable in the test suite. |

---

### TC-M03 — Running the Full Safety Test Suite
```bash
# Run from project root
pytest tests/test_safety.py -v
```

**Expected Output:**
```
tests/test_safety.py::test_stroke_emergency                  PASSED ✅
tests/test_safety.py::test_respiratory_emergency             PASSED ✅
tests/test_safety.py::test_severe_infection                  PASSED ✅
tests/test_safety.py::test_hemorrhagic_warning               PASSED ✅
tests/test_safety.py::test_no_emergency                      PASSED ✅
tests/test_safety.py::test_implausible_prediction_suppression PASSED ✅
tests/test_safety.py::test_safety_metrics                    PASSED ✅
============================== 7 passed in X.XXs ==============================
```

---

## 🧪 Edge Cases & Stress Tests

### TC-X01 — Mixed Emergency + Normal Symptoms
| | |
|---|---|
| **Input** | `"I have a runny nose but also have slurred speech suddenly"` |
| **Expected Behaviour** | 🚨 Emergency rule fires despite the benign "runny nose" prefix. |
| **Key Point to Explain** | The rule engine is a substring search — it fires if ANY emergency keyword is present, regardless of surrounding context. Safety always wins. |

---

### TC-X02 — Partial / Typo Input
| | |
|---|---|
| **Input** | `"cant breathe"` (no apostrophe) |
| **Expected Behaviour** | 🚨 Respiratory Emergency triggered. |
| **Triggered Keyword** | `cant breathe` (both `cant breathe` and `can't breathe` are in the rule set) |
| **Key Point to Explain** | The rule engine has both punctuation variants of critical phrases. |

---

### TC-X03 — Very Short / Uninformative Input
| | |
|---|---|
| **Input** | `"I feel sick"` |
| **Expected Behaviour** | No emergency. ML prediction with high entropy. Follow-up questions triggered. |
| **Key Point to Explain** | The system gracefully handles vague inputs by asking clarifying questions rather than guessing. |

---

### TC-X04 — Greeting Intent
| | |
|---|---|
| **Input** | `"Hello"` / `"Hi"` |
| **Expected Behaviour** | `"Hello! I am MediAssist. Please describe your symptoms."` |
| **Key Point to Explain** | The `IntentDetector` correctly identifies non-medical intents and routes them to a friendly response. |

---

### TC-X05 — Dual Emergency Categories
| | |
|---|---|
| **Input** | `"I have severe chest pain and I also have slurred speech"` |
| **Expected Behaviour** | Both `Cardiac Emergency` AND `Neurological Emergency` are detected and reported. |
| **Key Point to Explain** | The engine accumulates all matching categories — it doesn't stop at the first match. |

---

### TC-X06 — Critical Disease via Home Care Check
| | |
|---|---|
| **Simulated Disease** | `"Heart Failure"` |
| **`is_critical()` Return** | `True` (keyword match: `heart failure`) |
| **Home Care Output** | `home_care: []`, `severity: critical`, seek emergency care message. |
| **Key Point to Explain** | Even if the emergency rule engine is bypassed (e.g., symptom was ambiguous), the home care layer adds a second line of defence. |

---

## 🎬 End-to-End Demo Walkthrough Script

> Use this scripted sequence for a smooth, uninterrupted live demo (~10 minutes).

### Step 1 — Launch App
```bash
streamlit run src/frontend/app.py
```

---

### Step 2 — Show Normal Flow (Feature 2 + 4 + 5)
> **Type:** `"I have a runny nose and mild sore throat"`
- Point out: Top-3 predictions with probability bars.
- Point out: Follow-up questions if confidence is low.
- Point out: Home care section for Common Cold (non-critical).

---

### Step 3 — Trigger Emergency Override (Feature 1)
> **Type:** `"I am coughing up blood and can't breathe"`
- Point out: 🚨 Critical Alert replaces normal prediction UI.
- Point out: No disease prediction shown — ML was bypassed.
- Point out: Escalation message and emergency instructions.

---

### Step 4 — Show Consistency Validation (Feature 3)
> Open `logs/safety/medical_consistency.log` in a terminal.
> Describe a fever/cough scenario and show the suppressed `Conjunctivitis` entry.
- Point out: `penalty_applied: 0.1`, `consistency: LOW`.
- Point out: Reranking changes from rank #2 to rank #5.

---

### Step 5 — Show Audit Logs (Feature 6)
> Open `logs/safety/emergency_overrides.log`.
- Point out: Timestamp, input, triggered_symptoms, escalation message all recorded in JSON.
- Say: *"Every safety decision is logged — this is what makes it auditable for clinical settings."*

---

### Step 6 — Run Safety Test Suite (Feature 7)
```bash
pytest tests/test_safety.py -v
```
- Show all 7 tests passing.
- Highlight the EEA = 100% and MER = 0% result.

---

### Step 7 — Critical Disease Home Care Block (Feature 5)
> Simulate/show a `Pneumonia` prediction.
- Point out: Home care section is blank.
- Point out: Only "Seek Emergency Medical Care" is shown.
- Contrast with Common Cold which shows full home remedies.

---

## ⚖️ Medical Disclaimer

> **MediAssist AI is a research and educational triage framework — NOT a clinical diagnostic tool.**
> All test cases above are for demonstration and academic evaluation purposes only.
> Always consult a qualified healthcare professional for any medical concerns.

---

*Last updated: May 2026 | MediAssist AI — AdithyaK3106/MediAssist-AI*


---

## 🌏 Feature 8 — India-Specific Medical Intelligence (Regional Awareness)

> **What to demonstrate:** The system has been trained on a **15,000-row India-focused dataset expansion**, enabling high-precision triage for tropical, rural, and regional health conditions.

### TC-I01 — Vector-Borne: Malaria (Classic Pattern) ✅
| | |
|---|---|
| **Input** | "I have high fever with shivering and cold rigors every two days" |
| **Expected Top Prediction** | `Malaria` (~85% confidence) |
| **Consistency Pass** | `High` (fever + systemic overlap) |
| **Key Point to Explain** | The model recognizes the "cyclical fever" pattern common in Indian malaria cases. |

---

### TC-I02 — Vector-Borne: Dengue (Hemorrhagic Pattern) 🚨
| | |
|---|---|
| **Input** | "I have high fever, pain behind my eyes, and red spots on my skin" |
| **Expected Top Prediction** | `Dengue` |
| **Safety Intervention** | 🚨 **Emergency Trigger** (Hemorrhagic spots detected) |
| **Key Point to Explain** | The system combines ML prediction with a safety override for hemorrhagic indicators (red spots/petechiae). |

---

### TC-I03 — Chronic: Diabetes (Layman Terms) ✅
| | |
|---|---|
| **Input** | "I am going to the toilet very often and always feeling extremely thirsty" |
| **Expected Top Prediction** | `Diabetes` |
| **Follow-up Questions** | *"Do you have any unexplained weight loss?"*, *"Are your wounds taking long to heal?"* |
| **Key Point to Explain** | Recognizes layman descriptions ("toilet very often") and maps them to chronic metabolic disorders. |

---

### TC-I04 — Rural Emergency: Snake Bite 🚨
| | |
|---|---|
| **Input** | "I was bitten by something in the farm, I see two puncture marks and my vision is blurring" |
| **Expected Behaviour** | 🚨 **Immediate Emergency Escalation.** |
| **Detected Category** | `Rural Emergency` |
| **Key Point to Explain** | Critical for low-resource settings; the system prioritizes rapid escalation for neurotoxic symptoms. |

---

### TC-I05 — Women's Health: PCOS ✅
| | |
|---|---|
| **Input** | "I am having irregular periods, severe acne, and unexplained weight gain" |
| **Expected Top Prediction** | `PCOS (Hormonal Imbalance)` |
| **Category** | `Metabolic / Women's Health` |
| **Key Point to Explain** | Broadens the triage scope to common but often ignored hormonal conditions in the region. |

---

### TC-I06 — Tropical: Heat Stroke 🚨
| | |
|---|---|
| **Input** | "My skin is very hot and dry, I fainted after working in the sun" |
| **Expected Behaviour** | 🚨 **Emergency Escalation.** |
| **Detected Category** | `Environmental Emergency` |
| **Key Point to Explain** | Essential for Indian summer (monsoon/pre-monsoon) conditions. |

---

### TC-I07 — Water-Borne: Amoebiasis (Stomach Infection) ✅
| | |
|---|---|
| **Input** | "I have severe stomach cramps and loose motions after eating street food" |
| **Expected Top Prediction** | `Amoebiasis / Gastroenteritis` |
| **Home Care** | Provided (Hydration, ORS, Rest) — since it is non-critical. |
| **Key Point to Explain** | High relevance for urban environments with street-food-heavy diets. |

---

### TC-I08 — Chronic: Vitamin D Deficiency ✅
| | |
|---|---|
| **Input** | "My bones and muscles ache all the time and I feel very lethargic" |
| **Expected Top Prediction** | `Vitamin D Deficiency` |
| **Key Point to Explain** | Addresses the highly prevalent deficiency in urban Indian populations. |

---


---

## 🏥 Feature 9 — Real-World Hospital Integration (Proximity & Capability)

> **What to demonstrate:** The system connects users to real medical facilities using the **National Hospital Directory** (10MB+ CSV). It uses vectorized geo-location to find the nearest hospital that matches the predicted disease's specialty requirements.

### TC-R01 — Proximity Search (Bangalore Baseline) ✅
| | |
|---|---|
| **Scenario** | User in Bangalore with a suspected heart condition. |
| **Input** | *"I have crushing chest pain and palpitations"* |
| **Hospital Output** | 1. **Bangalore Heart Hospital** |
| | 2. **Bangalore Baptist Hospital** |
| **Logic Shown** | The system prioritized hospitals with "Heart" or "Cardiology" specialties over general clinics. |

---

### TC-R02 — Specialty Filtering ✅
| | |
|---|---|
| **Scenario** | User with a suspected Kidney issue. |
| **Top Prediction** | `Kidney Failure` |
| **Hospital Output** | **Bengaluru Kidney Stone Hospital** |
| **Logic Shown** | The recommender scanned the `Specialties` column in the 10,000-row directory to find the specific "Kidney" match. |

---

### TC-R03 — Regional Coverage (Delhi Case) ✅
| | |
|---|---|
| **Scenario** | User in Delhi (Coordinates: 28.6139, 77.2090) |
| **Predicted Disease** | `Snake Bite` |
| **Hospital Output** | **AIIMS (Delhi)**, **Max Super Speciality** |
| **Logic Shown** | Demonstrates the nationwide scale of the integrated directory. |

---
