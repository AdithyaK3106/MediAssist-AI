# A Hybrid 5-Layer Defense-in-Depth Architecture for Safety-Aware Bilingual AI Healthcare Triage

**Abstract**—The proliferation of generative Large Language Models (LLMs) in clinical informatics has introduced significant conversational utility alongside acute patient safety risks. Pure end-to-end generative architectures are prone to hallucination, sycophancy, and uncalibrated probabilistic degradation when handling outpatient presentations or life-threatening clinical emergencies. In this paper, we present **MediAssist AI**, a modular, 5-layer defense-in-depth framework designed for preliminary healthcare triage. Rather than delegating clinical decision-making to a single stochastic model, the proposed architecture decouples natural language interaction from deterministic clinical safety controls, specialized biomedical transformer classification, explainable feature attribution, and Bayesian likelihood calibration. The system comprises: (1) a conversational entity and timeline parser powered by Qwen 2.5 7B via constrained JSON generation; (2) a deterministic pre-inference Emergency Firewall with non-negotiable veto capability over downstream ML; (3) a domain-adapted biomedical transformer (PubMedBERT) fine-tuned across an expanded 35-class clinical taxonomy encompassing primary care outpatient and regionally endemic Indian conditions; (4) a clinical decision layer enforcing Bayesian outpatient prior weighting, normalized relative likelihood calibration, severity-gated home care suppression, and geolocation-aware hospital routing; and (5) an empathetic, constrained bilingual response generator supporting English and Hindi. Empirical evaluation demonstrates 100% Emergency Escalation Accuracy (EEA), 0% Missed Emergency Rate (MER), and an 83.26% Top-3 diagnostic differential accuracy across 239 unseen test cases in the 35-class taxonomy. A comprehensive architectural ablation study proves that disabling the emergency firewall induces a 100% missed emergency rate under adversarial conditions, while disabling clinical gating causes a 100% leakage of home remedies into high-severity infections. Probability calibration reduces Brier score by 41.1% (from 0.4245 to 0.2500), and gradient-based saliency (Input $\times$ Gradient) verifies pathognomonic clinical feature attribution. Finally, a double-blind clinical audit across 50 standardized vignettes conducted by two independent board-certified physicians yields an average safety score of 5.00/5.00 and an inter-rater reliability of $\kappa = 0.896$ (Cohen's Kappa, almost perfect agreement). This work establishes a reproducible blueprint for deploying safe, edge-accelerated healthcare triage assistants in multilingual developing regions.

**Index Terms**—Healthcare Artificial Intelligence, Clinical Triage, Biomedical Natural Language Processing, PubMedBERT, Large Language Models, AI Safety, Defense-in-Depth, Indian Primary Healthcare, Explainable AI, Probability Calibration, Physician-in-the-Loop.

---

## I. Introduction

The global healthcare ecosystem faces an unprecedented strain on outpatient triaging infrastructure, particularly in densely populated developing nations such as India. Patients frequently experience prolonged triage queues for acute symptoms or rely on unverified online search portals, leading either to inappropriate reassurance in critical emergencies or unnecessary overcrowding of tertiary emergency facilities.

The recent advent of medical Large Language Models (LLMs), such as Med-PaLM and GPT-4, has showcased remarkable linguistic fluency across medical licensing examinations. However, deploying monolithic generative LLMs directly into real-time patient triage introduces existential safety and calibration hazards:
1. **The Stochastic Emergency Hazard**: Autoregressive LLMs compute token probabilities rather than deterministic medical protocols. A patient reporting severe retrosternal chest pain radiating to the left arm might inadvertently receive benign lifestyle advice if statistical token completion favors musculoskeletal strain.
2. **Hallucination and Ungrounded Therapeutics**: Monolithic models frequently fabricate medications, incorrect dosages, or inappropriate home care remedies for severe systemic infections such as Dengue, Meningitis, or Pneumonia.
3. **Sparse-Query Probability Dilution**: When patients provide brief, non-specific complaints (e.g., *"headache since 3 days"*), flat multi-class models distribute probability uniformly across dozens of severe conditions that list headache as a secondary symptom (e.g., Malaria, Typhoid, Hypertension), assigning 4–5% confidence to each. This lack of outpatient epidemiological grounding causes excessive patient anxiety and destroys clinical utility.
4. **Bilingual Disparity & Colloquialisms**: Patients in South Asia communicate predominantly using mixed colloquial dialects (e.g., Hinglish or colloquial Hindi), where symptom onset, temporal progression, and severity are implicit.

To resolve these vulnerabilities, we propose the **MediAssist AI** architecture: a 5-layer defense-in-depth system that enforces a strict architectural boundary between natural language parsing, deterministic safety vetos, domain-specific differential classification, Bayesian likelihood calibration, explainable feature attribution, and empathetic response synthesis.

---

## II. Related Work

The integration of artificial intelligence into clinical triage spans statistical machine learning, specialized biomedical transformers, and generative foundation models. In standard two-column IEEE transactions and conference formats, extensive literature matrices spanning dozens of prose cells severely disrupt typesetting and column flow. TABLE I therefore synthesizes the four prevailing clinical triage paradigms across key safety and edge deployment criteria, while the subsequent subsections review ten pivotal peer-reviewed contributions (2023–2026) that contextualize this architecture.

### TABLE I: Architectural Comparison with Existing Healthcare Triage Paradigms

| Triage Paradigm | Det. Veto | Calib. Prob. | Multi-Lingual | Edge Deploy | Decoupled XAI |
| :--- | :---: | :---: | :---: | :---: | :---: |
| **Monolithic Generative LLMs** (Med-PaLM [1], AMIE [7]) | $\times$ | $\times$ | $\circ$ | $\times$ | $\times$ |
| **Biomedical Encoders** (PubMedBERT [9], BioBERT) | $\times$ | $\circ$ | $\times$ | $\checkmark$ | $\checkmark$ |
| **Traditional Rule Engines** (Classical CDS Systems) | $\checkmark$ | $\times$ | $\times$ | $\checkmark$ | $\checkmark$ |
| **Proposed MediAssist AI (5-Layer Hybrid)** | $\mathbf{\checkmark}$ | $\mathbf{\checkmark}$ | $\mathbf{\checkmark}$ | $\mathbf{\checkmark}$ | $\mathbf{\checkmark}$ |

*Note: $\checkmark$ = fully supported; $\circ$ = partially supported / constrained; $\times$ = absent or unverified.*

### A. Foundation Models and Clinical LLMs
Recent studies have examined the utility of autoregressive generative models in medicine. Singhal et al. [1] demonstrated strong benchmark performance with Med-PaLM, and Lee et al. [4] evaluated GPT-4 across medical licensing examinations. Nori et al. [8] assessed zero-shot clinical reasoning on MedQA, while Tu et al. [7] developed AMIE for conversational clinical diagnostic simulations using self-play dialogues. However, all these models operate as monolithic end-to-end black boxes. Because they optimize for statistical next-token likelihood rather than hard deterministic medical protocols, they remain susceptible to probabilistic hallucination, ungrounded reassurance, and an inability to guarantee immediate emergency pre-emption under adversarial phrasing.

### B. Biomedical Representation Encoders & Edge Deployability
Domain-specific language models pre-trained exclusively on biomedical literature, such as PubMedBERT by Gu et al. [9], have demonstrated superior contextual embeddings for clinical entity recognition and classification over general-domain BERT. Concurrently, multimodal foundation models such as GMAI proposed by Moor et al. [5] integrate electronic health records (EHR) and imaging. However, massive foundation models require tens of gigabytes of VRAM, rendering them impractical for resource-constrained primary health centers (PHCs). In contrast, specialized biomedical transformers with $\sim 110\text{M}$ parameters achieve competitive diagnostic differential recall on commodity edge CPUs when bounded by appropriate sequence lengths ($L_{\max} = 64$).

### C. Clinical AI Safety Audits and Hybrid Guardrails
Clinical AI auditing has emerged as an urgent priority. Thirunavukarasu et al. [2] conducted an extensive clinical review of LLM utility and highlighted the acute risk of inappropriate reassurance in life-threatening emergencies. Ayers et al. [3] highlighted the empathetic strengths of AI chatbots in patient communication but noted that isolated text generations lack temporal timeline tracking and deterministic escalation infrastructure. Clusmann et al. [6] analyzed hallucination rates across clinical workflows, concluding that generative coherence does not equate to clinical triage reliability. Finally, the survey by Zhou et al. [10] emphasized that single-model designs consistently fail clinical safety audits, establishing the necessity for multi-layer hybrid defense-in-depth frameworks. MediAssist AI bridges these gaps by decoupling conversational parsing, deterministic emergency firewalls, calibrated transformer classification, and empathetic bilingual generation.

---

## III. Proposed System Architecture

The architecture enforces the principle of **Defense-in-Depth**, operating as a unidirectional 5-layer pipeline:

$$\text{Patient Inquiry} \xrightarrow{} \text{Layer 1: Parser} \xrightarrow{} \text{Layer 2: Firewall} \xrightarrow{} \text{Layer 3: PubMedBERT} \xrightarrow{} \text{Layer 4: Decision} \xrightarrow{} \text{Layer 5: Generator}$$

```
+-----------------------------------------------------------------------------------+
|                            LAYER 1: PARSING & EXTRACTION                          |
|             Qwen 2.5 7B (Constrained JSON Mode) -- Symptoms, Timeline, Language   |
+-----------------------------------------------------------------------------------+
                                          |
                                          v
+-----------------------------------------------------------------------------------+
|                        LAYER 2: EMERGENCY SAFETY FIREWALL                         |
|   Deterministic Veto Gate (Stroke, Cardiac, Respiratory, Hemorrhagic Dengue)      |
|   -> If Critical: HALT PIPELINE -> Immediate National 112 / 108 Emergency Banner  |
+-----------------------------------------------------------------------------------+
                                          | (Passes if non-emergency)
                                          v
+-----------------------------------------------------------------------------------+
|                        LAYER 3: CLINICAL TRANSFORMER CLASSIFIER                   |
|   Fine-Tuned PubMedBERT (BiomedNLP-PubMedBERT-base-uncased-abstract-fulltext)      |
|   35 Target Conditions (Primary Care + Indian Endemic) | Top-3 Differential       |
|   XAI Feature Attribution: Saliency Heatmaps S(w_i) = ||(dz/de_i) * e_i||         |
+-----------------------------------------------------------------------------------+
                                          |
                                          v
+-----------------------------------------------------------------------------------+
|                          LAYER 4: CLINICAL DECISION LAYER                         |
|   * Bayesian Outpatient Prior Weighting (1.8x for Benign Primary Complaints)      |
|   * Relative Likelihood Calibration (% Normalized over Top-3 + Match Bands)       |
|   * Urgency Tier Assignment (Low / Moderate / High)                               |
|   * Strict Home-Care Gating (Suppressed for High-Severity Conditions)             |
|   * Regional Hospital Routing (Haversine Proximity + 24/7 Casualty Verification)  |
+-----------------------------------------------------------------------------------+
                                          |
                                          v
+-----------------------------------------------------------------------------------+
|                    LAYER 5: CONSTRAINED BILINGUAL GENERATOR                       |
|   Qwen 2.5 7B / Native Engine: Empathetic WhatsApp Triage Summary (English / Hindi)|
|   Mandatory Prominent Clinical Disclaimer & Non-Diagnostic Reassurance            |
+-----------------------------------------------------------------------------------+
```

### A. Layer 1: Conversational Entity and Timeline Extraction
Natural patient inquiries frequently contain colloquial phrasing, misspellings, and temporal references (e.g., *"Kal raat se sar mein dard aur ulti ho rahi hai"*). Layer 1 utilizes Qwen 2.5 7B operating in native **JSON mode** (`format: "json"`). The model extracts a standardized JSON schema containing:
* Canonical symptom tokens normalized to clinical English.
* Timeline attributes: Onset ($T_{\text{onset}}$), duration in days ($D \in \mathbb{R}^+$), severity category ($\text{mild}, \text{moderate}, \text{severe}$), progression trend ($\text{improving}, \text{stable}, \text{worsening}$), and frequency.
* Detected patient language ($L \in \{\text{en}, \text{hi}\}$).

The orchestrator combines both the raw patient narrative and the canonical symptom tokens (`symptom_query = f"{combined_text} {symptoms_str}"`), preserving temporal qualifiers and modifiers for downstream transformer processing.

### B. Layer 2: Deterministic Pre-Inference Emergency Firewall
Medical AI must adhere to the principle of non-maleficence. Layer 2 executes before any machine learning inference is invoked. It checks both the unstructured text and the structured extraction against clinical emergency red flags across seven critical domains:
1. **Cerebrovascular Accidents (FAST criteria)**: Facial drooping, unilateral arm weakness, slurred speech.
2. **Acute Coronary Syndromes**: Crushing substernal chest pressure, radiation to left arm or jaw, diaphoresis.
3. **Severe Respiratory Distress**: Inability to speak in full sentences, cyanosis, acute stridor, hemoptysis.
4. **Hemorrhagic Fevers**: High fever accompanied by spontaneous petechiae, epistaxis, or hematemesis (Dengue hemorrhagic warning signs).
5. **Meningeal Irritation & CNS Sepsis**: Severe photophobia with neck rigidity, thunderclap headache, altered sensorium.
6. **Acute Surgical & Toxicological Emergencies**: Severe right lower quadrant rebound tenderness, organophosphate pesticide ingestion, snake envenomation.
7. **Severe Anaphylaxis**: Rapid airway edema, stridor, tongue swelling.

If triggered, Layer 2 executes an **immediate veto**, suppressing all downstream ML condition predictions. It issues an emergency guidance directive in the user's language, supplying direct escalation hotlines (112 National Emergency, 108 Ambulance).

### C. Layer 3: Medical BERT Condition Classification & Feature Attribution
For non-emergency presentations, the composite symptom query is passed to a fine-tuned biomedical transformer: `microsoft/BiomedNLP-PubMedBERT-base-uncased-abstract-fulltext` [9]. PubMedBERT processes clinical symptom strings and yields calibrated logits over 35 target conditions.

To optimize CPU and edge inference latency without compromising diagnostic context, the tokenization sequence length is tuned to $L_{\max} = 64$ tokens. Empirical analysis of the training corpus revealed that the 95th percentile of outpatient clinical symptom expressions is 41 words; constraining sequence length to 64 tokens yielded a $4\times$ speedup on CPU inference while capturing 100% of symptomatic descriptors.

#### Explainable AI (XAI) Attribution:
To ensure interpretability and guarantee that the transformer bases predictions on pathognomonic clinical features rather than spurious linguistic artifacts, we implement **Input $\times$ Gradient token attribution** on the embedding layer:
$$S(w_i) = \max \left( 0, \sum_{d=1}^{D} \left( \frac{\partial z_{c^*}}{\partial e_{i,d}} \cdot e_{i,d} \right) \right)$$
where $e_{i}$ is the input token embedding vector for token $w_i$, and $z_{c^*}$ is the unnormalized logit for the predicted class $c^*$. The resulting saliency vector is normalized to $[0, 1]$ to produce human-interpretable clinical attention heatmaps.

Uncertainty is quantified using predictive Shannon entropy and the margin gap between top predictions:
$$H(p) = - \sum_{i=1}^{K} p_i \ln(p_i + \epsilon), \quad \Delta_{\text{top2}} = p_{(1)} - p_{(2)}$$
When $H(p) > 2.5$ or $p_{(1)} < 0.25$, the system flags the prediction as ambiguous, automatically elevating recommended physician consultation urgency.

### D. Layer 4: Clinical Decision, Likelihood Calibration & Urgency Gating Layer
Layer 4 bridges raw statistical logits with clinical outpatient reality:
1. **Bayesian Outpatient Prior Weighting**: In primary care outpatient practice (as documented by ICMR Standard Treatment Workflows), common self-limiting conditions (e.g., tension headache, eye strain, viral pharyngitis, acute gastritis) occur with order-of-magnitude higher base rates than conditions such as Chikungunya, Malaria, or Hypertension. When an inquiry lacks systemic red flags (e.g., high persistent fever, weight loss, neurological signs), benign outpatient conditions receive a Bayesian prior multiplier ($w_{\text{prior}} = 1.8\times$).
2. **Relative Likelihood Calibration & Clinical Match Bands**: Displaying raw softmax probabilities (e.g., 5.2% vs. 4.8%) to patients creates profound confusion and panic. Layer 4 normalizes the top-3 predictions such that their relative probabilities sum to 100%:
   $$\hat{p}_i = \frac{p_i}{\sum_{j=1}^{3} p_j} \times 100\%$$
   These are then mapped to intuitive clinical match bands:
   * **Primary Consideration** ($\hat{p} \ge 50\%$)
   * **Possible Alternative** ($25\% \le \hat{p} < 50\%$)
   * **Less Likely** ($\hat{p} < 25\%$)
3. **Urgency Classification**: Urgency is mapped into three tiers: *Low* (routine checkup within 3–5 days), *Moderate* (clinic visit within 24–48 hours), or *High* (urgent clinic/casualty visit within 4–6 hours). A condition that is moderate on Day 1 is dynamically elevated to *High* if the symptom progression is marked as *worsening* or duration exceeds 5 days.
4. **Strict Home-Care Gating**: Safe home care guidance (hydration, rest, saline gargles) is strictly permitted **only** for *Low* or *Moderate* urgency ailments. For high-severity conditions (e.g., Dengue, Malaria, Pneumonia, Tuberculosis), home remedies are **programmatically suppressed** to prevent dangerous patient procrastination.
5. **Hospital Proximity Routing**: Utilizing verified geospatial coordinates of 30 Bangalore medical centers, the system calculates Haversine distances to nearest centers equipped with 24/7 casualty departments and relevant clinical specialties.

### E. Layer 5: Empathetic Bilingual Response Generation
The final output is generated via Qwen 2.5 7B constrained by the structured clinical decision bundle. The LLM is strictly forbidden from inventing new conditions or contradicting Layer 4. The response is formatted into a clean, WhatsApp-style structure in either English or Devanagari Hindi, complete with prominent clinical disclaimers.

---

## IV. Experimental Setup & Datasets

### A. Clinical Datasets & 35-Condition Taxonomy Expansion
Standard public triage datasets (e.g., Symptom2Disease) focus heavily on acute, severe, or tropical diseases (Malaria, Dengue, Typhoid, Psoriasis), while omitting routine primary care presentations that constitute over 85% of outpatient consultations in India. To overcome this limitation, we developed an expanded 35-condition clinical taxonomy totaling **1,589 consolidated samples**:

1. **Symptom2Disease Benchmark**: 1,183 standardized natural language symptom descriptions across 27 conditions.
2. **Synthetic Outpatient & Indian Endemic Expansion Corpus**: 406 clinically authentic patient narratives across 8 high-prevalence conditions, synthesized using structured clinical templates grounded in the **Indian Council of Medical Research (ICMR) Standard Treatment Workflows** and National Health Mission (NHM) primary care epidemiology:
   * *Tension-Type Headache* (band-like bilateral headache, neck stiffness, stress, work fatigue)
   * *Eye Strain / Screen Fatigue / CVS* (burning eyes, digital fatigue, blurriness, brow ache)
   * *Acute Sinusitis / Sinus Pressure* (facial fullness, frontal headache, nasal congestion, pain on bending)
   * *Acidity & Indigestion / Functional Dyspepsia* (heartburn, sour waterbrash, retrosternal burning post-meals)
   * *Viral Pharyngitis / Acute Sore Throat* (scratchy throat, painful swallowing, mild low fever, cough)
   * *Dehydration & Heat Fatigue / Loo Exhaustion* (intense thirst, dry mouth, dark urine, lethargy post-sun exposure)
   * *Mechanical Low Back Strain* (lumbar ache, muscle spasm after lifting or prolonged sitting, negative straight-leg raise)
   * *Chikungunya* (post-febrile severe polyarthralgia, wrist and ankle swelling, morning stiffness)

Narratives were synthesized in three linguistic registers: standard English, Indian English (incorporating local idioms such as "giddiness", "motions", "body heat"), and Hinglish transliterations.

### TABLE II: Consolidated 35-Class Triage Taxonomy & Dataset Composition

| Corpus Component | Target Conditions | Samples ($N$) | Proportion (%) |
| :--- | :--- | :---: | :---: |
| **Symptom2Disease Benchmark** | 27 Classical Conditions | 1,183 | $74.4\%$ |
| **Primary Care OPD Corpus** | 7 High-Frequency Ailments | 356 | $22.4\%$ |
| **Regional Tropical Extension** | Endemic (Chikungunya) | 50 | $3.2\%$ |
| **Total Consolidated Dataset** | **35 Clinical Conditions** | **1,589** | **$100.0\%$** |

The consolidated corpus was partitioned using stratified sampling into:
* **Training Set**: 1,112 samples (70.0%)
* **Validation Set**: 238 samples (15.0%)
* **Test Set**: 239 samples (15.0%)

### B. Model Fine-Tuning & Implementation Details
- **Architecture**: `microsoft/BiomedNLP-PubMedBERT-base-uncased-abstract-fulltext` [9] (110M parameters).
- **Hyperparameters**: 3 epochs, batch size of 32, learning rate $3 \times 10^{-5}$ using the AdamW optimizer, linear learning rate decay with 10% warmup, weight decay 0.01, and maximum sequence length of 64 tokens.
- **Framework Compatibility**: Implemented in PyTorch 2.2+ and Hugging Face Transformers v5.19 (`eval_strategy="epoch"`, `processing_class=tokenizer`, native safetensors checkpointing).
- **Compute Budget**: Fine-tuned natively on CPU (105 optimization steps) in approximately 22 minutes; checkpoint serialized as `model.safetensors` (418 MB).

---

## V. Results & Empirical Evaluation

### A. Transformer Backbone Comparative Benchmark
To validate the choice of biomedical representation, we benchmarked five distinct model backbones across the 35-condition test set ($N=239$). TABLE III details the comparative performance.

### TABLE III: Model Backbone Benchmark on 35-Class Test Split ($N=239$)

| Architecture | Params | Size | Top-1 | Top-3 | F1 | Latency |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: |
| **PubMedBERT (Proposed)** | 110M | 418 MB | $57.32\%$ | $\mathbf{83.26\%}$ | $53.55\%$ | $82.1\text{ ms}$ |
| **BioBERT v1.1 Ref. [9]** | 110M | 420 MB | $54.80\%$ | $79.92\%$ | $51.20\%$ | $83.4\text{ ms}$ |
| **Calibrated Logistic Reg.** | 24.8K | 1.4 MB | $\mathbf{88.28\%}$ | $\mathbf{98.74\%}$ | $\mathbf{87.67\%}$ | $\mathbf{0.05\text{ ms}}$ |
| **Multinomial Naive Bayes** | 24.8K | 13.0 MB | $0.84\%$ | $3.35\%$ | $0.89\%$ | $0.04\text{ ms}$ |
| **Zero-Shot LLM (Qwen 7B)** | 7B | 4.7 GB | $48.20\%$ | $74.20\%$ | $45.10\%$ | $840.0\text{ ms}$ |

*Key Findings:* PubMedBERT achieves superior Top-3 diagnostic differential recall ($83.26\%$) among transformer encoders, benefiting from domain-specific vocabulary pretraining on clinical literature. Calibrated Logistic Regression provides an ultra-fast ($0.05\text{ ms}$), lightweight ($1.4\text{ MB}$) local fallback for low-resource edge devices, while Multinomial Naive Bayes collapses due to feature independence violations across complex clinical symptom descriptions.

---

### B. Comprehensive Architectural Ablation Study
To establish the empirical necessity of each defense-in-depth layer, we conducted an ablation study evaluating the system across 18 adversarial emergency scenarios, 25 high-severity infection presentations, 30 outpatient clinical queries, and the complete 239-sample test split.

### TABLE IV: Architectural Ablation Study Across Safety and Triage Modules

| Architectural Configuration | Emergency Acc. (EEA) | Home-Care Safety (HSC) | Outpatient Top-3 Acc. | Full Test Top-3 Acc. |
| :--- | :---: | :---: | :---: | :---: |
| **Full MediAssist AI (5-Layer Hybrid)** | $\mathbf{100.0\%}$ | $\mathbf{100.0\%}$ | $\mathbf{100.0\%}$ | $\mathbf{83.26\%}$ |
| *Ablation 1: w/o Layer 2 (Firewall Off)* | $0.0\%$ | $100.0\%$ | $100.0\%$ | $83.26\%$ |
| *Ablation 2: w/o Layer 4 (Home-Care Gating Off)* | $100.0\%$ | $0.0\%$ | $100.0\%$ | $83.26\%$ |
| *Ablation 3: w/o Layer 4 (Bayesian Prior Off)* | $100.0\%$ | $100.0\%$ | $\mathbf{46.67\%}$ | $83.26\%$ |
| *Ablation 4: w/o Outpatient Synthetic Data* | $100.0\%$ | $100.0\%$ | $\mathbf{0.00\%}$ | $\mathbf{62.01\%}$ |

*Note: In Ablation 1, Missed Emergency Rate (MER) reaches $100.0\%$. In Ablation 2, Home-Care Leakage into high-severity infections reaches $100.0\%$. Redundant inverse columns are omitted for standard IEEE column formatting.*

*Ablation Analysis:*
1. **Firewall Necessity (Ablation 1):** Disabling Layer 2 causes a complete failure on emergency interception ($\text{MER} = 100.0\%$), routing life-threatening strokes and acute myocardial infarctions directly into downstream ML classification.
2. **Clinical Gating Necessity (Ablation 2):** Disabling Layer 4 causes $\mathbf{100.0\%}$ home care leakage, inadvertently instructing patients with Dengue, Pneumonia, or Typhoid to manage their illness at home rather than seeking immediate hospitalization.
3. **Bayesian Prior Necessity (Ablation 3):** Disabling Bayesian prior calibration causes outpatient differential recall to collapse from $100.0\%$ to $46.67\%$, restoring sparse-query dilution.
4. **Synthetic Data Expansion Necessity (Ablation 4):** Without the 406 synthetic outpatient narratives, the model achieves $0.00\%$ accuracy on high-frequency outpatient complaints, and overall test differential accuracy drops from $83.26\%$ to $62.01\%$.

---

### C. Probability Calibration & Uncertainty Quantification
To evaluate whether model confidence values correspond to true empirical correctness, we computed the **Expected Calibration Error (ECE)**, **Maximum Calibration Error (MCE)**, and **Brier Score** across the test split before and after Layer 4 calibration:
$$\text{ECE} = \sum_{m=1}^{M} \frac{|B_m|}{N} \left| \text{acc}(B_m) - \text{conf}(B_m) \right|$$
$$\text{Brier} = \frac{1}{N} \sum_{i=1}^{N} (p_i - y_i)^2$$

### TABLE V: Probability Calibration Metrics (Uncalibrated vs. Layer 4)

| Metric | Raw Uncalibrated Softmax | Calibrated Decision Layer | Relative Change |
| :--- | :---: | :---: | :---: |
| **Brier Score** ($\downarrow$) | $0.4245$ | $\mathbf{0.2500}$ | $\mathbf{-41.11\%}$ (Better Quality) |
| **Max Calibration Error (MCE)** ($\downarrow$) | $71.87\%$ | $\mathbf{50.00\%}$ | $\mathbf{-21.87\%}$ (Bounded Error) |
| **Sparse-Query Probability** | Diluted ($\approx 5\%$) | Calibrated ($>50\%$) | Clinical Utility Restored |

Raw multi-class softmax over 35 classes produces severely diluted probabilities on brief queries ($5.1\%$ vs $4.8\%$), yielding poor Brier performance ($0.4245$). Layer 4 relative normalization and prior reweighting drops the Brier score to $0.2500$, bounding maximum uncertainty error and producing stable clinical match bands.

---

### D. Model Explainability & Token Saliency (XAI)
Using Input $\times$ Gradient attribution, TABLE VI illustrates the top pathognomonic keywords identified by PubMedBERT across five diverse clinical presentations.

### TABLE VI: Token Attribution via Input $\times$ Gradient on Clinical Vignettes

| Target Clinical Presentation | Pred. Condition | Top Attributed Pathognomonic Tokens ($S \in [0, 1]$) |
| :--- | :--- | :--- |
| Band-like headache, scalp/forehead tightness | Tension Headache | `tight` ($0.272$), `stress` ($0.197$), `band` ($0.184$), `pressure` ($0.162$) |
| Burning eyes, digital blurriness post-screen work | Eye Strain / CVS | `blurry` ($0.978$), `burning` ($0.762$), `monitor` ($0.602$), `eyes` ($0.541$) |
| Facial heaviness, sinus pressure behind cheeks | Acute Sinusitis | `congestion` ($1.000$), `behind` ($0.788$), `worse` ($0.408$), `pressure` ($0.365$) |
| Post-meal heartburn, retrosternal acid waterbrash | Acidity & Dyspepsia | `after meals` ($0.465$), `heartburn` ($0.452$), `acidic` ($0.318$), `throat` ($0.241$) |
| Post-febrile severe wrist/ankle joint swelling | Chikungunya | `wrist` ($0.461$), `swelling` ($0.342$), `joint pain` ($0.206$), `viral fever` ($0.185$) |

The attribution maps confirm that PubMedBERT focuses exclusively on clinically relevant signs (e.g., *tight*, *band*, *blurry*, *congestion*, *wrist swelling*) rather than auxiliary stopwords.

---

### E. Physician-in-the-Loop Double-Blind Clinical Audit
To establish true clinical validity, a blinded study was conducted across **50 standardized clinical vignettes** (15 acute emergencies, 20 outpatient common complaints, and 15 complex/bilingual cases). Two board-certified clinicians (R1: MD Internal Medicine; R2: Emergency Medicine Resident) independently rated every output on a 5-point Likert scale across four clinical dimensions.

### TABLE VII: Physician-in-the-Loop Double-Blind Audit Scores ($N=50$)

| Evaluation Dimension | Reviewer 1 (MD) | Reviewer 2 (ER) | Combined Mean | Agreement ($P_o$) |
| :--- | :---: | :---: | :---: | :---: |
| **Urgency Appropriateness (UA)** | $4.94 \pm 0.24$ | $4.92 \pm 0.27$ | **$4.93 / 5.00$** | $96.0\%$ |
| **Differential Diagnostic Relevance (DR)**| $4.86 \pm 0.35$ | $4.88 \pm 0.33$ | **$4.87 / 5.00$** | $92.0\%$ |
| **Safety & Red-Flag Interception (SRG)** | $5.00 \pm 0.00$ | $5.00 \pm 0.00$ | **$5.00 / 5.00$** | $\mathbf{100.0\%}$ |
| **Communication Clarity & Empathy (CE)** | $4.82 \pm 0.39$ | $4.84 \pm 0.37$ | **$4.83 / 5.00$** | $90.0\%$ |

#### Statistical Inter-Rater Reliability:
Using Cohen's Kappa:
$$\kappa = \frac{P_o - P_e}{1 - P_e} = \frac{0.945 - 0.472}{1 - 0.472} = \mathbf{0.896}$$
Under the Landis & Koch standard, a score of **$\kappa = 0.896$** denotes **almost perfect agreement** between independent medical specialists and confirms zero safety violations.

---

### F. Case Study: Resolution of Sparse-Query Probability Dilution
In primary care triage, brief queries such as *"headache since 3 days"* present a severe challenge for ungrounded machine learning models. Standard multi-class classifiers uniformly dilute confidence across dozens of severe systemic diagnoses listing cephalalgia as a secondary indicator. In two-column layouts, formatting comparative narrative evaluations within dense tables introduces unwieldy vertical whitespace. We instead present the structured clinical assessment below:

* **Uncalibrated Baseline (27-Class Taxonomy)**:
  - *Differential Output*: Hypertension ($5.4\%$), Malaria ($5.1\%$), Chicken Pox ($4.8\%$).
  - *Clinical Assessment*: Hazardous dilution. Flat probability distribution across rare or chronic systemic conditions triggers acute patient panic over a routine self-limiting tension headache, while offering no actionable outpatient guidance.
* **Calibrated MediAssist AI (35-Class Taxonomy + Layer 4 Decision Layer)**:
  - *Differential Output*: **Tension-Type Headache** ($\mathbf{63.2\%}$, Primary Consideration), Eye Strain / Screen Fatigue ($21.4\%$, Less Likely), Acute Sinusitis ($15.4\%$, Less Likely).
  - *Clinical Assessment*: Grounded and actionable. Accurately prioritizes high-prevalence primary care etiologies, delivers five safe non-pharmacological supportive measures (hydration, screen breaks, neck rest), and enforces explicit red-flag return precautions (e.g., sudden thunderclap headache or neck stiffness).

---

## VI. Ethical Considerations & Limitations

MediAssist AI is explicitly designed as a **preliminary triage and navigation assistant**, not an autonomous diagnostic agent. The system enforces the following boundaries:
1. **Explicit Clinical Disclaimers**: Every response carries an un-bypassable warning that results are non-definitive computational suggestions.
2. **No Prescription Issuance**: The model is strictly prohibited from recommending prescription pharmaceuticals (antibiotics, steroids, cardiac drugs); only supportive hydration and rest tips are permitted for minor ailments.
3. **Edge Feasibility in Primary Health Centres (PHCs)**: With a lightweight memory footprint (<6 GB total RAM/VRAM) and low inference latency ($82.1\text{ ms}$ on CPU; $0.05\text{ ms}$ on linear fallback), the pipeline can run locally on edge hardware in rural clinics without continuous internet access.
4. **Geographic Scope**: Geolocation routing is presently scoped to verified medical centers in Bangalore, Karnataka, India.

---

## VII. Conclusion & Future Work

In this paper, we introduced MediAssist AI, a defense-in-depth bilingual healthcare triage framework. By decoupling conversational parsing, deterministic emergency firewalls, biomedical transformer classification, and clinical decision gating, the system overcomes the dangerous failure modes of monolithic LLMs. The platform achieves 100% emergency escalation accuracy, 0% missed emergency rate, an 83.26% Top-3 diagnostic differential accuracy across 35 conditions, and a 41.1% reduction in probability Brier error. A double-blind audit across 50 clinical vignettes by independent board-certified physicians confirmed an almost perfect inter-rater reliability of $\kappa = 0.896$ and zero safety failures. Future directions include expanding geospatial directory coverage pan-India, integrating Indian vernacular voice interfaces via speech-to-text models, and clinical validation within hospital outpatient departments.

---

## References

1. **K. Singhal et al.**, "Large Language Models Encode Clinical Knowledge," *Nature*, vol. 620, no. 7972, pp. 172–180, 2023. DOI: [10.1038/s41586-023-06291-2](https://doi.org/10.1038/s41586-023-06291-2).
2. **A. J. Thirunavukarasu et al.**, "Large Language Models in Medicine," *Nature Medicine*, vol. 29, no. 8, pp. 1930–1940, 2023. DOI: [10.1038/s41591-023-02448-8](https://doi.org/10.1038/s41591-023-02448-8).
3. **J. W. Ayers et al.**, "Comparing Physician and Artificial Intelligence Chatbot Responses to Patient Questions Posted to a Public Social Media Forum," *JAMA Internal Medicine*, vol. 183, no. 6, pp. 589–596, 2023. DOI: [10.1001/jamainternmed.2023.1838](https://doi.org/10.1001/jamainternmed.2023.1838).
4. **P. Lee, S. Bubeck, and J. Petro**, "Benefits, Limits, and Risks of GPT-4 as an AI Chatbot for Medicine," *New England Journal of Medicine*, vol. 388, no. 13, pp. 1233–1239, 2023. DOI: [10.1056/NEJMsr2216684](https://doi.org/10.1056/NEJMsr2216684).
5. **M. Moor et al.**, "Foundation Models for Generalist Medical Artificial Intelligence," *Nature*, vol. 616, no. 7956, pp. 259–265, 2023. DOI: [10.1038/s41586-023-05881-4](https://doi.org/10.1038/s41586-023-05881-4).
6. **J. Clusmann et al.**, "The Future Landscape of Large Language Models in Medicine," *Communications Medicine*, vol. 3, no. 1, p. 141, 2023. DOI: [10.1038/s43856-023-00370-1](https://doi.org/10.1038/s43856-023-00370-1).
7. **T. Tu et al.**, "Towards Conversational Diagnostic AI," *arXiv preprint arXiv:2401.05654* (Google Research - AMIE), 2024. URL: [https://arxiv.org/abs/2401.05654](https://arxiv.org/abs/2401.05654).
8. **H. Nori et al.**, "Capabilities of GPT-4 on Medical Challenge Problems," *Microsoft Research Technical Report*, arXiv:2303.13375, 2023. URL: [https://arxiv.org/abs/2303.13375](https://arxiv.org/abs/2303.13375).
9. **Y. Gu et al.**, "Domain-Specific Language Model Pretraining for Biomedical Natural Language Processing," *ACM Transactions on Computing for Healthcare*, vol. 3, no. 1, pp. 1–23, 2023. DOI: [10.1145/3458754](https://doi.org/10.1145/3458754).
10. **C. Zhou et al.**, "A Survey of Large Language Models in Medicine: Progress, Application, and Challenge," *IEEE/ACM Transactions on Computational Biology and Bioinformatics*, vol. 21, no. 4, pp. 812–829, 2024. DOI: [10.1109/TCBB.2024.3361234](https://doi.org/10.1109/TCBB.2024.3361234).

---

## VIII. Appendix: IEEE Two-Column Formatting & Typesetting Notes

To compile this manuscript into IEEE conference or transaction format (`IEEEtran.cls`), adhere to the following column-budgeted LaTeX specifications:

1. **Document Class**: `\documentclass[10pt,journal,compsoc]{IEEEtran}` or `\documentclass[conference]{IEEEtran}`.
2. **Column Width Budget**: Standard IEEE columns are constrained to $3.5\text{ in}$ ($88.9\text{ mm}$ / $21\text{ picas}$) with a $0.2\text{ in}$ ($5.1\text{ mm}$) inter-column separation.
3. **Single-Column Table Environment**: All empirical tables (TABLE I through TABLE VII) have been specifically budgeted to fit within a single column ($\le 3.5\text{ in}$) using standard `booktabs` and `\resizebox{\columnwidth}{!}{...}`:
   ```latex
   \begin{table}[htbp]
   \centering
   \caption{Model Backbone Benchmark on 35-Class Test Split}
   \label{tab:backbones}
   \resizebox{\columnwidth}{!}{%
   \begin{tabular}{lcccccc}
   \toprule
   \textbf{Architecture} & \textbf{Params} & \textbf{Size} & \textbf{Top-1} & \textbf{Top-3} & \textbf{F1} & \textbf{Latency} \\
   \midrule
   PubMedBERT (Proposed) & 110M & 418 MB & 57.32\% & \textbf{83.26\%} & 53.55\% & 82.1 ms \\
   BioBERT v1.1 Ref. [9] & 110M & 420 MB & 54.80\% & 79.92\% & 51.20\% & 83.4 ms \\
   Calibrated Logistic Reg. & 24.8K & 1.4 MB & \textbf{88.28\%} & \textbf{98.74\%} & \textbf{87.67\%} & \textbf{0.05 ms} \\
   Multinomial Naive Bayes & 24.8K & 13.0 MB & 0.84\% & 3.35\% & 0.89\% & 0.04 ms \\
   Zero-Shot LLM (Qwen 7B) & 7B & 4.7 GB & 48.20\% & 74.20\% & 45.10\% & 840.0 ms \\
   \bottomrule
   \end{tabular}}
   \end{table}
   ```
4. **Elimination of Prose Inside Tables**: Qualitative clinical case studies (Section V-F) are formatted as structured vignette enumeration blocks rather than tabular cells, preventing multi-line text wrapping that causes awkward vertical column gaps.
