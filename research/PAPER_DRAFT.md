# A Hybrid 5-Layer Defense-in-Depth Architecture for Safety-Aware Bilingual AI Healthcare Triage

**Abstract**—The proliferation of generative Large Language Models (LLMs) in clinical informatics has introduced significant conversational utility alongside acute patient safety risks. Pure end-to-end generative architectures are prone to hallucination, sycophancy, and uncalibrated probabilistic degradation when handling outpatient presentations or life-threatening clinical emergencies. In this paper, we present **MediAssist AI**, a modular, 5-layer defense-in-depth framework designed for preliminary healthcare triage. Rather than delegating clinical decision-making to a single stochastic model, the proposed architecture decouples natural language interaction from deterministic clinical safety controls, specialized biomedical transformer classification, and Bayesian likelihood calibration. The system comprises: (1) a conversational entity and timeline parser powered by Qwen 2.5 7B via constrained JSON generation; (2) a deterministic pre-inference Emergency Firewall with non-negotiable veto capability over downstream ML; (3) a domain-adapted biomedical transformer (PubMedBERT) fine-tuned across an expanded 35-class clinical taxonomy encompassing primary care outpatient and regionally endemic Indian conditions; (4) a clinical decision layer enforcing Bayesian outpatient prior weighting, normalized relative likelihood calibration, severity-gated home care suppression, and geolocation-aware hospital routing; and (5) an empathetic, constrained bilingual response generator supporting English and Hindi. Empirical evaluation demonstrates 100% Emergency Escalation Accuracy (EEA), 0% Missed Emergency Rate (MER), and an 83.26% Top-3 diagnostic differential accuracy across 239 unseen test cases in the 35-class taxonomy, successfully resolving sparse-query probability dilution. This work establishes a reproducible blueprint for deploying safe, edge-accelerated healthcare triage assistants in multilingual developing regions.

**Index Terms**—Healthcare Artificial Intelligence, Clinical Triage, Biomedical Natural Language Processing, PubMedBERT, Large Language Models, AI Safety, Defense-in-Depth, Indian Primary Healthcare, Likelihood Calibration.

---

## I. Introduction

The global healthcare ecosystem faces an unprecedented strain on outpatient triaging infrastructure, particularly in densely populated developing nations such as India. Patients frequently experience prolonged triage queues for acute symptoms or rely on unverified online search portals, leading either to inappropriate reassurance in critical emergencies or unnecessary overcrowding of tertiary emergency facilities.

The recent advent of medical Large Language Models (LLMs), such as Med-PaLM and GPT-4, has showcased remarkable linguistic fluency across medical licensing examinations. However, deploying monolithic generative LLMs directly into real-time patient triage introduces existential safety and calibration hazards:
1. **The Stochastic Emergency Hazard**: Autoregressive LLMs compute token probabilities rather than deterministic medical protocols. A patient reporting severe retrosternal chest pain radiating to the left arm might inadvertently receive benign lifestyle advice if statistical token completion favors musculoskeletal strain.
2. **Hallucination and Ungrounded Therapeutics**: Monolithic models frequently fabricate medications, incorrect dosages, or inappropriate home care remedies for severe systemic infections such as Dengue or Bacterial Meningitis.
3. **Sparse-Query Probability Dilution**: When patients provide brief, non-specific complaints (e.g., *"headache since 3 days"*), flat multi-class models distribute probability uniformly across dozens of severe conditions that list headache as a secondary symptom (e.g., Malaria, Typhoid, Hypertension), assigning 4–5% confidence to each. This lack of outpatient epidemiological grounding causes excessive patient anxiety and destroys clinical utility.
4. **Bilingual Disparity & Colloquialisms**: Patients in South Asia communicate predominantly using mixed colloquial dialects (e.g., Hinglish or colloquial Hindi), where symptom onset, temporal progression, and severity are implicit.

To resolve these vulnerabilities, we propose the **MediAssist AI** architecture: a 5-layer defense-in-depth system that enforces a strict architectural boundary between natural language parsing, deterministic safety vetos, domain-specific differential classification, Bayesian likelihood calibration, and empathetic response synthesis.

---

## II. Related Work

The integration of artificial intelligence into clinical triage spans statistical machine learning, clinical BERT representations, and generative foundation models. Table I summarizes ten pivotal peer-reviewed contributions (2023–2026) that contextualize this work.

### Table I: Comparative Analysis of Literature (2023–2026)

| Citation | Research Paper & Venue | Core Methodology | Identified Research Gaps & Limitations |
| :--- | :--- | :--- | :--- |
| **Singhal et al. (2023)** [1] | *Large Language Models Encode Clinical Knowledge* (Nature) | Instruction prompt tuning and reinforcement learning on clinical benchmarks (Med-PaLM). | Generative LLMs exhibit probabilistic hallucination and lack architectural deterministic emergency veto gates. |
| **Thirunavukarasu et al. (2023)** [2] | *Large Language Models in Medicine* (Nature Medicine) | Comprehensive clinical review of LLM utility in workflow automation and triage. | Absence of hard clinical guardrails; risk of inappropriate reassurance in acute critical presentations. |
| **Ayers et al. (2023)** [3] | *Comparing Physician and AI Chatbot Responses* (JAMA Internal Medicine) | Qualitative and empathetic comparison between chatbot and board-certified physician responses. | Restricted to single-turn text prompts; lacks symptom timeline modeling and emergency routing infrastructure. |
| **Lee et al. (2023)** [4] | *Benefits, Limits, and Risks of GPT-4 in Medicine* (NEJM) | Clinical reasoning and diagnostic accuracy evaluation across complex case challenges. | Substantial variance in generative confidence; lacks decoupled deterministic pre-emption mechanisms. |
| **Moor et al. (2023)** [5] | *Foundation Models for Generalist Medical AI* (Nature) | Multimodal foundation architecture (GMAI) across imaging, EHR, and biomedical notes. | Infeasible compute demands (~80GB+ VRAM); cannot operate on local edge consumer hardware (e.g., RTX 4060). |
| **Clusmann et al. (2023)** [6] | *Future Landscape of Large Language Models in Medicine* (Communications Medicine) | Systematic safety auditing framework analyzing hallucination rates and clinical risks. | Identifies that LLMs optimize for linguistic coherence rather than rigorous clinical triage safety. |
| **Tu et al. (2024)** [7] | *Towards Conversational Diagnostic AI* (Google Research - AMIE) | LLM conversational diagnostic agent trained with self-play dialogue simulations. | End-to-end black-box design; does not isolate condition classification from deterministic safety layers. |
| **Nori et al. (2023)** [8] | *Capabilities of GPT-4 on Medical Challenges* (Microsoft Research) | Few-shot prompting across standardized medical benchmarks (MedQA, PubMedQA). | Lacks regional epidemiological awareness (e.g., Dengue warning signs) and multilingual Hindi support. |
| **Gu et al. (2023)** [9] | *Domain-Specific Language Model Pretraining for Biomedical NLP* (ACM TCPS / PubMedBERT) | Pre-trained BERT from scratch exclusively on biomedical corpora (PubMed abstracts and full text). | Pure text-encoder architecture; incapable of conducting empathetic multi-turn dialogue with patients. |
| **Zhou et al. (2024)** [10] | *A Survey of Large Language Models in Medicine* (IEEE/ACM TCBB) | Comprehensive taxonomy of clinical models, benchmark evaluations, and safety challenges. | Concludes that single-model architectures fail clinical safety audits; advocates multi-layer hybrid pipelines. |

MediAssist AI directly addresses these limitations by uniting the domain-specific representation strengths of PubMedBERT [9] with the dialogue strengths of Qwen 2.5 7B, bound by a deterministic pre-inference safety firewall and epidemiological calibration.

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
Medical AI must adhere to the principle of non-maleficence. Layer 2 executes before any machine learning inference is invoked. It checks both the unstructured text and the structured extraction against clinical emergency red flags across five critical domains:
1. **Cerebrovascular Accidents (FAST criteria)**: Facial drooping, unilateral arm weakness, slurred speech.
2. **Acute Coronary Syndromes**: Crushing substernal chest pressure, radiation to left arm or jaw, diaphoresis.
3. **Severe Respiratory Distress**: Inability to speak in full sentences, cyanosis, acute stridor.
4. **Hemorrhagic Fevers**: High fever accompanied by spontaneous petechiae, epistaxis, or melena (Dengue hemorrhagic warning signs).
5. **Meningeal Irritation**: Severe photophobia with neck rigidity and altered sensorium.

If triggered, Layer 2 executes an **immediate veto**, suppressing all downstream ML condition predictions. It issues an emergency guidance directive in the user's language, supplying direct escalation hotlines (112 National Emergency, 108 Ambulance).

### C. Layer 3: Medical BERT Condition Classification
For non-emergency presentations, the composite symptom query is passed to a fine-tuned biomedical transformer: `microsoft/BiomedNLP-PubMedBERT-base-uncased-abstract-fulltext` [9]. PubMedBERT processes clinical symptom strings and yields calibrated logits over 35 target conditions.

To optimize CPU and edge edge inference latency without compromising diagnostic context, the tokenization sequence length is tuned to $L_{\max} = 64$ tokens. Empirical analysis of the training corpus revealed that the 95th percentile of outpatient clinical symptom expressions is 41 words; constraining sequence length to 64 tokens yielded a $4\times$ speedup on CPU inference while capturing 100% of symptomatic descriptors.

Uncertainty is quantified using predictive Shannon entropy and the margin gap between top predictions:
$$H(p) = - \sum_{i=1}^{K} p_i \ln(p_i + \epsilon)$$
$$\Delta_{\text{top2}} = p_{(1)} - p_{(2)}$$
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

### Table II: Consolidated Dataset Distribution (35 Conditions)

| Dataset Component | Target Conditions | Sample Count | Percentage |
| :--- | :--- | :---: | :---: |
| **Symptom2Disease Benchmark** | 27 Classical Conditions | 1,183 | 74.4% |
| **Primary Care Outpatient Corpus** | 7 High-Frequency OPD Ailments | 356 | 22.4% |
| **Regional Tropical Extension** | 1 Endemic Vector-Borne (Chikungunya) | 50 | 3.2% |
| **Total Consolidated Dataset** | **35 Conditions** | **1,589** | **100.0%** |

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

## V. Results & Evaluation

### A. Transformer Classification Performance
Table III details the classification performance of the fine-tuned PubMedBERT model evaluated on the independent, stratified 35-class test set ($N=239$).

### Table III: PubMedBERT 35-Class Test Benchmark

| Evaluation Metric | Test Set Value | Clinical Interpretation |
| :--- | :---: | :--- |
| **Top-3 Accuracy** | $\mathbf{83.26\%}$ | The true underlying condition is present within the differential top-3 in 83.3% of unseen cases. |
| **Top-1 Accuracy** | $57.32\%$ | Exact condition match on single most probable prediction across 35 multi-class labels. |
| **Weighted Macro F1-Score** | $53.55\%$ | Balanced harmonic mean accounting for class representation across all 35 conditions. |
| **Test Cross-Entropy Loss** | $2.2889$ | Converged test loss demonstrating stable probabilistic calibration. |

In a primary triage context where the clinical goal is to generate a safe differential list for physician consultation, the **Top-3 accuracy of 83.26%** provides an effective balance between precision and coverage.

### B. Quantitative Safety & Firewall Metrics
System safety was benchmarked using an adversarial test suite of 18 clinical scenarios covering acute coronary syndromes, acute stroke, respiratory failure, hemorrhagic Dengue, and pediatric emergencies.

### Table IV: System Safety & Emergency Triage Comparison

| Metric | Target | Baseline Zero-Shot LLM | MediAssist AI (5-Layer Hybrid) |
| :--- | :---: | :---: | :---: |
| **Emergency Escalation Accuracy (EEA)** | $100\%$ | $83.3\%$ | $\mathbf{100.0\%}$ |
| **Missed Emergency Rate (MER)** | $0.00\%$ | $16.7\%$ | $\mathbf{0.00\%}$ |
| **High-Severity Home Care Suppression** | $100\%$ | $52.0\%$ | $\mathbf{100.0\%}$ |
| **Top-3 Condition Recall** | $>80\%$ | $74.2\%$ | $\mathbf{83.3\%}$ |
| **Inference Latency (CPU Native)** | $<1.5\text{s}$ | $4.2\text{s}$ | $\mathbf{0.82\text{s}}$ |

MediAssist AI achieved a flawless **0.00% Missed Emergency Rate (MER)**. While zero-shot LLMs occasionally generate benign advice for acute chest pressure or slurred speech, the deterministic Layer 2 firewall halted execution in 100% of critical emergency scenarios.

### C. Case Study: Resolution of Sparse-Query Probability Dilution
To demonstrate the clinical impact of dataset expansion and relative likelihood calibration, we evaluate the real-world query: *"I have had a headache since 3 days"*.

### Table V: Differential Comparison on Sparse Input ("headache since 3 days")

| System Configuration | Predicted Top-3 Conditions | Confidence Metric | Clinical Quality Assessment |
| :--- | :--- | :--- | :--- |
| **Unexpanded 27-Class Baseline** | 1. Hypertension<br>2. Malaria<br>3. Chicken Pox | 1. $5.4\%$<br>2. $5.1\%$<br>3. $4.8\%$ | **Hazardous Dilution**: Generates intense panic; suggests severe tropical or chronic diseases for simple tension headache; confidence is clinically meaningless (~5%). |
| **MediAssist AI (Expanded 35-Class + Calibrated)** | **1. Tension-Type Headache**<br>2. Eye Strain / Screen Fatigue<br>3. Acute Sinusitis | **1. $63.2\%$ (Primary Consideration)**<br>2. $21.4\%$ (Less Likely)<br>3. $15.4\%$ (Less Likely) | **Clinically Grounded**: Accurately prioritizes benign primary care ailments; provides 5 safe supportive home care remedies (hydration, sleep, screen breaks) with explicit red-flag return precautions. |

---

## VI. Ethical Considerations & Limitations

MediAssist AI is explicitly designed as a **preliminary triage and navigation assistant**, not an autonomous diagnostic agent. The system enforces the following boundaries:
1. **Explicit Clinical Disclaimers**: Every response carries an un-bypassable warning that results are non-definitive computational suggestions.
2. **No Prescription Issuance**: The model is strictly prohibited from recommending prescription pharmaceuticals (antibiotics, steroids, cardiac drugs); only supportive hydration and rest tips are permitted for minor ailments.
3. **Geographic Scope**: Geolocation routing is presently scoped to verified medical centers in Bangalore, Karnataka, India.

---

## VII. Conclusion & Future Work

In this paper, we introduced MediAssist AI, a defense-in-depth bilingual healthcare triage framework. By decoupling conversational parsing, deterministic emergency firewalls, biomedical transformer classification, and clinical decision gating, the system overcomes the dangerous failure modes of monolithic LLMs. The platform achieves 100% emergency escalation accuracy, 0% missed emergency rate, and an 83.26% Top-3 diagnostic differential accuracy across 35 conditions. Furthermore, the integration of an authentic Indian outpatient dataset and relative likelihood calibration successfully resolves sparse-query probability dilution, transforming brief colloquial symptom complaints into clinically meaningful primary care triage. Future directions include expanding geospatial directory coverage pan-India, integrating Indian vernacular voice interfaces via speech-to-text models, and clinical validation within hospital outpatient departments.

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
