import streamlit as st
import sys
import json
from pathlib import Path

# ── PATH SETUP ─────────────────────────────────────────────────────────────
project_root = Path(__file__).resolve().parents[2]
if str(project_root) not in sys.path:
    sys.path.append(str(project_root))

from src.conversational.conversation_manager import ConversationManager
from src.models.classical_ml.predict import ClassicalInferencePipeline
from src.recommendation.home_care import HomeCareRecommender
from src.recommendation.recommender import HospitalRecommender
from src.frontend.utils.confidence_formatter import normalize_topk_probabilities, get_confidence_label, get_confidence_color

st.set_page_config(page_title="MediAssist AI", layout="wide", page_icon="🏥", initial_sidebar_state="expanded")

# ── LOAD DISEASE NAMES ──────────────────────────────────────────────────────
try:
    with open(project_root / "data" / "disease_display_names.json", "r", encoding="utf-8") as f:
        DISEASE_DISPLAY_NAMES = json.load(f)
except Exception:
    DISEASE_DISPLAY_NAMES = {}

# ── INJECT CSS ──────────────────────────────────────────────────────────────
styles_dir = project_root / "src/frontend/styles"
try:
    css_content = ""
    for css_file in ["main.css", "theme.css", "glassmorphism.css", "animations.css", "components.css", "responsive.css"]:
        p = styles_dir / css_file
        if p.exists():
            css_content += p.read_text(encoding="utf-8") + "\n"
    st.markdown(f"<style>{css_content}</style>", unsafe_allow_html=True)
except Exception as e:
    st.warning(f"Could not load custom styles: {e}")

# ── BACKGROUND ORBS ─────────────────────────────────────────────────────────
st.markdown("""
<div style="position:fixed;top:-120px;right:-80px;width:420px;height:420px;background:radial-gradient(circle,rgba(139,92,246,0.22) 0%,transparent 70%);border-radius:50%;pointer-events:none;z-index:0;animation:floatBob 7s ease-in-out infinite;"></div>
<div style="position:fixed;bottom:-100px;left:-60px;width:380px;height:380px;background:radial-gradient(circle,rgba(59,130,246,0.18) 0%,transparent 70%);border-radius:50%;pointer-events:none;z-index:0;animation:floatBob 9s ease-in-out infinite reverse;"></div>
<div style="position:fixed;top:40%;left:35%;width:260px;height:260px;background:radial-gradient(circle,rgba(96,165,250,0.10) 0%,transparent 70%);border-radius:50%;pointer-events:none;z-index:0;animation:floatBob 11s ease-in-out infinite 2s;"></div>
""", unsafe_allow_html=True)

# ── SESSION STATE ───────────────────────────────────────────────────────────
if "pipeline" not in st.session_state:
    try:
        st.session_state.pipeline = ClassicalInferencePipeline(model_name="logistic_regression")
    except Exception as e:
        st.error(f"Failed to load prediction pipeline: {e}")
        st.session_state.pipeline = None

if "manager" not in st.session_state:
    st.session_state.manager = ConversationManager()

if "home_care_recommender" not in st.session_state:
    st.session_state.home_care_recommender = HomeCareRecommender()

if "hospital_recommender" not in st.session_state:
    st.session_state.hospital_recommender = HospitalRecommender()

if "history" not in st.session_state:
    st.session_state.history = []

# ── COMPACT HEADER ──────────────────────────────────────────────────────────
st.markdown("""
<div class="glass-panel animate-fade-in" style="padding:10px 24px;display:flex;justify-content:space-between;align-items:center;margin-bottom:12px;">
<div style="display:flex;align-items:center;gap:10px;">
<div style="background:linear-gradient(135deg,#3B82F6,#2563EB);width:34px;height:34px;border-radius:9px;display:flex;align-items:center;justify-content:center;font-size:1rem;color:white;box-shadow:0 4px 10px rgba(37,99,235,0.3);">&#127973;</div>
<div><h2 style="margin:0;font-size:1.05rem;color:#1e293b;line-height:1.2;">MediAssist AI</h2><div style="font-size:0.72rem;color:#64748b;">Safety-Aware Healthcare Platform</div></div>
</div>
<div style="display:flex;gap:8px;">
<span class="badge" style="background:#dcfce7;color:#166534;">&#128994; System Operational</span>
<span class="badge" style="background:#f1f5f9;color:#475569;">v2.1.0</span>
</div>
</div>
""", unsafe_allow_html=True)

# ── 3-COLUMN DASHBOARD LAYOUT ───────────────────────────────────────────────
col1, col2, col3 = st.columns([1.05, 0.95, 0.95])

# Shared state
manager = st.session_state.manager
latest_predictions = getattr(manager, 'latest_predictions', [])
latest_flag_unreliable = getattr(manager, 'latest_flag_unreliable', False)
latest_consistency = getattr(manager, 'latest_consistency_data', None)
is_emergency = (latest_predictions == [] and latest_flag_unreliable)

# Pre-compute prediction data if available
normalized_preds = friendly_name = description = severity = is_critical = None
conf_label = conf_color = top_reasoning = home_care_data = None

if latest_predictions:
    normalized_preds = normalize_topk_probabilities(latest_predictions)
    top_disease = normalized_preds[0]['disease']
    top_rel_prob = normalized_preds[0]['relative_prob']
    friendly_data = DISEASE_DISPLAY_NAMES.get(top_disease, {"friendly_name": top_disease, "description": "No description available."})
    friendly_name = friendly_data['friendly_name']
    description = friendly_data['description']
    recommender = st.session_state.home_care_recommender
    home_care_data = recommender.get_home_care_data(top_disease)
    severity = home_care_data.get("severity", "medium")
    is_critical = severity == "critical"
    if latest_consistency:
        top_reasoning = next((r for r in latest_consistency["reasoning"] if r["disease"] == top_disease), None)
    conf_label = get_confidence_label(top_rel_prob)
    conf_color = get_confidence_color(top_rel_prob, severity)

# ════════════════════════════════════════════════════════════
# COL 1 — CHAT PANEL
# ════════════════════════════════════════════════════════════
with col1:
    st.markdown('<div class="glass-panel animate-slide-in" style="padding:16px !important;">', unsafe_allow_html=True)
    st.markdown('<h3 style="display:flex;align-items:center;gap:8px;margin-bottom:10px;font-size:1rem;">&#128172; Patient Assessment</h3>', unsafe_allow_html=True)

    if not st.session_state.history:
        st.markdown("""
<div style="text-align:center;padding:20px 10px;">
<div style="font-size:2.6rem;margin-bottom:8px;">&#128075;</div>
<h3 style="margin-bottom:6px;color:#1e293b;font-size:1rem;">Welcome to MediAssist</h3>
<p style="color:#64748b;font-size:0.82rem;margin-bottom:14px;">Describe your symptoms to begin.</p>
<div style="display:flex;flex-wrap:wrap;justify-content:center;gap:6px;">
<div class="symptom-chip">Fever and chills</div>
<div class="symptom-chip">Severe headache</div>
<div class="symptom-chip">Persistent cough</div>
<div class="symptom-chip">Stomach pain</div>
</div>
</div>
        """, unsafe_allow_html=True)
    else:
        with st.container(height=380, border=False):
            for msg in st.session_state.history:
                with st.chat_message(msg["role"]):
                    st.markdown(msg["content"])

    st.markdown('</div>', unsafe_allow_html=True)

    if prompt := st.chat_input("Enter your symptoms"):
        st.session_state.history.append({"role": "user", "content": prompt})
        if st.session_state.pipeline:
            response = manager.handle_message(prompt, prediction_pipeline=st.session_state.pipeline)
        else:
            response = "Sorry, the prediction pipeline is not loaded."
        st.session_state.history.append({"role": "assistant", "content": response})
        st.rerun()

# ════════════════════════════════════════════════════════════
# COL 2 — ASSESSMENT RESULTS + ALERT
# ════════════════════════════════════════════════════════════
with col2:
    if is_emergency:
        st.markdown("""
<div class="glass-panel animate-pulse-critical" style="border:2px solid #b91c1c;background:rgba(254,226,226,0.85);padding:20px !important;">
<h3 style="color:#b91c1c;display:flex;align-items:center;gap:8px;font-size:1rem;">&#128680; CRITICAL ALERT</h3>
<p style="color:#7f1d1d;font-size:0.88rem;line-height:1.5;margin-bottom:10px;">Your symptoms indicate a <strong>potential medical emergency</strong>. Seek immediate care.</p>
<ul class="custom-list" style="color:#991b1b;font-weight:600;font-size:0.85rem;">
<li>Call your local emergency number immediately.</li>
<li>Go to the nearest hospital emergency room.</li>
</ul>
<div style="margin-top:12px;text-align:center;"><span style="color:#b91c1c;font-size:0.75rem;font-weight:700;text-transform:uppercase;letter-spacing:0.06em;">&#128737; Emergency Override Active</span></div>
</div>
        """, unsafe_allow_html=True)

    elif latest_predictions:
        # Assessment card
        alert_class = f"glass-alert-{severity}"
        if is_critical:
            alert_class += " animate-pulse-critical"

        st.markdown(f"""
<div class="glass-panel hover-lift animate-slide-in" style="padding:18px !important;">
<h3 style="display:flex;align-items:center;gap:8px;font-size:0.95rem;margin-bottom:8px;">&#129658; Assessment Results</h3>
<h2 style="color:{conf_color};margin:6px 0 10px;font-size:1.25rem;line-height:1.2;">{friendly_name}</h2>
<div style="display:flex;gap:8px;flex-wrap:wrap;margin-bottom:10px;">
<span class="badge" style="background-color:{conf_color};color:white;">Severity: {severity.upper()}</span>
<span class="badge badge-outline" style="color:{conf_color};">Match: {conf_label}</span>
</div>
<div class="confidence-bar-bg">
<div class="progress-bar-fill" style="width:{top_rel_prob}%;background-color:{conf_color};height:100%;"></div>
</div>
<p style="color:#4B5563;font-size:0.85rem;line-height:1.5;margin:0;">{description}</p>
</div>
        """, unsafe_allow_html=True)

        # Alert / next-steps card
        card2 = f'<div class="glass-panel {alert_class} animate-slide-in" style="padding:16px !important;">'
        if is_critical:
            card2 += '<h3 style="color:#b91c1c;display:flex;align-items:center;gap:8px;font-size:0.95rem;">&#128680; Critical Alert</h3>'
            card2 += '<p style="color:#991b1b;font-size:0.85rem;font-weight:500;">Seek emergency care immediately.</p>'
        elif latest_flag_unreliable or severity == "high" or (top_reasoning and top_reasoning["consistency"] == "LOW"):
            card2 += '<h3 style="color:#c2410c;display:flex;align-items:center;gap:8px;font-size:0.95rem;">&#9888;&#65039; Medical Consultation Advised</h3>'
            if top_reasoning and top_reasoning["consistency"] == "LOW" and latest_flag_unreliable:
                card2 += '<p style="color:#c2410c;font-size:0.82rem;">Symptoms overlap multiple conditions — professional diagnosis needed.</p>'
            card2 += '<ul class="custom-list" style="color:#9a3412;font-size:0.85rem;"><li>Monitor symptoms closely</li><li>Seek attention if symptoms worsen</li></ul>'
        else:
            card2 += '<h3 style="color:#15803d;display:flex;align-items:center;gap:8px;font-size:0.95rem;">&#128204; Recommended Next Steps</h3>'
            card2 += '<ul class="custom-list" style="color:#166534;font-size:0.85rem;"><li>Monitor symptoms 24–48 hours</li><li>Stay hydrated and rest well</li><li>Follow the home care tips</li></ul>'
        card2 += '</div>'
        st.markdown(card2, unsafe_allow_html=True)

        # Expanders (fit naturally below the two cards)
        with st.expander("&#128269; Alternative Possibilities"):
            alt_html = "<ul class='custom-list'>"
            for pred in normalized_preds[1:]:
                ad = pred['disease']
                af = DISEASE_DISPLAY_NAMES.get(ad, {"friendly_name": ad})['friendly_name']
                al = get_confidence_label(pred['relative_prob'])
                alt_html += f"<li style='font-size:0.85rem;'><strong style='color:#334155;'>{af}</strong><br><span style='color:#94a3b8;font-size:0.8rem;'>Match: {al}</span></li>"
            alt_html += "</ul>"
            st.markdown(alt_html, unsafe_allow_html=True)

        if latest_consistency and latest_consistency.get("detected_symptoms"):
            with st.expander("&#129300; Why were these suggested?"):
                syms = ", ".join([s.capitalize() for s in latest_consistency["detected_symptoms"]])
                rh = f"<p style='color:#475569;font-size:0.85rem;'><strong>Symptoms detected:</strong> {syms}</p>"
                if top_reasoning:
                    cats = ", ".join([c.capitalize() for c in top_reasoning["disease_categories"]])
                    lvl = top_reasoning["consistency"]
                    if cats:
                        rh += f"<p style='color:#64748b;font-size:0.82rem;'><em>{friendly_name} typically presents with: {cats} symptoms.</em></p>"
                    if lvl == "HIGH":
                        rh += "<p style='color:#16a34a;font-size:0.82rem;font-weight:500;'>&#10003; High medical consistency</p>"
                    elif lvl == "LOW":
                        rh += "<p style='color:#ea580c;font-size:0.82rem;font-weight:500;'>&#9888;&#65039; Low medical consistency</p>"
                st.markdown(rh, unsafe_allow_html=True)

    else:
        st.markdown("""
<div class="glass-panel animate-fade-in" style="text-align:center;padding:40px 16px;border:2px dashed rgba(203,213,225,0.5) !important;">
<div style="font-size:2.5rem;margin-bottom:12px;opacity:0.45;">&#128202;</div>
<h3 style="color:#64748b;font-size:1rem;">Awaiting Assessment</h3>
<p style="color:#94a3b8;font-size:0.82rem;">Your analysis will appear here once you describe your symptoms.</p>
</div>
        """, unsafe_allow_html=True)

# ════════════════════════════════════════════════════════════
# COL 3 — HOME CARE & DETAILS
# ════════════════════════════════════════════════════════════
with col3:
    if is_emergency:
        # Get hospital recommendations
        hospitals = st.session_state.hospital_recommender.get_recommendations(28.6139, 77.2090, top_k=3)
        hospital_html = '<div style="margin-top:16px;padding-top:16px;border-top:1px solid rgba(185,28,28,0.2);">'
        hospital_html += '<h4 style="color:#b91c1c;font-size:0.8rem;margin-bottom:10px;text-transform:uppercase;letter-spacing:0.05em;">Nearest Hospitals</h4>'
        for h in hospitals:
            hospital_html += f'<div class="location-card"><div class="location-name">🏥 {h["name"]}</div>'
            hospital_html += f'<div class="location-meta"><span>{h["city"]}</span><span style="font-weight:700;color:#2563EB;">{h["distance"]:.2f} km</span></div></div>'
        hospital_html += '</div>'

        st.markdown(f"""
<div class="glass-panel glass-alert-critical animate-fade-in" style="padding:18px !important;">
<h3 style="color:#b91c1c;font-size:0.95rem;">&#128680; Emergency Resources</h3>
<ul class="custom-list" style="color:#991b1b;font-size:0.85rem;">
<li><strong>Emergency:</strong> 911 / 112</li>
<li><strong>Poison Control:</strong> 1-800-222-1222</li>
<li><strong>Crisis Line:</strong> 988</li>
</ul>
{hospital_html}
</div>
        """, unsafe_allow_html=True)

    elif latest_predictions and not is_critical:
        # Home care card with internal scroll if content is long
        home_care_tips = home_care_data.get("home_care", [])
        medical_attn_tips = home_care_data.get("seek_medical_attention_if", [])

        card3 = '<div class="glass-panel animate-slide-in" style="padding:18px !important;">'
        card3 += '<h3 style="display:flex;align-items:center;gap:8px;font-size:0.95rem;margin-bottom:10px;">&#127968; Supportive Home Care</h3>'
        if home_care_tips:
            card3 += "<ul class='custom-list'>"
            for tip in home_care_tips:
                card3 += f"<li style='color:#475569;font-size:0.85rem;'>{tip}</li>"
            card3 += "</ul>"
        else:
            card3 += "<p style='color:#64748b;font-size:0.85rem;'>No specific home care suggestions available.</p>"

        if medical_attn_tips:
            card3 += "<div style='margin-top:12px;padding-top:12px;border-top:1px solid #e2e8f0;'>"
            card3 += "<strong style='color:#ea580c;font-size:0.8rem;'>SEEK ATTENTION IF:</strong>"
            card3 += "<ul class='custom-list' style='margin-top:6px;'>"
            for tip in medical_attn_tips:
                card3 += f"<li style='color:#64748b;font-size:0.82rem;'>{tip}</li>"
            card3 += "</ul></div>"
        card3 += "</div>"
        st.markdown(card3, unsafe_allow_html=True)

        # Safety footer
        st.markdown("""
<div style="text-align:center;margin-top:8px;">
<span style="color:#94a3b8;font-size:0.72rem;text-transform:uppercase;letter-spacing:0.05em;">&#128737; MediAssist Safety Protocol Active</span>
</div>
        """, unsafe_allow_html=True)

    elif latest_predictions and is_critical:
        # Get hospital recommendations
        hospitals = st.session_state.hospital_recommender.get_recommendations(28.6139, 77.2090, disease_category=top_disease, top_k=3)
        hospital_html = '<div style="margin-top:16px;padding-top:16px;border-top:1px solid rgba(185,28,28,0.2);">'
        hospital_html += '<h4 style="color:#b91c1c;font-size:0.8rem;margin-bottom:10px;text-transform:uppercase;letter-spacing:0.05em;">Nearest Hospitals</h4>'
        for h in hospitals:
            hospital_html += f'<div class="location-card"><div class="location-name">🏥 {h["name"]}</div>'
            hospital_html += f'<div class="location-meta"><span>{h["city"]}</span><span style="font-weight:700;color:#2563EB;">{h["distance"]:.2f} km</span></div></div>'
        hospital_html += '</div>'

        st.markdown(f"""
<div class="glass-panel glass-alert-critical animate-fade-in" style="padding:18px !important;">
<h3 style="color:#b91c1c;font-size:0.95rem;">&#128680; Emergency Resources</h3>
<p style="color:#991b1b;font-size:0.85rem;">For critical conditions, home care is not appropriate. Please seek emergency medical attention immediately.</p>
{hospital_html}
</div>
        """, unsafe_allow_html=True)

    else:
        st.markdown("""
<div class="glass-panel animate-fade-in" style="text-align:center;padding:40px 16px;border:2px dashed rgba(203,213,225,0.5) !important;">
<div style="font-size:2.5rem;margin-bottom:12px;opacity:0.45;">&#127968;</div>
<h3 style="color:#64748b;font-size:1rem;">Home Care Tips</h3>
<p style="color:#94a3b8;font-size:0.82rem;">Personalized care recommendations will appear here.</p>
</div>
        """, unsafe_allow_html=True)

# ── SIDEBAR ─────────────────────────────────────────────────────────────────
with st.sidebar:
    st.markdown("<div style='padding:10px 0 20px;'><h3 style='color:#1e293b;margin:0;'>Settings &amp; Info</h3></div>", unsafe_allow_html=True)

    if st.button("&#128465;&#65039; Reset Assessment", use_container_width=True):
        st.session_state.history = []
        st.session_state.manager = ConversationManager()
        st.rerun()

    st.markdown("<div style='margin:20px 0;border-top:1px solid rgba(203,213,225,0.5);'></div>", unsafe_allow_html=True)

    st.markdown("""
<div class="glass-panel" style="padding:14px !important;">
<h4 style="font-size:0.85rem;color:#334155;margin-bottom:10px;">System Status</h4>
<div style="display:flex;justify-content:space-between;margin-bottom:5px;"><span style="color:#64748b;font-size:0.78rem;">Prediction Engine</span><span style="color:#16a34a;font-size:0.78rem;font-weight:600;">Online</span></div>
<div style="display:flex;justify-content:space-between;margin-bottom:5px;"><span style="color:#64748b;font-size:0.78rem;">Safety Layer</span><span style="color:#16a34a;font-size:0.78rem;font-weight:600;">Active</span></div>
<div style="display:flex;justify-content:space-between;"><span style="color:#64748b;font-size:0.78rem;">Consistency Val</span><span style="color:#16a34a;font-size:0.78rem;font-weight:600;">Active</span></div>
</div>
    """, unsafe_allow_html=True)

    st.markdown("""
<div style="margin-top:24px;background:rgba(241,245,249,0.5);padding:14px;border-radius:12px;border:1px solid #e2e8f0;">
<div style="display:flex;align-items:center;gap:8px;margin-bottom:6px;"><span style="font-size:1.1rem;">&#9878;&#65039;</span><strong style="font-size:0.78rem;color:#475569;">Medical Disclaimer</strong></div>
<p style="font-size:0.72rem;color:#64748b;margin:0;line-height:1.4;">MediAssist AI provides preliminary assistance only and is NOT a substitute for professional medical diagnosis.</p>
</div>
    """, unsafe_allow_html=True)
