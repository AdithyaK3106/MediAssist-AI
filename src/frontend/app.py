import streamlit as st
import sys
import json
from pathlib import Path

# Add project root to path to ensure imports work
project_root = Path(__file__).resolve().parents[2]
if str(project_root) not in sys.path:
    sys.path.append(str(project_root))

from src.conversational.conversation_manager import ConversationManager
from src.models.classical_ml.predict import ClassicalInferencePipeline
from src.recommendation.home_care import HomeCareRecommender
from src.frontend.utils.confidence_formatter import normalize_topk_probabilities, get_confidence_label, get_confidence_color

st.set_page_config(page_title="MediAssist AI", layout="wide", page_icon="🏥", initial_sidebar_state="expanded")

# Load disease display names
try:
    with open(project_root / "data" / "disease_display_names.json", "r", encoding="utf-8") as f:
        DISEASE_DISPLAY_NAMES = json.load(f)
except Exception:
    DISEASE_DISPLAY_NAMES = {}

# Read and inject custom CSS files
styles_dir = project_root / "src/frontend/styles"
try:
    css_files = ["main.css", "glassmorphism.css", "animations.css", "components.css", "theme.css"]
    css_content = ""
    for css_file in css_files:
        with open(styles_dir / css_file, "r", encoding="utf-8") as f:
            css_content += f.read() + "\n"
    st.markdown(f"<style>{css_content}</style>", unsafe_allow_html=True)
except Exception as e:
    st.warning(f"Could not load custom styles: {e}")

# Initialize pipelines and manager in session state
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
    
if "history" not in st.session_state:
    st.session_state.history = []

# Top Navigation / Header
st.markdown("""
<div class='glass-panel animate-fade-in' style='padding: 15px 30px; display: flex; justify-content: space-between; align-items: center; margin-bottom: 30px;'>
    <div style='display: flex; align-items: center; gap: 12px;'>
        <div style='background: linear-gradient(135deg, #3B82F6 0%, #2563EB 100%); width: 40px; height: 40px; border-radius: 10px; display: flex; align-items: center; justify-content: center; font-size: 1.2rem; color: white; box-shadow: 0 4px 10px rgba(37,99,235,0.3);'>🏥</div>
        <div>
            <h2 style='margin: 0; font-size: 1.2rem; color: #1e293b;'>MediAssist AI</h2>
            <div style='font-size: 0.8rem; color: #64748b;'>Safety-Aware Healthcare Platform</div>
        </div>
    </div>
    <div style='display: flex; gap: 10px;'>
        <span class='badge' style='background: #dcfce7; color: #166534;'><span style='margin-right:4px;'>🟢</span> System Operational</span>
        <span class='badge' style='background: #f1f5f9; color: #475569;'>v2.1.0</span>
    </div>
</div>
""", unsafe_allow_html=True)

# Main Layout
col1, col2 = st.columns([1.2, 1])

# Left Column: Conversation
with col1:
    chat_html = "<div class='glass-panel animate-slide-in'>"
    chat_html += "<h3 style='display: flex; align-items: center; gap: 8px;'><span style='font-size: 1.2rem;'>💬</span> Patient Assessment</h3>"
    
    # Welcome Screen
    if not st.session_state.history:
        chat_html += """
        <div style='text-align: center; padding: 40px 20px;'>
            <div style='font-size: 3rem; margin-bottom: 10px;'>👋</div>
            <h3 style='margin-bottom: 10px; color: #1e293b;'>Welcome to MediAssist</h3>
            <p style='color: #64748b; margin-bottom: 20px;'>Please describe your symptoms to begin your safety-aware assessment.</p>
            
            <div style='display: flex; flex-wrap: wrap; justify-content: center; gap: 8px;'>
                <div class='symptom-chip'>Fever and chills</div>
                <div class='symptom-chip'>Severe headache</div>
                <div class='symptom-chip'>Persistent cough</div>
                <div class='symptom-chip'>Stomach pain</div>
            </div>
        </div>
        """
    else:
        # Chat Display
        for msg in st.session_state.history:
            role_class = "glass-chat-user" if msg['role'] == "user" else "glass-chat-assistant"
            icon = "👤" if msg['role'] == "user" else "🤖"
            chat_html += f"""
            <div class='animate-fade-in {role_class}'>
                <div style='display: flex; align-items: center; gap: 8px; margin-bottom: 4px;'>
                    <span style='font-size: 0.9rem;'>{icon}</span>
                    <strong style='font-size: 0.85rem; color: #64748b; text-transform: uppercase; letter-spacing: 0.05em;'>{msg['role']}</strong>
                </div>
                <div style='color: #334155; line-height: 1.5;'>{msg['content']}</div>
            </div>
            """
            
    chat_html += "</div>"
    st.markdown(chat_html, unsafe_allow_html=True)

    # Chat Input
    if prompt := st.chat_input("Type your symptoms here (e.g., 'I have a high fever and headache')..."):
        st.session_state.history.append({"role": "user", "content": prompt})
        
        if st.session_state.pipeline:
            response = st.session_state.manager.handle_message(prompt, prediction_pipeline=st.session_state.pipeline)
        else:
            response = "Sorry, the prediction pipeline is not loaded. I cannot provide assessments right now."
        
        st.session_state.history.append({"role": "assistant", "content": response})
        st.rerun()

# Right Column: Insights & Recommendations
with col2:
    manager = st.session_state.manager
    latest_uncertainty = getattr(manager, 'latest_uncertainty', None)
    latest_predictions = getattr(manager, 'latest_predictions', [])
    latest_flag_unreliable = getattr(manager, 'latest_flag_unreliable', False)
    latest_consistency = getattr(manager, 'latest_consistency_data', None)

    # Handle Emergency UI
    if manager.latest_predictions == [] and getattr(manager, 'latest_flag_unreliable', False):
        st.markdown("""
        <div class='glass-panel animate-pulse-critical hover-lift' style='border: 2px solid #b91c1c; background-color: #fef2f2;'>
            <h2 style='color: #b91c1c; display: flex; align-items: center; gap: 12px; font-size: 1.8rem;'><span style='font-size: 2.2rem;'>🚨</span> CRITICAL MEDICAL ALERT</h2>
            <div style='margin-top: 15px; padding: 15px; background: rgba(185, 28, 28, 0.1); border-radius: 8px;'>
                <h4 style='color: #991b1b; margin-top: 0;'>Immediate Action Required</h4>
                <p style='color: #7f1d1d; font-size: 1.1rem; line-height: 1.6;'>
                    Your reported symptoms strongly indicate a potential medical emergency. 
                    <strong>Please bypass casual home care and seek immediate professional evaluation.</strong>
                </p>
                <ul class='custom-list' style='color: #991b1b; font-weight: bold;'>
                    <li>Call your local emergency number immediately.</li>
                    <li>Or proceed directly to the nearest hospital emergency room.</li>
                </ul>
            </div>
        </div>
        """, unsafe_allow_html=True)
        
        # Trust/Safety Footer
        st.markdown("""
        <div style='text-align: center; margin-top: 20px;'>
            <span style='color: #b91c1c; font-size: 0.85rem; font-weight: bold; text-transform: uppercase; letter-spacing: 0.05em;'>🛡️ MediAssist Emergency Override Active</span>
        </div>
        """, unsafe_allow_html=True)

    elif latest_predictions:
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
        
        top_reasoning = None
        if latest_consistency:
            top_reasoning = next((r for r in latest_consistency["reasoning"] if r["disease"] == top_disease), None)
            
        conf_label = get_confidence_label(top_rel_prob)
        conf_color = get_confidence_color(top_rel_prob, severity)

        # 1. Main Prediction Card
        card1_html = f"""
        <div class='glass-panel hover-lift animate-slide-in' style='animation-delay: 0.1s;'>
            <h3 style='display: flex; align-items: center; gap: 8px;'><span style='font-size: 1.2rem;'>🩺</span> Assessment Results</h3>
            <h2 style='color: {conf_color}; margin-top: 10px; margin-bottom: 15px;'>{friendly_name}</h2>
            <div style='display: flex; gap: 10px; flex-wrap: wrap; margin-bottom: 15px;'>
                <span class='badge' style='background-color: {conf_color}; color: white; padding: 6px 12px; font-size: 0.85rem;'>Severity: {severity.upper()}</span>
                <span class='badge badge-outline' style='color: {conf_color}; padding: 6px 12px; font-size: 0.85rem;'>Match: {conf_label}</span>
            </div>
            <div class="confidence-bar-bg">
                <div class="progress-bar-fill" style="width: {top_rel_prob}%; background-color: {conf_color}; height: 100%;"></div>
            </div>
            <p style='color: #4B5563; font-size: 0.95rem; line-height: 1.6;'>{description}</p>
        </div>
        """
        st.markdown(card1_html, unsafe_allow_html=True)

        # 2. Emergency Guidance / Next Steps Card
        alert_class = f"glass-alert-{severity}"
        if is_critical or latest_flag_unreliable or (top_reasoning and top_reasoning["consistency"] == "LOW"):
            if is_critical:
                alert_class += " animate-pulse-critical"
        
        card2_html = f"<div class='glass-panel {alert_class} animate-slide-in' style='animation-delay: 0.2s;'>"
        
        if is_critical:
            card2_html += f"<h3 style='color: #b91c1c; display: flex; align-items: center; gap: 8px;'><span>🚨</span> Critical Medical Alert</h3>"
            card2_html += "<p style='color: #991b1b; font-weight: 500;'>Immediate medical attention is strongly recommended. Please contact emergency services or visit a hospital immediately.</p>"
        elif latest_flag_unreliable or severity == "high" or (top_reasoning and top_reasoning["consistency"] == "LOW"):
            card2_html += f"<h3 style='color: #c2410c; display: flex; align-items: center; gap: 8px;'><span>⚠️</span> Medical Consultation Advised</h3>"
            if top_reasoning and top_reasoning["consistency"] == "LOW" and latest_flag_unreliable:
                 card2_html += "<p style='color: #c2410c;'>Your symptoms overlap with multiple possible conditions. A professional diagnosis is needed.</p>"
            card2_html += "<ul class='custom-list' style='color: #9a3412;'><li>Monitor symptoms closely</li><li>Seek immediate attention if symptoms worsen</li></ul>"
        else:
            card2_html += f"<h3 style='color: #15803d; display: flex; align-items: center; gap: 8px;'><span>📌</span> Recommended Next Steps</h3>"
            card2_html += "<ul class='custom-list' style='color: #166534;'><li>Monitor symptoms for the next 24–48 hours</li><li>Stay hydrated and get adequate rest</li><li>Follow the supportive care suggestions below</li></ul>"
        
        card2_html += "</div>"
        st.markdown(card2_html, unsafe_allow_html=True)

        # 3. Home Care Recommendations Card
        if not is_critical:
            card3_html = f"<div class='glass-panel animate-slide-in' style='animation-delay: 0.3s;'>"
            card3_html += "<h3 style='display: flex; align-items: center; gap: 8px;'><span style='font-size: 1.2rem;'>🏠</span> Supportive Home Care</h3>"
            
            home_care_tips = home_care_data.get("home_care", [])
            if home_care_tips:
                card3_html += "<ul class='custom-list'>"
                for tip in home_care_tips:
                    card3_html += f"<li style='color: #475569;'>{tip}</li>"
                card3_html += "</ul>"
            else:
                card3_html += "<p style='color: #64748b;'>No specific home care suggestions available for this condition.</p>"
                
            medical_attention_tips = home_care_data.get("seek_medical_attention_if", [])
            if medical_attention_tips:
                card3_html += "<div style='margin-top: 15px; padding-top: 15px; border-top: 1px solid #e2e8f0;'>"
                card3_html += "<strong style='color: #ea580c; font-size: 0.9rem;'>SEEK ATTENTION IF YOU EXPERIENCE:</strong>"
                card3_html += "<ul class='custom-list' style='margin-top: 8px;'>"
                for tip in medical_attention_tips:
                    card3_html += f"<li style='color: #64748b; font-size: 0.9rem;'>{tip}</li>"
                card3_html += "</ul></div>"
                
            card3_html += "</div>"
            st.markdown(card3_html, unsafe_allow_html=True)

        # 4. Expandable Sections (Alternatives & Reasoning)
        # Using Streamlit expander for this part, wrapped nicely
        st.markdown(f"<div class='glass-panel animate-slide-in' style='animation-delay: 0.4s; padding: 10px 24px !important;'>", unsafe_allow_html=True)
        
        with st.expander("View Alternative Possibilities"):
            alt_html = "<ul class='custom-list'>"
            for pred in normalized_preds[1:]:
                alt_disease = pred['disease']
                alt_friendly = DISEASE_DISPLAY_NAMES.get(alt_disease, {"friendly_name": alt_disease})['friendly_name']
                alt_rel_prob = pred['relative_prob']
                alt_label = get_confidence_label(alt_rel_prob)
                alt_html += f"<li><strong style='color: #334155;'>{alt_friendly}</strong> <br><span style='color: #94a3b8; font-size: 0.85rem;'>Match: {alt_label}</span></li>"
            alt_html += "</ul>"
            st.markdown(alt_html, unsafe_allow_html=True)
            
        if latest_consistency and latest_consistency.get("detected_symptoms"):
            with st.expander("Why were these suggested?"):
                symptoms_list = ", ".join([s.capitalize() for s in latest_consistency["detected_symptoms"]])
                reason_html = f"<p style='color: #475569; font-size: 0.9rem;'><strong>Symptoms Detected:</strong> {symptoms_list}</p>"
                
                if top_reasoning:
                    disease_cats = ", ".join([c.capitalize() for c in top_reasoning["disease_categories"]])
                    consistency_level = top_reasoning["consistency"]
                    if disease_cats:
                        reason_html += f"<p style='color: #64748b; font-size: 0.85rem;'><em>{friendly_name} typically presents with: {disease_cats} symptoms.</em></p>"
                    if consistency_level == "HIGH":
                        reason_html += "<p style='color: #16a34a; font-size: 0.85rem; font-weight: 500;'>✓ High medical consistency</p>"
                    elif consistency_level == "LOW":
                        reason_html += "<p style='color: #ea580c; font-size: 0.85rem; font-weight: 500;'>⚠️ Low medical consistency (partial symptom overlap)</p>"
                st.markdown(reason_html, unsafe_allow_html=True)
                        
        st.markdown("</div>", unsafe_allow_html=True)
        
        # Trust/Safety Footer
        st.markdown("""
        <div style='text-align: center; margin-top: 20px;'>
            <span style='color: #94a3b8; font-size: 0.75rem; text-transform: uppercase; letter-spacing: 0.05em;'>🛡️ MediAssist Safety Protocol Active</span>
        </div>
        """, unsafe_allow_html=True)

    else:
        # Empty State
        st.markdown("""
        <div class='glass-panel animate-fade-in' style='text-align: center; padding: 60px 20px; border: 2px dashed rgba(203, 213, 225, 0.5) !important;'>
            <div style='font-size: 3rem; margin-bottom: 15px; opacity: 0.5;'>📊</div>
            <h3 style='color: #64748b;'>Awaiting Assessment</h3>
            <p style='color: #94a3b8; font-size: 0.9rem;'>Once you describe your symptoms, your safety-aware analysis and personalized recommendations will appear here.</p>
        </div>
        """, unsafe_allow_html=True)

# Sidebar (SaaS Style)
with st.sidebar:
    st.markdown("""
    <div style='padding: 10px 0; margin-bottom: 20px;'>
        <h3 style='color: #1e293b; margin: 0;'>Settings & Info</h3>
    </div>
    """, unsafe_allow_html=True)
    
    if st.button("🗑️ Reset Assessment", use_container_width=True):
        st.session_state.history = []
        st.session_state.manager = ConversationManager()
        st.rerun()
        
    st.markdown("<div style='margin: 30px 0; border-top: 1px solid rgba(203, 213, 225, 0.5);'></div>", unsafe_allow_html=True)
    
    st.markdown("""
    <div class='glass-panel' style='padding: 15px !important;'>
        <h4 style='font-size: 0.9rem; color: #334155; margin-bottom: 10px;'>System Status</h4>
        <div style='display: flex; justify-content: space-between; margin-bottom: 5px;'>
            <span style='color: #64748b; font-size: 0.8rem;'>Prediction Engine</span>
            <span style='color: #16a34a; font-size: 0.8rem; font-weight: 500;'>Online</span>
        </div>
        <div style='display: flex; justify-content: space-between; margin-bottom: 5px;'>
            <span style='color: #64748b; font-size: 0.8rem;'>Safety Layer</span>
            <span style='color: #16a34a; font-size: 0.8rem; font-weight: 500;'>Active</span>
        </div>
        <div style='display: flex; justify-content: space-between;'>
            <span style='color: #64748b; font-size: 0.8rem;'>Consistency Val</span>
            <span style='color: #16a34a; font-size: 0.8rem; font-weight: 500;'>Active</span>
        </div>
    </div>
    """, unsafe_allow_html=True)
    
    st.markdown("""
    <div style='margin-top: auto; padding-top: 40px;'>
        <div style='background: rgba(241, 245, 249, 0.5); padding: 15px; border-radius: 12px; border: 1px solid #e2e8f0;'>
            <div style='display: flex; align-items: center; gap: 8px; margin-bottom: 8px;'>
                <span style='font-size: 1.2rem;'>⚖️</span>
                <strong style='font-size: 0.8rem; color: #475569;'>Medical Disclaimer</strong>
            </div>
            <p style='font-size: 0.75rem; color: #64748b; margin: 0; line-height: 1.4;'>
                MediAssist AI provides preliminary assistance only and is NOT a substitute for professional medical diagnosis.
            </p>
        </div>
    </div>
    """, unsafe_allow_html=True)
