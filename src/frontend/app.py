import streamlit as st
import sys
import json
from pathlib import Path
from datetime import datetime

# ── PATH SETUP ─────────────────────────────────────────────────────────────
project_root = Path(__file__).resolve().parents[2]
if str(project_root) not in sys.path:
    sys.path.append(str(project_root))

from src.conversational.triage_orchestrator import MediAssistTriageOrchestrator

st.set_page_config(
    page_title="MediAssist AI — WhatsApp Triage",
    page_icon="🏥",
    layout="wide",
    initial_sidebar_state="collapsed"
)

# ── INJECT WHATSAPP STYLES ─────────────────────────────────────────────────
css_path = project_root / "src/frontend/styles/whatsapp.css"
if css_path.exists():
    st.markdown(f"<style>{css_path.read_text(encoding='utf-8')}</style>", unsafe_allow_html=True)

# ── SESSION STATE ───────────────────────────────────────────────────────────
if "orchestrator" not in st.session_state:
    st.session_state.orchestrator = MediAssistTriageOrchestrator()

if "messages" not in st.session_state:
    st.session_state.messages = []

if "language" not in st.session_state:
    st.session_state.language = "en"

if "theme" not in st.session_state:
    st.session_state.theme = "light"

if st.session_state.theme == "dark":
    st.markdown("""
    <style>
    .stApp {
        background-color: #0B141A !important;
        background-image: radial-gradient(#202C33 1px, transparent 1px) !important;
        color: #E9EDEF !important;
    }
    .wa-header {
        background: #202C33 !important;
        box-shadow: 0 4px 12px rgba(0, 0, 0, 0.4) !important;
        border-bottom: 1px solid #2A3942 !important;
    }
    .wa-avatar {
        background-color: #111B21 !important;
    }
    .wa-bubble-incoming {
        background-color: #202C33 !important;
        color: #E9EDEF !important;
    }
    .wa-bubble-outgoing {
        background-color: #005C4B !important;
        color: #E9EDEF !important;
    }
    .wa-bubble-time {
        color: #8696A0 !important;
    }
    .wa-triage-card {
        background: #111B21 !important;
        color: #E9EDEF !important;
        border-left: 4px solid #00A884 !important;
    }
    .wa-emergency-card {
        background: #2D1519 !important;
        color: #FFCDD2 !important;
        border: 2px solid #EA0038 !important;
    }
    .wa-audit-header {
        background: #202C33 !important;
        border: 1px solid #2A3942 !important;
    }
    .wa-audit-header h3 {
        color: #25D366 !important;
    }
    .wa-audit-header p {
        color: #8696A0 !important;
    }
    [data-testid="stExpander"] {
        background-color: #111B21 !important;
        border: 1px solid #222E35 !important;
        color: #E9EDEF !important;
        border-radius: 10px !important;
    }
    [data-testid="stExpander"] details summary p {
        color: #E9EDEF !important;
        font-weight: 600 !important;
    }
    [data-testid="stChatInput"] textarea {
        background-color: #202C33 !important;
        color: #E9EDEF !important;
        border-color: #2A3942 !important;
    }
    .stMarkdown, p, span, label {
        color: #E9EDEF !important;
    }
    </style>
    """, unsafe_allow_html=True)

# ── TOP WHATSAPP HEADER ─────────────────────────────────────────────────────
lang_label = "🇮🇳 हिंदी" if st.session_state.language == "hi" else "🇬🇧 English"

st.markdown(f"""
<div class="wa-header">
    <div class="wa-header-profile">
        <div class="wa-avatar">🏥</div>
        <div>
            <div class="wa-title">MediAssist AI</div>
            <div class="wa-status"><span class="wa-status-dot"></span> Online &bull; Safety Firewall Active &bull; Bangalore Directory</div>
        </div>
    </div>
    <div style="font-size: 0.85rem; font-weight: 600; opacity: 0.95;">
        🔒 End-to-End Triage &bull; Pre-Clinical Guidance
    </div>
</div>
""", unsafe_allow_html=True)

# ── TWO-COLUMN WORKSPACE: CHAT & LIVE CLINICAL AUDIT ───────────────────────
chat_col, audit_col = st.columns([1.15, 0.85])

# ═══════════════════════════════════════════════════════════════════════════
# COLUMN 1: WHATSAPP CHAT THREAD
# ═══════════════════════════════════════════════════════════════════════════
with chat_col:
    # Action toolbar
    tcol1, tcol2, tcol3 = st.columns([0.4, 0.35, 0.25])
    with tcol1:
        selected_lang = st.radio("Language / भाषा", ["English (en)", "हिंदी (hi)"], horizontal=True, label_visibility="collapsed")
        st.session_state.language = "hi" if "हिंदी" in selected_lang else "en"
    with tcol2:
        theme_label = "☀️ Light" if st.session_state.theme == "dark" else "🌙 Dark"
        if st.button(f"{theme_label} Mode", use_container_width=True):
            st.session_state.theme = "light" if st.session_state.theme == "dark" else "dark"
            st.rerun()
    with tcol3:
        if st.button("🗑️ Clear Chat", use_container_width=True):
            st.session_state.messages = []
            st.session_state.orchestrator.reset()
            st.rerun()

    # Chat history container
    with st.container(height=520):
        if not st.session_state.messages:
            welcome_msg = (
                "नमस्ते! मैं मेडीअसिस्ट एआई (MediAssist AI) हूँ। कृपया अपने लक्षण और वे कब से शुरू हुए (जैसे: 3 दिन से बुखार) बताएं।"
                if st.session_state.language == "hi"
                else "Hello! I am MediAssist AI. Please describe your symptoms and timeline (e.g., 'Fever and headache for 3 days, getting worse')."
            )
            st.markdown(f"""
            <div class="wa-message-row wa-incoming-row">
                <div class="wa-bubble wa-bubble-incoming">
                    {welcome_msg}
                    <div class="wa-bubble-time">{datetime.now().strftime('%I:%M %p')}</div>
                </div>
            </div>
            """, unsafe_allow_html=True)
        else:
            for msg in st.session_state.messages:
                role = msg["role"]
                content = msg["content"]
                time_str = msg.get("time", datetime.now().strftime('%I:%M %p'))
                
                if role == "user":
                    st.markdown(f"""
                    <div class="wa-message-row wa-outgoing-row">
                        <div class="wa-bubble wa-bubble-outgoing">
                            {content}
                            <div class="wa-bubble-time">{time_str} <span class="wa-ticks">✓✓</span></div>
                        </div>
                    </div>
                    """, unsafe_allow_html=True)
                else:
                    is_emerg = msg.get("is_emergency", False)
                    card_class = "wa-emergency-card" if is_emerg else ""
                    st.markdown(f"""
                    <div class="wa-message-row wa-incoming-row">
                        <div class="wa-bubble wa-bubble-incoming {card_class}">
                            <div style="white-space: pre-wrap;">{content}</div>
                            <div class="wa-bubble-time">{time_str}</div>
                        </div>
                    </div>
                    """, unsafe_allow_html=True)

    # Chat input bar
    input_placeholder = "लक्षण या समस्या बताएं..." if st.session_state.language == "hi" else "Type your symptoms here..."
    if prompt := st.chat_input(input_placeholder):
        now_str = datetime.now().strftime('%I:%M %p')
        st.session_state.messages.append({"role": "user", "content": prompt, "time": now_str})
        
        # Process through 5-Layer Master Orchestrator
        res = st.session_state.orchestrator.process_message(prompt)
        
        st.session_state.messages.append({
            "role": "assistant",
            "content": res["message"],
            "time": datetime.now().strftime('%I:%M %p'),
            "is_emergency": res.get("is_emergency", False)
        })
        st.rerun()

# ═══════════════════════════════════════════════════════════════════════════
# COLUMN 2: LIVE CLINICAL AUDIT PANEL (FOR VIVA / DEMO)
# ═══════════════════════════════════════════════════════════════════════════
with audit_col:
    st.markdown("""
    <div class="wa-audit-header">
        <h3>
            🔬 Clinical Triage &amp; Defense Inspector
        </h3>
        <p style="font-size:0.75rem;color:#667781;margin:0;">
            Live architectural audit displaying state separation (Qwen Extractor &bull; Emergency Veto &bull; PubMedBERT &bull; Decision Layer).
        </p>
    </div>
    """, unsafe_allow_html=True)

    orch = st.session_state.orchestrator
    last_profile = getattr(orch, "last_profile", {})
    last_decision = getattr(orch, "last_decision", {})

    with st.expander("⏱️ 1. Extracted Symptoms & Timeline (Qwen)", expanded=True):
        if last_profile:
            st.json(last_profile)
        else:
            st.info("Awaiting input to extract timeline...")

    with st.expander("🚨 2. Emergency Safety Firewall (Pre-BERT Veto)", expanded=True):
        if st.session_state.messages and st.session_state.messages[-1].get("is_emergency"):
            st.error("🚨 EMERGENCY OVERRIDE TRIGGERED: ML Inference Halted!")
        else:
            st.success("🛡️ Safety Firewall Status: PASS (No acute life-threatening pattern)")

    with st.expander("🧠 3. Medical BERT (Top-3 Predictions)", expanded=True):
        if last_decision and last_decision.get("top_conditions"):
            for cond in last_decision["top_conditions"]:
                c_name = cond["condition"]
                c_conf = cond["confidence"]
                st.write(f"**{c_name}** — Match: `{c_conf}%`")
                st.progress(c_conf / 100.0)
        else:
            st.info("Awaiting routine prediction...")

    with st.expander("⚖️ 4. Decision Layer & Bangalore Hospitals", expanded=True):
        if last_decision:
            urgency = last_decision.get("urgency_level", "Low")
            badge_color = "#C62828" if urgency == "High" else "#E65100" if urgency == "Moderate" else "#2E7D32"
            st.markdown(f"**Urgency Level:** <span style='color:{badge_color};font-weight:700;'>{urgency}</span>", unsafe_allow_html=True)
            st.write(f"**Home Care Allowed:** {'Yes' if last_decision.get('allow_home_care') else '❌ Blocked (High Severity)'}")
            
            hospitals = last_decision.get("hospitals", [])
            if hospitals:
                st.markdown("**🏥 Recommended Facilities (Bangalore):**")
                for h in hospitals[:3]:
                    st.markdown(f"- **{h['name']}** ({h['distance']} km away)")
        else:
            st.info("Decision layer will calculate urgency upon classification.")

    st.markdown("""
    <div style="text-align:center;padding:10px;font-size:0.72rem;color:#8696A0;">
        MediAssist AI &bull; College Mini-Project &bull; Hybrid Architecture
    </div>
    """, unsafe_allow_html=True)
