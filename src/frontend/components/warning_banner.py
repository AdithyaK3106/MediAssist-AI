import streamlit as st

def render_warning_banner(message):
    """Renders a highlighted warning card."""
    st.error(f"⚠️ {message}")
