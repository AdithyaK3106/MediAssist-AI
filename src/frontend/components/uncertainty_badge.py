import streamlit as st

def render_uncertainty_badge(level):
    """Renders a minimalistic uncertainty indicator."""
    if level == "Low":
        st.success("Low Uncertainty")
    elif level == "Medium":
        st.warning("Medium Uncertainty")
    else:
        st.error("High Uncertainty")
