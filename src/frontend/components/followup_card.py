import streamlit as st

def render_followup_card(questions):
    """Renders follow-up questions neatly."""
    with st.container():
        st.markdown("**To improve prediction reliability:**")
        for q in questions:
            st.write(f"• {q}")
