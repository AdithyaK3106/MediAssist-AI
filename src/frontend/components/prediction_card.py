import streamlit as st

def render_prediction_card(top_predictions):
    """Renders an elegant card for predictions."""
    with st.container():
        st.markdown("### 📋 Possible Conditions")
        for i, pred in enumerate(top_predictions[:3]):
            st.write(f"{i+1}. **{pred['disease']}** — {pred['probability']*100:.1f}%")
