import streamlit as st
import requests

st.set_page_config(page_title="MediAssist AI", layout="wide")

st.title("🏥 MediAssist AI: Healthcare Assistant")
st.markdown("---")

with st.sidebar:
    st.header("Settings")
    model_type = st.selectbox("Inference Model", ["Hybrid (Fast)", "Deep Transformer", "LLM Specialist"])
    st.info("Using Qwen2.5 for local synthesis.")

col1, col2 = st.columns([2, 1])

with col1:
    user_input = st.text_area("Describe your symptoms:", placeholder="e.g., I have a persistent headache and high fever...")
    if st.button("Analyze Symptoms", type="primary"):
        with st.spinner("Processing medical insights..."):
            # Call FastAPI backend
            st.success("Analysis Complete")
            st.subheader("Predicted Condition: Flu")
            st.progress(0.85, "Confidence Score")

with col2:
    st.subheader("Recommended Actions")
    st.info("1. Rest and hydrate\n2. Monitor temperature\n3. Consult a doctor if symptoms persist")
    
    st.subheader("Nearby Hospitals")
    st.map() # Placeholder for geographic recommendations
