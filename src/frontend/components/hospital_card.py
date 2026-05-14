import streamlit as st

def render_hospital_card(name, distance):
    """Renders clean hospital recommendation cards."""
    with st.container():
        st.markdown("### 🏥 Recommended Facility")
        st.write(f"**{name}**")
        st.write(f"Approx Distance: {distance} km")
