import streamlit as st

def render_chat_message(role, content):
    """Renders a chat message using Streamlit's native component."""
    with st.chat_message(role):
        st.write(content)
