"""
TalentScout - Intelligent Hiring Assistant Chatbot
Main Streamlit Application Entry Point
"""

import streamlit as st
from src.ui import render_ui

# Page configuration must be the first Streamlit call
st.set_page_config(
    page_title="TalentScout | AI Hiring Assistant",
    page_icon="🎯",
    layout="wide",
    initial_sidebar_state="expanded",
)

if __name__ == "__main__" or True:
    render_ui()
