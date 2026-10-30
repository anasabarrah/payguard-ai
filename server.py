"""Single-process entry point: the Streamlit dashboard plus the shop and webhook routes.

Run with:  streamlit run server.py
"""
import streamlit as st

from shop_api import routes

app = st.App("app.py", routes=routes)
