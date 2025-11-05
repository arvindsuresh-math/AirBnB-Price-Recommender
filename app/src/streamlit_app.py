# src/streamlit_app.py
# Basic Python libraries
import os  # For working with file paths and directories
import streamlit as st  # The main web app framework
import importlib.util
import sys  # For importing modules

# Configure the Streamlit page - MUST be first Streamlit command
st.set_page_config(layout="wide", page_title="NYC Airbnb Price Navigator")

# Get the directory where this script is located
script_dir = os.path.dirname(os.path.abspath(__file__))

# Use a folder name other than "pages" to avoid Streamlit's automatic multipage registration.
PAGE_MODULES_DIR = os.path.join(script_dir, "page_modules")

def load_page_module(page_name):
    """Dynamically load a page module from the page_modules directory."""
    module_path = os.path.join(PAGE_MODULES_DIR, f"{page_name}.py")
    if not os.path.exists(module_path):
        raise FileNotFoundError(f"Module file not found: {module_path}")
    spec = importlib.util.spec_from_file_location(page_name, module_path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[page_name] = module
    spec.loader.exec_module(module)
    return module

def page1_wrapper():
    try:
        load_page_module("page1")
    except Exception as e:
        st.error(f"Error loading page: {e}")
        import traceback
        st.code(traceback.format_exc())

def page2_wrapper():
    try:
        load_page_module("page2")
    except Exception as e:
        st.error(f"Error loading page: {e}")
        import traceback
        st.code(traceback.format_exc())

page_names_to_funcs = {
    "Main Map View": page1_wrapper,
    "Price Estimator": page2_wrapper,
}

st.sidebar.title("Select a page")
selected_page = st.sidebar.selectbox("", list(page_names_to_funcs.keys()))
page_names_to_funcs[selected_page]()