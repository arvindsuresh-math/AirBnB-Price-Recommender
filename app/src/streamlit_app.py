# src/streamlit_app.py
# Basic Python libraries
import os  # For working with file paths and directories
import streamlit as st  # The main web app framework
import importlib.util
import sys  # For importing modules

# Configure the Streamlit page - MUST be first Streamlit command
st.set_page_config(layout="wide", page_title="Airbnb Price Navigator")

## Custom CSS for Cabin font and navy blue/cream color scheme
#st.markdown("""
#    <style>
#    @import url('https://fonts.googleapis.com/css2?family=Cabin:wght@400;500;600;700&display=swap');
#    
#    /* Apply Cabin font to all text */
#    html, body, [class*="css"], .stApp {
#        font-family: 'Cabin', sans-serif;
#    }
#    
#    /* Main app background */
#    .stApp {
#        background-color: #F5F5DC;
#    }
#    
#    /* Sidebar styling */
#    [data-testid="stSidebar"] {
#        background-color: #001f3f;
#    }
#    
#    [data-testid="stSidebar"] * {
#        color: #F5F5DC !important;
#    }
#    
#    /* Sidebar headers */
#    [data-testid="stSidebar"] h1, 
#    [data-testid="stSidebar"] h2, 
#    [data-testid="stSidebar"] h3 {
#        color: #F5F5DC !important;
#    }
#    
#    /* Main content headers */
#    h1, h2, h3, h4, h5, h6 {
#        color: #001f3f;
#        font-family: 'Cabin', sans-serif;
#    }
#    
#    /* Buttons */
#    .stButton > button {
#        background-color: #001f3f;
#        color: #F5F5DC;
#        border: 2px solid #001f3f;
#        font-family: 'Cabin', sans-serif;
#        font-weight: 600;
#    }
#    
#    .stButton > button:hover {
#        background-color: #003366;
#        border-color: #003366;
#    }
#    
#    /* Input fields and selectboxes */
#    .stTextInput > div > div > input,
#    .stSelectbox > div > div > select,
#    .stMultiSelect > div > div,
#    .stNumberInput > div > div > input {
#        background-color: #F5F5DC !important;
#        color: #001f3f !important;
#        border-color: #001f3f !important;
#    }
#    
#    /* Ensure text color in all input elements */
#    input, select, textarea {
#        color: #001f3f !important;
#    }
#
#    /* Slider styling */
#    .stSlider > div > div > div {
#        background-color: #001f3f;
#    }
#    
#    /* Dataframe/table styling */
#    .dataframe {
#        background-color: #F5F5DC;
#        color: #001f3f;
#    }
#    
#    /* Metric styling */
#    [data-testid="stMetricValue"] {
#        color: #001f3f;
#    }
#    
#    /* Info/warning/error boxes */
#    .stAlert {
#        background-color: #F5F5DC;
#        border-color: #001f3f;
#        color: #001f3f;
#    }
#    
#    /* Tab styling */
#    .stTabs [data-baseweb="tab-list"] {
#        background-color: #F5F5DC;
#    }
#    
#    .stTabs [data-baseweb="tab"] {
#        color: #001f3f;
#        font-family: 'Cabin', sans-serif;
#    }
#    
#    .stTabs [aria-selected="true"] {
#        background-color: #001f3f;
#        color: #F5F5DC !important;
#    }
#    
#    /* Expander styling */
#    .streamlit-expanderHeader {
#        background-color: #F5F5DC;
#        color: #001f3f;
#        font-family: 'Cabin', sans-serif;
#    }
#    
#    /* Code blocks */
#    .stCodeBlock {
#        background-color: #F5F5DC;
#        border: 1px solid #001f3f;
#    }
#    
#    /* Links */
#    a {
#        color: #001f3f;
#        font-weight: 600;
#    }
#    
#    a:hover {
#        color: #003366;
#    }
#    </style>
#    """, unsafe_allow_html=True)

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
    "Price Recommender": page2_wrapper,
}

st.sidebar.title("Select a page")
selected_page = st.sidebar.selectbox("", list(page_names_to_funcs.keys()))
page_names_to_funcs[selected_page]()