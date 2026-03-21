import streamlit as st
import os
import sys
import pandas as pd
import numpy as np

# Fix for refactored structure: add project root to path so 'src' can be imported
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from extraction.agent import IDPExtractor
from extraction.loader import load_text_documents
from synthesis.database import MedBridgeStore
from synthesis.agent import SynthesisAgent

# Page configuration
st.set_page_config(
    page_title="MedBridge AI Dashboard",
    page_icon="🏥",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for a premium look
st.markdown("""
    <style>
    .main { background-color: #0e1117; }
    .stButton>button { width: 100%; border-radius: 5px; height: 3em; background-color: #262730; color: white; }
    .stTextInput>div>div>input { background-color: #262730; color: white; }
    .success-box { padding: 1rem; border-radius: 0.5rem; background-color: rgba(40, 167, 69, 0.1); border: 1px solid rgba(40, 167, 69, 0.2); color: #28a745; }
    </style>
    """, unsafe_allow_html=True)

@st.cache_resource
def load_resources():
    """Initializes the Gemini agents and database store."""
    store = MedBridgeStore()
    extractor = IDPExtractor()
    agent = SynthesisAgent(store=store)
    return store, extractor, agent

# Initialize Resources
with st.spinner("Connecting to Gemini Intelligence..."):
    st.session_state.store, st.session_state.extractor, st.session_state.agent = load_resources()

if 'chat_history' not in st.session_state:
    st.session_state.chat_history = []

# Sidebar
with st.sidebar:
    st.title("🏥 MedBridge AI")
    st.success("✓ Gemini 2.0 Flash Active")
    st.info("Direct Mapping Deduplication Enabled")
    
    st.divider()
    if st.button("Clear Chat History"):
        st.session_state.chat_history = []
        st.rerun()

# Main UI Tabs
tab1, tab2, tab3 = st.tabs(["📄 Document Analysis", "🧠 Medical Assistant", "📊 Database View"])

# --- Tab 1: Document Analysis ---
with tab1:
    st.header("Upload Medical Documents")
    st.write("Extract structured facility and NGO data using Gemini AI.")
    
    uploaded_file = st.file_uploader("Choose a Markdown or Text file", type=['md', 'txt'])
    if uploaded_file is not None:
        temp_path = os.path.join("data", "temp_upload.md")
        os.makedirs("data", exist_ok=True)
        with open(temp_path, "wb") as f:
            f.write(uploaded_file.getbuffer())
        
        if st.button("Analyze & Store in Database"):
            with st.spinner("Analyzing document with Gemini..."):
                docs = load_text_documents(temp_path)
                if docs:
                    extraction = st.session_state.extractor.extract_from_text(docs[0].page_content)
                    st.write(f"Found {len(extraction.facilities)} facilities, {len(extraction.ngos)} NGOs, and {len(extraction.other_organizations)} other entities.")
                    st.session_state.store.add_extractions(extraction, source_doc=uploaded_file.name)
                    st.toast(f"Data from {uploaded_file.name} persisted to LanceDB", icon="💾")
                    with st.expander("Show Extracted JSON"):
                        st.json(extraction.model_dump())
                    os.remove(temp_path)

# --- Tab 2: Medical Assistant ---
with tab2:
    st.header("Intelligence Assistant")
    st.write("Synthesize insights from your local database using Gemini RAG.")
    
    for message in st.session_state.chat_history:
        with st.chat_message(message["role"]):
            st.markdown(message["content"])
            
    if prompt := st.chat_input("Ask about facility capabilities (e.g., 'Who has MRI?')"):
        st.session_state.chat_history.append({"role": "user", "content": prompt})
        with st.chat_message("user"):
            st.markdown(prompt)
            
        with st.chat_message("assistant"):
            with st.spinner("Thinking..."):
                response = st.session_state.agent.answer_question(prompt)
                st.markdown(response)
                st.session_state.chat_history.append({"role": "assistant", "content": response})

# --- Tab 3: Database View ---
with tab3:
    st.header("Stored Medical Data")
    
    facilities_df = st.session_state.store.search_facilities("", limit=500)
    ngos_df = st.session_state.store.search_ngos("", limit=500)
    
    def format_list_column(val):
        """Clean list/str formatting for UI."""
        if val is None: return "None"
        s = str(val).strip()
        if s.lower() in ["nan", "none", "", "unknown", "[]", "['']", "[none]"]: return "None"
        
        # Detect if it's a bracketed "list-like" string or actual list
        if isinstance(val, (list, pd.Series, np.ndarray)) or s.startswith("["):
            try:
                # If it's already a list-like object, use it
                items = val if isinstance(val, (list, pd.Series, np.ndarray)) else []
                
                # If it's a string starting with [, parse it
                if not items and s.startswith("["):
                    import re
                    # Pure surgical cleaning: remove outer brackets
                    core = s[1:-1].strip()
                    # Split by common patterns: ' ' or " " or , or newline
                    # This handles ['A' 'B'] and ["A" "B"] and ['A', 'B']
                    parts = re.split(r"['\"]\s+['\"]|['\"],\s+['\"]|,", core)
                    items = [p.strip("'\" \n\t") for p in parts if p.strip("'\" \n\t")]
                
                if not items: return "None"
                # Final clean of noise items
                cleaned = [str(x).strip() for x in items if x and str(x).strip().lower() not in ["", "nan", "none", "unknown", "+", "[]"]]
                return ", ".join(cleaned) if cleaned else "None"
            except: 
                pass
        
        # Final fallback cleaning for any remaining brackets/quotes
        res = s.replace("[", "").replace("]", "").replace("'", "").replace('"', "").strip()
        return res if res and res.lower() not in ["nan", "none", "unknown"] else "None"

    col1, col2 = st.columns(2)
    with col1:
        st.subheader("Facilities")
        if not facilities_df.empty:
            f_display = facilities_df.drop(columns=['vector'], errors='ignore')
            
            # Cast numeric columns to avoid ArrowTypeError
            num_cols = ['yearEstablished', 'numberDoctors', 'capacity', 'area', 'latitude', 'longitude']
            for c in num_cols:
                if c in f_display.columns:
                    f_display[c] = pd.to_numeric(f_display[c], errors='coerce')
                    if c in ['yearEstablished', 'numberDoctors', 'capacity', 'area']:
                        f_display[c] = f_display[c].apply(lambda x: str(int(x)) if pd.notna(x) else "None")
                    else:
                        f_display[c] = f_display[c].apply(lambda x: str(x) if pd.notna(x) else "None")
            
            list_cols = ['capability', 'equipment', 'procedure', 'phone_numbers', 'websites', 'specialties', 'affiliationTypeIds']
            for col in list_cols:
                if col in f_display.columns: f_display[col] = f_display[col].apply(format_list_column)
            
            st.dataframe(f_display.fillna("None"), hide_index=True)
            
            # Full Download
            full_f = st.session_state.store.get_all_facilities().drop(columns=['vector'], errors='ignore')
            for col in ['capability', 'equipment', 'procedure', 'phone_numbers', 'websites']:
                if col in full_f.columns: full_f[col] = full_f[col].apply(format_list_column)
            st.download_button("Download Facilities CSV", full_f.to_csv(index=False).encode('utf-8'), "facilities.csv", "text/csv")
        else: st.info("No facilities stored.")

    with col2:
        st.subheader("NGOs")
        if not ngos_df.empty:
            n_display = ngos_df.drop(columns=['vector'], errors='ignore')
            
            # Cast numeric columns to avoid ArrowTypeError
            for c in ['yearEstablished', 'latitude', 'longitude']:
                if c in n_display.columns:
                    n_display[c] = pd.to_numeric(n_display[c], errors='coerce')
                    if c == 'yearEstablished':
                        n_display[c] = n_display[c].apply(lambda x: str(int(x)) if pd.notna(x) else "None")
                    else:
                        n_display[c] = n_display[c].apply(lambda x: str(x) if pd.notna(x) else "None")
                    
            ngo_list_cols = ['phone_numbers', 'websites', 'countries', 'email', 'missionStatement']
            for col in ngo_list_cols:
                if col in n_display.columns: n_display[col] = n_display[col].apply(format_list_column)
            st.dataframe(n_display.fillna("None"), hide_index=True)
            
            # Full Download
            full_n = st.session_state.store.get_all_ngos().drop(columns=['vector'], errors='ignore')
            for col in ['phone_numbers', 'websites']:
                if col in full_n.columns: full_n[col] = full_n[col].apply(format_list_column)
            st.download_button("Download NGOs CSV", full_n.to_csv(index=False).encode('utf-8'), "ngos.csv", "text/csv")
        else: st.info("No NGOs stored.")

st.caption("Virtue Foundation x MedBridgeAI | Gemini Powered Intelligence")
