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

# @st.cache_resource
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
    st.info("🛡️ Local LLM Fallback (Qwen) Ready")
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

# --- Tab 3: Database View (Map) ---
with tab3:
    col_a, col_b = st.columns([3, 1])
    with col_a:
        st.subheader("Global Medical Resource Map")
    with col_b:
        if st.button("🔄 Refresh Analyzed Data"):
            st.session_state.store = MedBridgeStore()
            st.rerun()
    
    # 1. Load data from LanceDB (Analyzed Data)
    f_all = st.session_state.store.get_all_facilities()
    n_all = st.session_state.store.get_all_ngos()
    
    # 2. Load data from Original Dataset (Reference/Miner)
    try:
        csv_path = "data/Virtue Foundation Ghana v0.3 - Sheet1.csv"
        df_dataset = pd.read_csv(csv_path, encoding='latin1')
        df_dataset.columns = [c.strip() for c in df_dataset.columns]
        
        # Support both dedicated columns and buried text coordinates
        import re
        def extract_coord(text, is_lat=True):
            if pd.isna(text): return None
            text_str = str(text).strip()
            # Pattern 0: Pure number
            try:
                val = float(text_str)
                # Reasonable bounds check for Ghana
                if is_lat and (4.0 <= val <= 12.0): return val
                if not is_lat and (-4.0 <= val <= 2.0): return val
            except: pass

            # Pattern 1: Label then Number (latitude: 5.6)
            pat1 = r"(?:latitude|lat)\s*[:\s]*([-+]?[0-9]*\.?[0-9]+)" if is_lat else r"(?:longitude|long|lon)\s*[:\s]*([-+]?[0-9]*\.?[0-9]+)"
            match1 = re.search(pat1, text_str, re.IGNORECASE)
            if match1: return float(match1.group(1))
            
            # Pattern 2: Number then Label (5.6 latitude)
            pat2 = r"([-+]?[0-9]*\.?[0-9]+)\s*(?:latitude|lat)" if is_lat else r"([-+]?[0-9]*\.?[0-9]+)\s*(?:longitude|long|lon)"
            match2 = re.search(pat2, text_str, re.IGNORECASE)
            if match2: return float(match2.group(1))
            return None

        # --- REFACTORED MINER START ---
        # 1. Start with float64 columns
        df_dataset['latitude'] = pd.to_numeric(df_dataset.get('latitude', np.nan), errors='coerce').astype('float64')
        df_dataset['longitude'] = pd.to_numeric(df_dataset.get('longitude', np.nan), errors='coerce').astype('float64')

        # 2. Mine from all columns
        for col in df_dataset.columns:
            mask = df_dataset['latitude'].isna() | (df_dataset['latitude'] == 0)
            if not mask.any(): break
            
            # 2a. Mining Coordinates from text
            # Use a temporary series to avoid SettingWithCopy and dtype warnings
            new_lats = df_dataset.loc[mask, col].apply(lambda x: extract_coord(x, is_lat=True))
            new_lons = df_dataset.loc[mask, col].apply(lambda x: extract_coord(x, is_lat=False))
            
            # Only update where we found something non-null
            found_mask = new_lats.notna()
            df_dataset.loc[new_lats[found_mask].index, 'latitude'] = new_lats[found_mask].astype(float)
            df_dataset.loc[new_lons[found_mask].index, 'longitude'] = new_lons[found_mask].astype(float)

            # 2b. RECOVERY: If still null, search for CITY NAMES in the text
            mask_reset = df_dataset['latitude'].isna() | (df_dataset['latitude'] == 0)
            if not mask_reset.any(): break
            
            def find_city_in_text(text):
                if pd.isna(text): return None, None
                t = str(text).lower()
                for city, coords in ghana_cities.items():
                    if city in t: return coords
                return None, None
            
            city_coords = df_dataset.loc[mask_reset, col].apply(find_city_in_text)
            df_dataset.loc[mask_reset, 'latitude'] = city_coords.apply(lambda x: x[0]).astype(float)
            df_dataset.loc[mask_reset, 'longitude'] = city_coords.apply(lambda x: x[1]).astype(float)

        # 3. City Fallback Table (Common Ghana Cities)
        ghana_cities = {
            "accra": (5.6037, -0.1870), "kumasi": (6.6666, -1.6163), "tamale": (9.4008, -0.8393),
            "takoradi": (4.8845, -1.7554), "tema": (5.6698, -0.0166), "cape coast": (5.1053, -1.2466),
            "sekondi": (4.9340, -1.7137), "obuasi": (6.2000, -1.6667), "koforidua": (6.0946, -0.2591),
            "wa": (10.0607, -2.5019), "sunyani": (7.3399, -2.3267), "ho": (6.6111, 0.4722),
            "bawku": (11.0616, -0.2417), "bolgatanga": (10.7856, -0.8514), "techiman": (7.5851, -1.9392),
            "asaimangas": (5.6667, -0.1667), "dansoman": (5.5500, -0.2500), "kasoa": (5.5342, -0.4244),
            "tarkwa": (5.3047, -1.9847), "ashiaman": (5.7000, -0.0333)
        }

        # City Fallback: Use address_city if latitude is still null
        if 'address_city' in df_dataset.columns:
            mask_city = df_dataset['latitude'].isna() | (df_dataset['latitude'] == 0)
            def city_to_lat(city):
                c = str(city).lower().strip()
                return ghana_cities.get(c, (None, None))[0]
            def city_to_lon(city):
                c = str(city).lower().strip()
                return ghana_cities.get(c, (None, None))[1]
            
            df_dataset.loc[mask_city, 'latitude'] = df_dataset.loc[mask_city, 'address_city'].apply(city_to_lat)
            df_dataset.loc[mask_city, 'longitude'] = df_dataset.loc[mask_city, 'address_city'].apply(city_to_lon)

        # Add Jitter to prevent perfect overlap at city centers
        mask_jitter = df_dataset['latitude'].notna()
        df_dataset.loc[mask_jitter, 'latitude'] += np.random.uniform(-0.015, 0.015, size=mask_jitter.sum())
        df_dataset.loc[mask_jitter, 'longitude'] += np.random.uniform(-0.015, 0.015, size=mask_jitter.sum())

        # Final cleanup for plotting
        ds_coords = df_dataset[['latitude', 'longitude']].dropna()
        ds_coords = ds_coords[(ds_coords['latitude'] != 0) & (ds_coords['longitude'] != 0)]
    except Exception as e:
        st.sidebar.error(f"Map Miner Error: {e}")
        ds_coords = pd.DataFrame()

    # Combine coordinates for mapping
    map_list = []
    
    # Add Original Dataset (Gray/White)
    if not ds_coords.empty:
        ds_coords = ds_coords.copy()
        ds_coords['color'] = "#808080" # Gray for original data
        map_list.append(ds_coords)

    # Add Analyzed Facilities (Green)
    if not f_all.empty:
        f_coords = f_all[['latitude', 'longitude']].dropna().copy()
        f_coords = f_coords[(f_coords['latitude'] != 0) & (f_coords['longitude'] != 0)]
        f_coords['color'] = "#28a745" 
        map_list.append(f_coords)
        
    # Add Analyzed NGOs (Blue)
    if not n_all.empty:
        n_coords = n_all[['latitude', 'longitude']].dropna().copy()
        n_coords = n_coords[(n_coords['latitude'] != 0) & (n_coords['longitude'] != 0)]
        n_coords['color'] = "#007bff"
        map_list.append(n_coords)
        
    if map_list:
        combined_map = pd.concat(map_list)
        # Explicitly cast to float64 to avoid Pandas warning
        combined_map['latitude'] = pd.to_numeric(combined_map['latitude'], errors='coerce')
        combined_map['longitude'] = pd.to_numeric(combined_map['longitude'], errors='coerce')
        st.map(combined_map, size=20, color="color", width="stretch")

    else:
        st.info("No geospatial data available. Ensure coordinates are present in the CSV (either as columns or in text descriptions).")