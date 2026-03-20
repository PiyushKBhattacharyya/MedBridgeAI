import streamlit as st
import os
import pandas as pd
import onnxruntime as ort
from src.extraction.agent import IDPExtractor
from src.extraction.loader import load_text_documents
from src.synthesis.database import MedBridgeStore
from src.synthesis.agent import SynthesisAgent
from transformers import AutoTokenizer
from optimum.onnxruntime import ORTModelForCausalLM

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
    .main {
        background-color: #0e1117;
    }
    .stButton>button {
        width: 100%;
        border-radius: 5px;
        height: 3em;
        background-color: #262730;
        color: white;
    }
    .stTextInput>div>div>input {
        background-color: #262730;
        color: white;
    }
    .success-box {
        padding: 1rem;
        border-radius: 0.5rem;
        background-color: rgba(40, 167, 69, 0.1);
        border: 1px solid rgba(40, 167, 69, 0.2);
        color: #28a745;
    }
    </style>
    """, unsafe_allow_html=True)

# Resource Loading with Caching
@st.cache_resource
def load_shared_resources():
    # Detect best provider
    available = ort.get_available_providers()
    provider = "DmlExecutionProvider" if "DmlExecutionProvider" in available else "CPUExecutionProvider"
    if "CUDAExecutionProvider" in available: 
        provider = "CUDAExecutionProvider"
        
    model_id = "Qwen/Qwen2.5-0.5B-Instruct"
    local_model_path = os.path.join("models", "qwen2.5-0.5b-onnx")
    export_needed = True
    if os.path.exists(local_model_path):
        model_id = local_model_path
        export_needed = False
        
    tokenizer = AutoTokenizer.from_pretrained(model_id)
    model = ORTModelForCausalLM.from_pretrained(
        model_id,
        export=export_needed,
        provider=provider
    )
    return model, tokenizer

# Initialize Resources
with st.spinner("Initializing Private Medical Intelligence..."):
    shared_model, shared_tokenizer = load_shared_resources()

if 'store' not in st.session_state:
    st.session_state.store = MedBridgeStore()
if 'extractor' not in st.session_state:
    st.session_state.extractor = IDPExtractor(model=shared_model, tokenizer=shared_tokenizer)
if 'agent' not in st.session_state:
    st.session_state.agent = SynthesisAgent(st.session_state.store, model=shared_model, tokenizer=shared_tokenizer)
if 'chat_history' not in st.session_state:
    st.session_state.chat_history = []

# Sidebar
with st.sidebar:
    st.title("🏥 MedBridge AI")
    st.subheader("Hardware Acceleration")
    
    available_providers = ort.get_available_providers()
    
    if "CUDAExecutionProvider" in available_providers:
        st.success("✓ NVIDIA GPU (CUDA) Active")
    elif "DmlExecutionProvider" in available_providers:
        st.success("✓ AMD GPU (DirectML) Active")
    else:
        st.warning("⚠ Using CPU (AMD/Intel Optimized)")
    
    with st.expander("System Details"):
        st.write(f"Active Backend: {available_providers[0] if available_providers else 'CPU'}")
        st.write(f"Available: {', '.join(available_providers)}")
    
    st.divider()
    if st.button("Clear Chat History"):
        st.session_state.chat_history = []
        st.rerun()

# Main UI Tabs
tab1, tab2, tab3 = st.tabs(["📄 Document Analysis", "🧠 Medical Assistant", "📊 Database View"])

# --- Tab 1: Document Analysis ---
with tab1:
    st.header("Upload Medical Documents")
    st.write("Extract structured facility and NGO data using local SLM intelligence.")
    
    uploaded_file = st.file_uploader("Choose a Markdown or Text file", type=['md', 'txt'])
    
    if uploaded_file is not None:
        # Save temp file
        temp_path = os.path.join("data", "temp_upload.md")
        os.makedirs("data", exist_ok=True)
        with open(temp_path, "wb") as f:
            f.write(uploaded_file.getbuffer())
        
        if st.button("Analyze & Store in Database"):
            with st.spinner("Analyzing document with GPU-accelerated SLM..."):
                docs = load_text_documents(temp_path)
                if docs:
                    extraction = st.session_state.extractor.extract_from_text(docs[0].page_content)
                    
                    st.success(f"✓ Extraction Complete! Found {len(extraction.facilities)} facilities and {len(extraction.ngos)} NGOs.")
                    
                    # Store with source tracking
                    st.session_state.store.add_extractions(extraction, source_doc=uploaded_file.name)
                    st.toast(f"Data from {uploaded_file.name} persisted to LanceDB", icon="💾")
                    
                    # Display JSON result
                    with st.expander("Show Extracted JSON"):
                        st.json(extraction.model_dump())
                    
                    # Clean up
                    os.remove(temp_path)

# --- Tab 2: Medical Assistant ---
with tab2:
    st.header("Intelligence Assistant")
    st.write("Synthesize insights from your local database using private vector search.")
    
    # Display Chat History
    for message in st.session_state.chat_history:
        with st.chat_message(message["role"]):
            st.markdown(message["content"])
            
    # Chat Input
    if prompt := st.chat_input("Ask a question about facility capabilities or NGO missions..."):
        st.session_state.chat_history.append({"role": "user", "content": prompt})
        with st.chat_message("user"):
            st.markdown(prompt)
            
        with st.chat_message("assistant"):
            with st.spinner("Synthesizing answer from local intelligence..."):
                response = st.session_state.agent.answer_question(prompt)
                st.markdown(response)
                st.session_state.chat_history.append({"role": "assistant", "content": response})

# --- Tab 3: Database View & Mapping ---
with tab3:
    st.header("Stored Medical Data & Mapping")
    
    # Combined Data for Map
    facilities_df = st.session_state.store.search_facilities("", limit=500)
    ngos_df = st.session_state.store.search_ngos("", limit=500)
    
    map_data = []
    if not facilities_df.empty:
        f_map = facilities_df[['name', 'latitude', 'longitude']].dropna(subset=['latitude', 'longitude'])
        f_map['type'] = 'Facility'
        map_data.append(f_map)
    if not ngos_df.empty:
        n_map = ngos_df[['name', 'latitude', 'longitude']].dropna(subset=['latitude', 'longitude'])
        n_map['type'] = 'NGO'
        map_data.append(n_map)
        
    if map_data:
        full_map_df = pd.concat(map_data)
        st.subheader("Medical Resource Map")
        st.map(full_map_df)
    else:
        st.info("Add facilities with coordinates to see them on the map.")

    def format_list_column(val):
        """Helper to format lists into clean, comma-separated strings for UI display."""
        if isinstance(val, list):
            if not val: return ""
            return ", ".join(map(str, val))
        if isinstance(val, str) and val.startswith("[") and val.endswith("]"):
            try:
                import ast
                parsed = ast.literal_eval(val)
                if isinstance(parsed, list):
                    return ", ".join(map(str, parsed))
            except:
                pass
        return val

    st.divider()
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.subheader("Facilities")
        if not facilities_df.empty:
            # Fix numeric types for Arrow before display
            f_display = facilities_df.drop(columns=['vector'], errors='ignore')
            num_cols = ['latitude', 'longitude', 'yearEstablished', 'numberDoctors', 'capacity', 'area']
            for c in num_cols:
                if c in f_display.columns: f_display[c] = pd.to_numeric(f_display[c], errors='coerce')
            
            # Clean up list columns for display
            list_cols = ['capability', 'equipment', 'procedure', 'phone_numbers', 'websites']
            for col in list_cols:
                if col in f_display.columns:
                    f_display[col] = f_display[col].apply(format_list_column)
            
            # Fill only object/string columns
            obj_cols = f_display.select_dtypes(include=['object']).columns
            f_display[obj_cols] = f_display[obj_cols].fillna("")
            
            st.dataframe(f_display, hide_index=True, width="stretch")
            
            # Full Download Option
            full_f = st.session_state.store.get_all_facilities()
            if not full_f.empty:
                full_f = full_f.drop(columns=['vector'], errors='ignore')
                for col in list_cols:
                    if col in full_f.columns:
                        full_f[col] = full_f[col].apply(format_list_column)
                csv_f = full_f.to_csv(index=False).encode('utf-8')
                st.download_button(
                    label="Download FULL Facilities CSV (900+ records)",
                    data=csv_f,
                    file_name="medical_facilities_full.csv",
                    mime="text/csv",
                    help="Downloads the entire database table, bypassing the 500-row dashboard limit."
                )
        else:
            st.info("No facilities stored yet.")

            
    with col2:
        st.subheader("NGOs")
        if not ngos_df.empty:
            # Fix numeric types for Arrow before display
            n_display = ngos_df.drop(columns=['vector'], errors='ignore')
            for c in ['latitude', 'longitude', 'yearEstablished']:
                if c in n_display.columns: n_display[c] = pd.to_numeric(n_display[c], errors='coerce')
            
            # Fill only object/string columns
            obj_cols = n_display.select_dtypes(include=['object']).columns
            n_display[obj_cols] = n_display[obj_cols].fillna("")
            
            st.dataframe(n_display, hide_index=True, width="stretch")
            
            # Full Download Option
            full_n = st.session_state.store.get_all_ngos()
            if not full_n.empty:
                full_n = full_n.drop(columns=['vector'], errors='ignore')
                # Format URL lists if present
                for col in ['phone_numbers', 'websites']:
                    if col in full_n.columns:
                        full_n[col] = full_n[col].apply(format_list_column)
                csv_n = full_n.to_csv(index=False).encode('utf-8')
                st.download_button(
                    label="Download FULL NGOs CSV",
                    data=csv_n,
                    file_name="medical_ngos_full.csv",
                    mime="text/csv",
                    help="Downloads the entire NGO database table."
                )
        else:
            st.info("No NGOs stored yet.")


st.divider()
st.caption("Virtue Foundation x MedBridgeAI | Local-First Medical Intelligence")
