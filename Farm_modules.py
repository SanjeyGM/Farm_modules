import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import json
import os
from datetime import datetime
from dotenv import load_dotenv
import plotly.express as px
import plotly.graph_objects as go
import fitz  # PyMuPDF
import hashlib
import time
import google.generativeai as genai
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import re
import pickle

# Load environment variables
load_dotenv()

# Set API key (with fallback for demo mode)
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY", "demo_mode")

# Configure Gemini API (when not in demo mode)
if GEMINI_API_KEY != "demo_mode":
    genai.configure(api_key=GEMINI_API_KEY)

# Global variables
IDEAL_THRESHOLDS = {
    "ph": 6,
    "do": 5
}

# Initialize session state
if "messages" not in st.session_state:
    st.session_state.messages = []

if "manual_text" not in st.session_state:
    st.session_state.manual_text = ""
    st.session_state.manual_hash = ""
    st.session_state.manual_processed = False
    st.session_state.chunks = []
    st.session_state.vectorizer = None
    st.session_state.chunk_vectors = None

if "sensor_data" not in st.session_state:
    st.session_state.sensor_data = []

# Create cache directory if it doesn't exist
os.makedirs("cache", exist_ok=True)
os.makedirs("vector_cache", exist_ok=True)

def chunk_text(text, chunk_size=500, overlap=100):
    """Split text into chunks with overlap for better context preservation"""
    # Clean text and normalize whitespace
    text = re.sub(r'\s+', ' ', text).strip()
    
    # Split by sentences to avoid cutting in the middle of a sentence
    sentences = re.split(r'(?<=[.!?])\s+', text)
    
    chunks = []
    current_chunk = []
    current_size = 0
    
    for sentence in sentences:
        sentence_words = len(sentence.split())
        
        if current_size + sentence_words <= chunk_size:
            current_chunk.append(sentence)
            current_size += sentence_words
        else:
            # Save current chunk if not empty
            if current_chunk:
                chunks.append(' '.join(current_chunk))
            
            # Start new chunk with this sentence
            current_chunk = [sentence]
            current_size = sentence_words
    
    # Add the last chunk if not empty
    if current_chunk:
        chunks.append(' '.join(current_chunk))
    
    return chunks

def initialize_vector_store(chunks):
    """Create TF-IDF vectors for chunks"""
    # Initialize the vectorizer
    vectorizer = TfidfVectorizer(stop_words='english')
    
    # Create vectors
    chunk_vectors = vectorizer.fit_transform(chunks)
    
    return vectorizer, chunk_vectors

def get_relevant_chunks(query, vectorizer, chunk_vectors, chunks, top_k=3):
    """Retrieve most relevant chunks using TF-IDF and cosine similarity"""
    # Vectorize the query
    query_vector = vectorizer.transform([query])
    
    # Calculate similarity scores
    # Ensure chunk_vectors is a scipy sparse matrix
    if not isinstance(chunk_vectors, type(query_vector)):
        st.error(f"Type mismatch: query_vector is {type(query_vector)} but chunk_vectors is {type(chunk_vectors)}")
        # Try to convert chunk_vectors to a sparse matrix if it's a numpy array
        from scipy import sparse
        if isinstance(chunk_vectors, np.ndarray):
            chunk_vectors = sparse.csr_matrix(chunk_vectors)
    
    similarities = cosine_similarity(query_vector, chunk_vectors).flatten()
    
    # Get indices of top K most similar chunks
    top_indices = similarities.argsort()[-top_k:][::-1]
    
    # Get the relevant chunks and their similarity scores
    relevant_chunks = [(chunks[i], similarities[i]) for i in top_indices if similarities[i] > 0.1]
    
    # Sort by similarity score (descending)
    relevant_chunks.sort(key=lambda x: x[1], reverse=True)
    
    # Return just the chunks (without scores)
    return [chunk for chunk, score in relevant_chunks]

def extract_text_from_pdf(pdf_file):
    """Extract text from uploaded PDF file using PyMuPDF (faster than OCR)"""
    # Calculate file hash for caching
    pdf_bytes = pdf_file.getvalue()
    file_hash = hashlib.md5(pdf_bytes).hexdigest()
    cache_file = f"cache/{file_hash}.txt"
    vector_cache_file = f"vector_cache/{file_hash}.pkl"
    chunks_cache_file = f"vector_cache/{file_hash}_chunks.json"
    
    # Check if we've already processed this file
    if os.path.exists(cache_file):
        with open(cache_file, 'r', encoding='utf-8') as f:
            text = f.read()
        
        # Also load the vectorized chunks if they exist
        if os.path.exists(vector_cache_file) and os.path.exists(chunks_cache_file):
            # Load chunks
            with open(chunks_cache_file, 'r', encoding='utf-8') as f:
                st.session_state.chunks = json.load(f)
            
            # Load vectorizer and vectors using pickle
            with open(vector_cache_file, 'rb') as f:
                cached_data = pickle.load(f)
                st.session_state.vectorizer = cached_data['vectorizer']
                st.session_state.chunk_vectors = cached_data['vectors']
            
            return text, file_hash
    
    # Process the PDF if not cached
    text = ""
    with fitz.open(stream=pdf_bytes, filetype="pdf") as doc:
        for page in doc:
            text += page.get_text()
    
    # Save text to cache
    with open(cache_file, 'w', encoding='utf-8') as f:
        f.write(text)
    
    # Create chunks
    chunks = chunk_text(text)
    st.session_state.chunks = chunks
    
    # Save chunks for caching
    with open(chunks_cache_file, 'w', encoding='utf-8') as f:
        json.dump(chunks, f)
    
    # Create vector store
    vectorizer, chunk_vectors = initialize_vector_store(chunks)
    st.session_state.vectorizer = vectorizer
    st.session_state.chunk_vectors = chunk_vectors
    
    # Save vectors and vectorizer using pickle for proper serialization
    with open(vector_cache_file, 'wb') as f:
        pickle.dump({
            'vectorizer': vectorizer,
            'vectors': chunk_vectors
        }, f)
    
    return text, file_hash

def load_sample_data():
    """Load sample sensor data"""
    return [
        {
            "datetime": "2025-01-01 08:59:31",
            "ph_fish_tank": 6,
            "ph_biofilter": 6.3,
            "do_fish_tank": 5,
            "do_biofilter": 5.3
        },
        {
            "datetime": "2019-01-01 20:59:31",
            "ph_fish_tank": 6,
            "ph_biofilter": 6.3,
            "do_fish_tank": 5,
            "do_biofilter": 5.3
        },
        {
            "datetime": "2019-01-02 08:59:31",
            "ph_fish_tank": 5.6,
            "ph_biofilter": 5.8,
            "do_fish_tank": 4,
            "do_biofilter": 4.2
        },
        {
            "datetime": "2019-01-01 20:59:31",
            "ph_fish_tank": 6.2,
            "ph_biofilter": 6.2,
            "do_fish_tank": 3.6,
            "do_biofilter": 3.9
        }
    ]

def analyze_sensor_data(data, thresholds):
    """Analyze sensor data and return insights"""
    df = pd.DataFrame(data)
    
    # Convert datetime strings to datetime objects
    df['datetime'] = pd.to_datetime(df['datetime'])
    df = df.sort_values('datetime')
    
    # Identify data points below thresholds
    issues = []
    
    if df['ph_fish_tank'].min() < thresholds['ph']:
        issues.append({
            "parameter": "pH in fish tank",
            "value": df['ph_fish_tank'].min(),
            "threshold": thresholds['ph'],
            "datetime": df.loc[df['ph_fish_tank'].idxmin(), 'datetime'].strftime("%Y-%m-%d %H:%M:%S")
        })
    
    if df['ph_biofilter'].min() < thresholds['ph']:
        issues.append({
            "parameter": "pH in biofilter",
            "value": df['ph_biofilter'].min(),
            "threshold": thresholds['ph'],
            "datetime": df.loc[df['ph_biofilter'].idxmin(), 'datetime'].strftime("%Y-%m-%d %H:%M:%S")
        })
    
    if df['do_fish_tank'].min() < thresholds['do']:
        issues.append({
            "parameter": "Dissolved oxygen in fish tank",
            "value": df['do_fish_tank'].min(),
            "threshold": thresholds['do'],
            "datetime": df.loc[df['do_fish_tank'].idxmin(), 'datetime'].strftime("%Y-%m-%d %H:%M:%S")
        })
    
    if df['do_biofilter'].min() < thresholds['do']:
        issues.append({
            "parameter": "Dissolved oxygen in biofilter",
            "value": df['do_biofilter'].min(),
            "threshold": thresholds['do'],
            "datetime": df.loc[df['do_biofilter'].idxmin(), 'datetime'].strftime("%Y-%m-%d %H:%M:%S")
        })
    
    return {
        "data_frame": df,
        "issues": issues
    }

def get_ai_advice(prompt, demo_mode=False):
    """Get advice from Gemini API based on prompt and relevant manual chunks"""
    if demo_mode or GEMINI_API_KEY == "demo_mode":
        # In demo mode, generate synthetic responses
        st.session_state.demo_thinking = True
        progress_bar = st.progress(0)
        
        # Simulate AI thinking
        for i in range(100):
            time.sleep(0.01)
            progress_bar.progress(i + 1)
        
        # Generate canned responses based on keywords in the prompt
        if "pH" in prompt and "low" in prompt:
            return """
            # pH Level Management
            
            Based on your sensor data, I notice that pH levels have dropped below the recommended threshold in both your fish tank and biofilter. 
            
            ## Immediate Actions:
            
            1. **Check your buffering system** - The operations manual recommends adding calcium carbonate at 5g per 100L when pH drops below 6.0.
            
            2. **Reduce feeding temporarily** - Overfeeding can increase acidity through waste decomposition.
            
            3. **Verify water source pH** - Recent changes in source water can contribute to system-wide pH drops.
            
            The manual specifically warns against sudden pH corrections. Aim to increase by no more than 0.2 units per day to avoid shocking fish.
            
            Would you like me to provide the exact procedures for pH adjustment from the operations manual?
            """
        elif "oxygen" in prompt or "DO" in prompt:
            return """
            # Dissolved Oxygen Management
            
            I see your DO levels have fallen below recommended thresholds. According to your operations manual, this requires immediate attention.
            
            ## Recommended Actions:
            
            1. **Increase aeration immediately** - The manual recommends adding an emergency air stone in the affected areas.
            
            2. **Reduce feeding by 50%** for the next 24 hours to reduce oxygen demand.
            
            3. **Check for biofilter clogging** - Section 4.3 of your manual indicates that reduced flow through biofilters often causes DO drops.
            
            4. **Verify water temperature** - The manual indicates that DO levels naturally decrease with rising temperatures.
            
            The operations manual specifically recommends that DO levels should never remain below 4mg/L for more than 4 hours. Continue monitoring closely.
            
            Would you like specific instructions for emergency aeration procedures?
            """
        else:
            return """
            # Aquafarming System Maintenance
            
            Based on your latest readings and the information in your operations manual, here are some recommendations:
            
            ## Regular Maintenance Tasks:
            
            1. **Monitor pH and DO levels twice daily** - The operations manual emphasizes this as critical for early issue detection.
            
            2. **Check all mechanical components** - Ensure pumps, aerators, and filters are functioning at optimal levels.
            
            3. **Maintain consistent feeding schedules** - The manual highlights this as key for stable water parameters.
            
            4. **Regular biofilter backwashing** - Section 5.2 of your manual recommends this weekly to prevent clogging.
            
            Your system parameters are currently within acceptable ranges, but continued monitoring is essential. The operations manual specifically mentions that maintaining stable conditions is more important than achieving "perfect" readings.
            
            Would you like me to provide more detailed maintenance procedures based on your specific system configuration?
            """
    else:
        try:
            # Get relevant chunks from the manual if available
            if st.session_state.vectorizer is None or not st.session_state.chunks:
                return "Error: No manual has been processed. Please upload an operations manual PDF first."
            
            # Get relevant chunks for context
            relevant_chunks = get_relevant_chunks(
                prompt, 
                st.session_state.vectorizer, 
                st.session_state.chunk_vectors, 
                st.session_state.chunks,
                top_k=3
            )
            
            # Join chunks with section markers
            context_text = "\n\n--- Section ---\n\n".join(relevant_chunks)
            
            # Create the prompt for Gemini
            system_prompt = f"""
            You are an aquafarming expert assistant. Use the following information from the operations manual 
            to provide specific, actionable advice to farmers.
            
            Operations Manual Information:
            {context_text}
            
            Analyze the sensor data trends and issues carefully. Provide specific recommendations to address
            any problems identified. Focus on practical solutions that the farmer can implement immediately.
            """
            
            full_prompt = system_prompt + "\n\n" + prompt
            
            # Create a Gemini model instance
            model = genai.GenerativeModel('gemini-1.5-pro')
            
            # Generate response
            response = model.generate_content(full_prompt)
            
            return response.text
        except Exception as e:
            return f"Error getting advice: {str(e)}\n\nPlease check your API key configuration."

def create_visualization(df):
    """Create visualizations for sensor data"""
    # Create time series plots
    fig = go.Figure()
    
    # Add pH traces
    fig.add_trace(go.Scatter(
        x=df['datetime'],
        y=df['ph_fish_tank'],
        name='pH Fish Tank',
        line=dict(color='blue')
    ))
    
    fig.add_trace(go.Scatter(
        x=df['datetime'],
        y=df['ph_biofilter'],
        name='pH Biofilter',
        line=dict(color='lightblue')
    ))
    
    # Add horizontal line for pH threshold
    fig.add_trace(go.Scatter(
        x=[df['datetime'].min(), df['datetime'].max()],
        y=[IDEAL_THRESHOLDS['ph'], IDEAL_THRESHOLDS['ph']],
        name='pH Threshold',
        line=dict(color='red', dash='dash')
    ))
    
    fig.update_layout(
        title='pH Levels Over Time',
        xaxis_title='Date & Time',
        yaxis_title='pH Level',
        height=400
    )
    
    # Create DO plot
    fig2 = go.Figure()
    
    fig2.add_trace(go.Scatter(
        x=df['datetime'],
        y=df['do_fish_tank'],
        name='DO Fish Tank',
        line=dict(color='green')
    ))
    
    fig2.add_trace(go.Scatter(
        x=df['datetime'],
        y=df['do_biofilter'],
        name='DO Biofilter',
        line=dict(color='lightgreen')
    ))
    
    # Add horizontal line for DO threshold
    fig2.add_trace(go.Scatter(
        x=[df['datetime'].min(), df['datetime'].max()],
        y=[IDEAL_THRESHOLDS['do'], IDEAL_THRESHOLDS['do']],
        name='DO Threshold',
        line=dict(color='red', dash='dash')
    ))
    
    fig2.update_layout(
        title='Dissolved Oxygen Levels Over Time',
        xaxis_title='Date & Time',
        yaxis_title='Dissolved Oxygen (mg/L)',
        height=400
    )
    
    return fig, fig2

# Main application
def main():
    st.set_page_config(page_title="Aquafarming Advisor", page_icon="🐟", layout="wide")
    
    st.title("🐟 Aquafarming Advisor")
    
    # Show demo mode notice if applicable
    if GEMINI_API_KEY == "demo_mode":
        st.warning("⚠️ Running in DEMO MODE - AI responses are simulated. Add your Gemini API key to .env file for full functionality.")
    
    st.markdown("""
    This application helps aquafarmers monitor their systems and get expert advice based on sensor data 
    and farming best practices from the operations manual.
    """)
    
    # Create sidebar
    with st.sidebar:
        st.header("Setup")
        
        # Upload operations manual
        st.subheader("1. Upload Operations Manual")
        uploaded_file = st.file_uploader("Upload your operations manual (PDF)", type="pdf")
        
        if uploaded_file:
            # Extract text from the PDF
            with st.spinner("Processing operations manual and building search index..."):
                manual_text, file_hash = extract_text_from_pdf(uploaded_file)
                
                # Only update if it's a new file
                if file_hash != st.session_state.manual_hash:
                    st.session_state.manual_text = manual_text
                    st.session_state.manual_hash = file_hash
                    st.session_state.manual_processed = True
                    
            st.success(f"✅ Uploaded and indexed: {uploaded_file.name}")
            
            # Show chunk preview
            with st.expander("Preview Indexed Chunks"):
                for i, chunk in enumerate(st.session_state.chunks[:3]):
                    st.markdown(f"**Chunk {i+1}:**")
                    st.text(chunk[:200] + "...")
                
                if len(st.session_state.chunks) > 3:
                    st.text(f"... and {len(st.session_state.chunks) - 3} more chunks")
        
        # Load sample data or upload custom data
        st.subheader("2. Sensor Data")
        data_option = st.radio("Choose data source:", ["Use sample data", "Upload custom data"])
        
        if data_option == "Use sample data":
            if st.button("Load Sample Data"):
                st.session_state.sensor_data = load_sample_data()
                st.success("Sample data loaded!")
        else:
            uploaded_data = st.file_uploader("Upload JSON sensor data", type="json")
            if uploaded_data:
                try:
                    st.session_state.sensor_data = json.load(uploaded_data)
                    st.success("Custom data loaded successfully!")
                except Exception as e:
                    st.error(f"Error loading data: {str(e)}")
        
        # Set or adjust ideal thresholds
        st.subheader("3. Adjust Ideal Thresholds")
        ph_threshold = st.slider("pH Threshold", 5.0, 8.0, float(IDEAL_THRESHOLDS["ph"]), 0.1)
        do_threshold = st.slider("Dissolved Oxygen Threshold (mg/L)", 2.0, 10.0, float(IDEAL_THRESHOLDS["do"]), 0.1)
        
        # Update global thresholds
        IDEAL_THRESHOLDS["ph"] = ph_threshold
        IDEAL_THRESHOLDS["do"] = do_threshold
        
        # Show API Key Input
        st.subheader("4. API Configuration")
        new_api_key = st.text_input("Gemini API Key", value="", type="password")
        if new_api_key:
            os.environ["GEMINI_API_KEY"] = new_api_key
            genai.configure(api_key=new_api_key)
            st.success("API Key updated! Refresh for full functionality.")
        
        # Show demo mode info
        if GEMINI_API_KEY == "demo_mode":
            st.info("📝 In demo mode, AI responses are pre-written examples. The application will work fully when a Gemini API key is provided.")
    
    # Main content - tabs
    tab1, tab2, tab3 = st.tabs(["Dashboard", "Chat with Advisor", "Raw Data"])
    
    # Tab 1: Dashboard
    with tab1:
        st.header("Aquafarming System Dashboard")
        
        if not st.session_state.sensor_data:
            st.info("Please load sensor data from the sidebar to view the dashboard.")
        else:
            # Analyze sensor data
            analysis = analyze_sensor_data(st.session_state.sensor_data, IDEAL_THRESHOLDS)
            df = analysis["data_frame"]
            issues = analysis["issues"]
            
            # Display current status
            st.subheader("Current System Status")
            
            # Create columns for latest readings
            latest = df.iloc[-1]
            col1, col2, col3, col4 = st.columns(4)
            
            with col1:
                ph_ft = latest["ph_fish_tank"]
                st.metric(
                    "pH Fish Tank", 
                    f"{ph_ft:.1f}", 
                    f"{ph_ft - IDEAL_THRESHOLDS['ph']:.1f}",
                    delta_color="inverse" if ph_ft < IDEAL_THRESHOLDS['ph'] else "normal"
                )
            
            with col2:
                ph_bf = latest["ph_biofilter"]
                st.metric(
                    "pH Biofilter", 
                    f"{ph_bf:.1f}", 
                    f"{ph_bf - IDEAL_THRESHOLDS['ph']:.1f}",
                    delta_color="inverse" if ph_bf < IDEAL_THRESHOLDS['ph'] else "normal"
                )
            
            with col3:
                do_ft = latest["do_fish_tank"]
                st.metric(
                    "DO Fish Tank (mg/L)", 
                    f"{do_ft:.1f}", 
                    f"{do_ft - IDEAL_THRESHOLDS['do']:.1f}",
                    delta_color="inverse" if do_ft < IDEAL_THRESHOLDS['do'] else "normal"
                )
            
            with col4:
                do_bf = latest["do_biofilter"]
                st.metric(
                    "DO Biofilter (mg/L)", 
                    f"{do_bf:.1f}", 
                    f"{do_bf - IDEAL_THRESHOLDS['do']:.1f}",
                    delta_color="inverse" if do_bf < IDEAL_THRESHOLDS['do'] else "normal"
                )
            
            # Create visualizations
            st.subheader("Sensor Data Trends")
            fig1, fig2 = create_visualization(df)
            st.plotly_chart(fig1, use_container_width=True)
            st.plotly_chart(fig2, use_container_width=True)
            
            # Display issues
            st.subheader("Identified Issues")
            if issues:
                for i, issue in enumerate(issues):
                    st.warning(
                        f"{issue['parameter']} dropped to {issue['value']:.1f} " 
                        f"(below threshold of {issue['threshold']:.1f}) on {issue['datetime']}"
                    )
            else:
                st.success("No issues detected! All parameters are within ideal thresholds.")
            
            # Get AI recommendation if manual is uploaded
            if st.session_state.manual_processed:
                st.subheader("AI Recommendations")
                
                # Create prompt based on analysis
                if issues:
                    issues_text = "\n".join([
                        f"- {issue['parameter']} dropped to {issue['value']:.1f} " 
                        f"(below threshold of {issue['threshold']:.1f}) on {issue['datetime']}"
                        for issue in issues
                    ])
                    
                    prompt = f"""
                    Based on the sensor data analysis, the following issues were detected:
                    
                    {issues_text}
                    
                    Current readings:
                    - pH Fish Tank: {latest["ph_fish_tank"]:.1f}
                    - pH Biofilter: {latest["ph_biofilter"]:.1f}
                    - Dissolved Oxygen Fish Tank: {latest["do_fish_tank"]:.1f} mg/L
                    - Dissolved Oxygen Biofilter: {latest["do_biofilter"]:.1f} mg/L
                    
                    Please provide specific advice to address these issues based on the operations manual.
                    """
                else:
                    prompt = f"""
                    All sensor readings are currently within ideal thresholds:
                    
                    - pH Fish Tank: {latest["ph_fish_tank"]:.1f} (threshold: {IDEAL_THRESHOLDS['ph']:.1f})
                    - pH Biofilter: {latest["ph_biofilter"]:.1f} (threshold: {IDEAL_THRESHOLDS['ph']:.1f})
                    - Dissolved Oxygen Fish Tank: {latest["do_fish_tank"]:.1f} mg/L (threshold: {IDEAL_THRESHOLDS['do']:.1f} mg/L)
                    - Dissolved Oxygen Biofilter: {latest["do_biofilter"]:.1f} mg/L (threshold: {IDEAL_THRESHOLDS['do']:.1f} mg/L)
                    
                    Please provide maintenance advice and best practices for maintaining optimal conditions.
                    """
                
                with st.spinner("Generating recommendations..."):
                    advice = get_ai_advice(prompt, GEMINI_API_KEY == "demo_mode")
                    st.markdown(advice)
            else:
                st.info("Upload an operations manual to get AI-powered recommendations based on your sensor data.")

    # Tab 2: Chat Interface
    with tab2:
        st.header("Chat with the Aquafarming Advisor")
        
        if not st.session_state.manual_processed:
            st.info("Please upload your operations manual first to enable the chat advisor.")
        else:
            # Show search statistics if available
            if st.session_state.chunks:
                st.success(f"✅ Knowledge base ready! The system indexed {len(st.session_state.chunks)} sections from your manual.")
            
            # Display chat messages
            for message in st.session_state.messages:
                with st.chat_message(message["role"]):
                    st.markdown(message["content"])
            
            # Get user input
            user_input = st.chat_input("Ask a question about your aquafarming system...")
            
            if user_input:
                # Add user message to chat history
                st.session_state.messages.append({"role": "user", "content": user_input})
                
                # Display user message
                with st.chat_message("user"):
                    st.markdown(user_input)
                
                # Generate context for AI
                sensor_context = ""
                if st.session_state.sensor_data:
                    analysis = analyze_sensor_data(st.session_state.sensor_data, IDEAL_THRESHOLDS)
                    df = analysis["data_frame"]
                    latest = df.iloc[-1]
                    
                    sensor_context = f"""
                    Current sensor readings:
                    - pH Fish Tank: {latest["ph_fish_tank"]:.1f} (threshold: {IDEAL_THRESHOLDS['ph']:.1f})
                    - pH Biofilter: {latest["ph_biofilter"]:.1f} (threshold: {IDEAL_THRESHOLDS['ph']:.1f})
                    - Dissolved Oxygen Fish Tank: {latest["do_fish_tank"]:.1f} mg/L (threshold: {IDEAL_THRESHOLDS['do']:.1f} mg/L)
                    - Dissolved Oxygen Biofilter: {latest["do_biofilter"]:.1f} mg/L (threshold: {IDEAL_THRESHOLDS['do']:.1f} mg/L)
                    """
                
                # Create prompt with user question and context
                prompt = f"""
                User question: {user_input}
                
                {sensor_context}
                
                Please provide a helpful and informative response based on the operations manual and best aquafarming practices.
                Reference specific sections from the manual when appropriate.
                """
                
                # Get relevant chunks and generate response
                try:
                    if st.session_state.vectorizer is not None and st.session_state.chunks:
                        with st.spinner("Searching for relevant information..."):
                            relevant_chunks = get_relevant_chunks(
                                user_input,
                                st.session_state.vectorizer,
                                st.session_state.chunk_vectors,
                                st.session_state.chunks,
                                top_k=3
                            )
                            
                            # Show relevant sections in expandable section
                            with st.expander("Relevant sections from the manual", expanded=False):
                                for i, chunk in enumerate(relevant_chunks):
                                    st.markdown(f"**Section {i+1}:**")
                                    st.text(chunk[:300] + "..." if len(chunk) > 300 else chunk)
                            
                            # Generate AI response
                            with st.chat_message("assistant"):
                                with st.spinner("Generating response..."):
                                    response = get_ai_advice(prompt, GEMINI_API_KEY == "demo_mode")
                                    st.markdown(response)
                            
                            # Add assistant response to chat history
                            st.session_state.messages.append({"role": "assistant", "content": response})
                    
                except Exception as e:
                    st.error(f"Error processing request: {str(e)}")

    # Tab 3: Raw Data View
    with tab3:
        st.header("Raw Sensor Data")
        
        if not st.session_state.sensor_data:
            st.info("Please load sensor data from the sidebar to view the raw data.")
        else:
            # Show raw data table
            df = pd.DataFrame(st.session_state.sensor_data)
            
            # Format datetime and create display copy
            if 'datetime' in df.columns:
                df['datetime'] = pd.to_datetime(df['datetime'])
                display_df = df.copy()
                display_df['datetime'] = display_df['datetime'].dt.strftime('%Y-%m-%d %H:%M:%S')
                # Sort by datetime
                display_df = display_df.sort_values('datetime')
            
            # Display the formatted dataframe
            st.dataframe(display_df, use_container_width=True)
            
            # Option to download data as CSV
            csv = df.to_csv(index=False).encode('utf-8')
            st.download_button(
                label="Download data as CSV",
                data=csv,
                file_name="aquafarming_sensor_data.csv",
                mime="text/csv",
            )
            
            # Summary statistics
            st.subheader("Summary Statistics")
            st.dataframe(df.describe(), use_container_width=True)
            
            # Add manual data entry form
            st.subheader("Add New Sensor Reading")
            with st.form("new_reading_form"):
                col1, col2 = st.columns(2)
                
                with col1:
                    new_datetime = st.date_input("Date", datetime.now())
                    new_time = st.time_input("Time", datetime.now().time())
                    
                    new_ph_ft = st.number_input("pH Fish Tank", 0.0, 14.0, 7.0, 0.1)
                    new_ph_bf = st.number_input("pH Biofilter", 0.0, 14.0, 7.0, 0.1)
                
                with col2:
                    new_do_ft = st.number_input("DO Fish Tank (mg/L)", 0.0, 20.0, 5.0, 0.1)
                    new_do_bf = st.number_input("DO Biofilter (mg/L)", 0.0, 20.0, 5.0, 0.1)
                
                submit_button = st.form_submit_button("Add Reading")
                
                if submit_button:
                    # Combine date and time
                    combined_datetime = datetime.combine(new_datetime, new_time)
                    datetime_str = combined_datetime.strftime("%Y-%m-%d %H:%M:%S")
                    
                    # Create new reading
                    new_reading = {
                        "datetime": datetime_str,
                        "ph_fish_tank": new_ph_ft,
                        "ph_biofilter": new_ph_bf,
                        "do_fish_tank": new_do_ft,
                        "do_biofilter": new_do_bf
                    }
                    
                    # Add to session state
                    st.session_state.sensor_data.append(new_reading)
                    st.success("New reading added successfully!")
                    st.experimental_rerun()

if __name__ == "__main__":
    main()