import streamlit as st
import torch
from indic_transliteration.sanscript import transliterate, DEVANAGARI, ITRANS
from parler_tts import ParlerTTSForConditionalGeneration
from transformers import AutoTokenizer
import soundfile as sf
import numpy as np
import os

# Page configuration
st.set_page_config(
    page_title="Text-to-Speech Generator",
    page_icon="🗣️",
    layout="wide"
)

# Custom CSS for better styling
st.markdown("""
    <style>
    .main {
        padding: 2rem;
    }
    .stButton>button {
        width: 100%;
        margin-top: 1rem;
    }
    .success-message {
        padding: 1rem;
        border-radius: 0.5rem;
        background-color: #d4edda;
        color: #155724;
        margin: 1rem 0;
    }
    </style>
""", unsafe_allow_html=True)

# Title and description
st.title("🗣️Text-to-Speech Generator")
st.markdown("""
    Convert Hindi text (Devanagari script) to natural-sounding speech using advanced AI technology.
    This application is specifically trained for Indian male accent.
""")

# Initialize session state for model loading
if 'model' not in st.session_state:
    st.session_state.model = None
if 'tokenizer' not in st.session_state:
    st.session_state.tokenizer = None

def load_models():
    """Load the TTS model and tokenizer"""
    with st.spinner("Loading models... This might take a few moments."):
        device = "cuda:0" if torch.cuda.is_available() else "cpu"
        model = ParlerTTSForConditionalGeneration.from_pretrained(
            "En1gma02/Parler-TTS-Mini-v0.1-Indian-Male-Accent-Hindi-Kaggle"
        ).to(device)
        tokenizer = AutoTokenizer.from_pretrained("parler-tts/parler_tts_mini_v0.1")
        st.session_state.model = model
        st.session_state.tokenizer = tokenizer
        st.success("✅ Models loaded successfully!")

def generate_audio(text, description, model, tokenizer):
    """Generate audio from input text"""
    device = "cuda:0" if torch.cuda.is_available() else "cpu"
    
    # Transliterate Devanagari to ITRANS
    prompt_transliterated = transliterate(text, DEVANAGARI, ITRANS)
    
    # Tokenize inputs
    input_ids = tokenizer(description, return_tensors="pt").input_ids.to(device)
    prompt_input_ids = tokenizer(prompt_transliterated, return_tensors="pt").input_ids.to(device)
    
    # Generate audio
    with torch.no_grad():
        generation = model.generate(input_ids=input_ids, prompt_input_ids=prompt_input_ids)
    audio_arr = generation.cpu().numpy().squeeze()
    
    return audio_arr

# Sidebar for model loading and information
with st.sidebar:
    st.header("📚 Model Information")
    st.markdown("""
        **Model Details:**
        - Name: Parler-TTS-Mini
        - Version: 0.1
        - Type: Indian Male Accent (Hindi)
        - Platform: Kaggle
    """)
    
    if st.button("Load Model"):
        load_models()

# Main content area
main_col1, main_col2 = st.columns([2, 1])

with main_col1:
    # Text input section
    st.subheader("📝 Input Text")
    input_text = st.text_area(
        "Enter Hindi text in Devanagari script",
        value="भारत में कई प्रकार की भाषाएँ बोली जाती हैं।",
        height=150
    )

with main_col2:
    # Voice settings
    st.subheader("🎤 Voice Settings")
    voice_description = st.text_area(
        "Voice Description",
        value="Male Hindi Speaker, in Hindi delivers his words slowly and clearly, in a calm environment with clear audio quality.",
        height=150
    )

# Generate button and audio output
if st.button("🎵 Generate Audio"):
    if st.session_state.model is None or st.session_state.tokenizer is None:
        st.warning("⚠️ Please load the model first using the button in the sidebar!")
    else:
        try:
            with st.spinner("🎵 Generating audio..."):
                audio_array = generate_audio(
                    input_text,
                    voice_description,
                    st.session_state.model,
                    st.session_state.tokenizer
                )
                
                # Convert to float32
                audio_array_float32 = audio_array.astype(np.float32)
                
                # Save audio file
                output_path = 'generated_audio.wav'
                sf.write(output_path, audio_array_float32, st.session_state.model.config.sampling_rate)
                
                # Display audio player
                st.subheader("🎧 Generated Audio")
                st.audio(output_path)
                
                # Download button
                with open(output_path, 'rb') as f:
                    st.download_button(
                        label="📥 Download Audio",
                        data=f,
                        file_name="hindi_tts_output.wav",
                        mime="audio/wav"
                    )
                
        except Exception as e:
            st.error(f"❌ An error occurred: {str(e)}")

# Footer
st.markdown("---")
st.markdown("""
    <div style='text-align: center'>
        <p>Created with ❤️ by team DataKraft Corps using Streamlit</p>
    </div>
""", unsafe_allow_html=True)

# Add requirements information
if st.sidebar.checkbox("Show Requirements"):
    st.sidebar.markdown("""
        **Required Packages:**
        ```
        streamlit
        torch
        indic_transliteration
        parler_tts
        transformers
        soundfile
        numpy
        ```
    """)