# 🎙️ Voice Synthesis Studio

## Project Overview

Voice Synthesis Studio is an innovative Streamlit application that leverages cutting-edge text-to-speech (TTS) models to generate high-quality audio from text inputs. The application supports multiple models with unique capabilities, including Indian English accents, Hindi language synthesis, and emotion-based voice generation.

## 🌟 Key Features

- **Multiple Voice Models**:

  - Indian Accent Model: Specialized for Indian English pronunciations
  - Hindi Language Model: Native Hindi text-to-speech conversion
  - Emotions Model: Dynamic voice modulation with emotional expressions

- **Advanced Customization**:

  - Text input for synthesis
  - Voice description and tone control
  - Emotion selection (for Emotions model)
  - Transliteration support for Hindi inputs

- **User-Friendly Interface**:
  - Intuitive sidebar for model selection
  - Progress tracking during model loading and audio generation
  - Audio playback and download options
  - Generation history tracking

## 🚀 How to Use

1. **Select a Model**:

   - Choose from Indian Accent, Hindi, or Emotions models
   - Load the selected model using the "Load Model" button

2. **Generate Audio**:

   - Enter your text in the input area
   - Customize voice description (optional)
   - For Emotions model, select a specific emotion
   - Click "Generate Voice" to create audio

3. **Audio Management**:
   - Listen to the generated audio
   - Download the audio file
   - View generation history

## 🔧 Technical Details

- **Technologies Used**:

  - Streamlit
  - Transformers (Hugging Face)
  - Parler TTS
  - Torch
  - Indic Transliteration

- **Model Sources**:
  - Parler TTS Models from Hugging Face
  - Specialized Indian English and Hindi models

## 🌈 Requirements

- Python 3.8+
- Streamlit
- Torch
- Transformers
- Indic Transliteration
- Parler TTS

## 📦 Installation

```bash
pip install streamlit torch transformers indic_transliteration parler_tts soundfile
```

## 👥 Contributors

Created with ❤️ by DataKraft Corps

## 🚀 Deployment

Easily deployable on Streamlit Community Cloud with minimal configuration.

---

**Note**: Ensure you have the necessary dependencies installed before running the application.
