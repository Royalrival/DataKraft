# 🎙️ Voice Synthesis Studio

## Project Overview

Voice Synthesis Studio is an advanced text-to-speech application built with Streamlit, offering multiple voice models and emotional voice generation capabilities. The application provides a user-friendly interface for generating high-quality synthesized speech across different languages and styles.

## Features

- 🗣️ Multiple Voice Models:

  - Indian Accent English
  - Native Hindi
  - Emotion-based Voice Generation

- 🎭 Emotion Control

  - Support for various emotional tones
  - Customizable voice descriptions

- 📜 Generation History
  - Track and replay previous audio generations
  - Download generated audio files

## Prerequisites

- Python 3.8+
- Required libraries (see `requirements.txt`):
  - Streamlit
  - Torch
  - Transformers
  - Parler TTS
  - Indic Transliteration
  - SoundFile

## Installation

1. Clone the repository:

   ```bash
   git clone https://github.com/Royalrival/DataKraft.git
   cd DataKraft
   ```

2. Create a virtual environment:

   ```bash
   python -m venv venv
   source venv/bin/activate  # On Windows, use `venv\Scripts\activate`
   ```

3. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```

## Running the Application

```bash
streamlit run main.py
```

## Project Structure

```
DataKraft/
│
├── .devcontainer/
│   └── devcontainer.json
│
├── .gitignore
│
├── combined_streamlit/
│
├── constraints.txt
│
├── indianaccent_streamlit/
│
├── indic_streamlit.py
│
├── main.py             # Main Streamlit application
│
├── parler_indic.py
│
├── parlerstreamlit.py
│
├── parlertts/
│
├── README.md
│
└── requirements.txt
```

## Models Used

1. **Indian Accent Model**

   - Path: `En1gma02/Parler-TTS-Mini-v0.1-Indian-Accent-Kaggle`
   - Specializes in Indian English accents
   - High clarity and pronunciation

2. **Hindi Model**

   - Path: `En1gma02/Parler-TTS-Mini-v0.1-Indian-Male-Accent-Hindi-Kaggle`
   - Native Hindi language support
   - Devanagari script compatible

3. **Emotions Model**
   - Path: `En1gma02/Parler-TTS-Mini-v1-English-Emotions`
   - Multiple emotion styles
   - Dynamic voice modulation

## Usage Guidelines

- Select a model from the sidebar
- Load the model before generation
- Enter text and adjust voice description
- Generate and download audio

## Contributing

1. Fork the repository
2. Create your feature branch (`git checkout -b feature/AmazingFeature`)
3. Commit your changes (`git commit -m 'Add some AmazingFeature'`)
4. Push to the branch (`git push origin feature/AmazingFeature`)
5. Open a Pull Request

## License

Distributed under the MIT License. See `LICENSE` for more information.

## Acknowledgments

- [Parler TTS](https://huggingface.co/parler-tts)
- [Streamlit](https://streamlit.io/)
- DataKraft Corps Team

---

Made with ❤️ by DataKraft Corps
