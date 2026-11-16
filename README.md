# Filmo — NLP Semantic Movie Recommendation & Conversational Assistant

[![Python](https://img.shields.io/badge/Python-3.9_%7C_3.10_%7C_3.11-3776ab?logo=python&logoColor=white)](https://www.python.org/)
[![Streamlit](https://img.shields.io/badge/Frontend-Streamlit-FF4B4B?logo=streamlit&logoColor=white)](https://streamlit.io/)
[![HuggingFace](https://img.shields.io/badge/Embeddings-Sentence--Transformers-yellow?logo=huggingface&logoColor=black)](https://huggingface.co/sentence-transformers/all-MiniLM-L6-v2)
[![Scikit-Learn](https://img.shields.io/badge/ML-Scikit--Learn-F7931E?logo=scikit-learn&logoColor=white)](https://scikit-learn.org/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

**Filmo** is an intelligent cinematic information retrieval system and conversational movie recommendation assistant. Combining **dense semantic vector representations** (`all-MiniLM-L6-v2`), **sparse lexical retrieval** (TF-IDF vectorization with cosine similarity), and **sentiment polarity analysis** (TextBlob & VADER), Filmo understands colloquial natural-language movie descriptions and connects users to tailored films from an indexed IMDb dataset.

---

## ️ Algorithmic Architecture

```
 ┌────────────────────────────────────────────────────────────────────────┐
 │                      User Query (Text or Voice)                        │
 │    "A cerebral psychological thriller set in space with high tension"   │
 └───────────────────────────────────┬────────────────────────────────────┘
                                     │
           ┌─────────────────────────┴─────────────────────────┐
           ▼                                                   ▼
 ┌───────────────────────────────────┐               ┌───────────────────────────────────┐
 │   Dense Neural Vector Matching    │               │     Sparse Lexical TF-IDF Match   │
 │   • all-MiniLM-L6-v2 Embeddings   │               │   • Sublinear Term Frequency      │
 │   • Cosine Similarity vs Corpus   │               │   • Stopword Removal & N-grams    │
 └─────────────────┬─────────────────┘               └─────────────────┬─────────────────┘
                   │                                                   │
                   └─────────────────────────┬─────────────────────────┘
                                             ▼
                               ┌───────────────────────────┐
                               │  Hybrid Score Reciprocal  │
                               │  Rank Fusion (Dense+TFIDF)│
                               └─────────────┬─────────────┘
                                             ▼
                               ┌───────────────────────────┐
                               │  Sentiment Polarity Check │
                               │  (VADER & TextBlob Score) │
                               └─────────────┬─────────────┘
                                             ▼
                               ┌───────────────────────────┐
                               │   Streamlit Presentation  │
                               │  • Title, Year, Director  │
                               │  • IMDb Rating & Synopsis │
                               │  • Audio Feedback (TTS)   │
                               └───────────────────────────┘
```

---

## Features

- **Hybrid Information Retrieval:** Merges semantic deep neural embeddings with classic lexical keyword weighting to eliminate the vocabulary mismatch problem.
- **Multimodal Interaction:** Supports speech input via `SpeechRecognition` and voice feedback through local `pyttsx3` text-to-speech synthesis.
- **Conversational Memory:** Preserves multi-turn filtering state (director, era, minimum rating, genre) using Streamlit's `st.session_state`.
- **Enriched Metadata Presentation:** Displays IMDb ratings, release years, cover artwork, and director filmography.

---

## ️ Quickstart

### Prerequisites
- Python 3.9+
- A microphone and speaker (optional, for voice speech-to-text and text-to-speech)

### Installation
```bash
# 1. Clone the repository
git clone https://github.com/yasmenmad/chatbot.git
cd chatbot

# 2. Set up virtual environment
python -m venv venv
# On Windows:
venv\Scripts\activate
# On Linux/macOS:
source venv/bin/activate

# 3. Install dependencies
pip install -r requirements.txt

# 4. Launch the Streamlit application
streamlit run app.py
```

---

## Repository Structure

```
chatbot/
├── app.py                # Main Streamlit NLP application
├── python                # Legacy execution alias
├── imdbspe_fixed.csv     # Curated IMDb movie metadata & synopses dataset
├── Filmo.png             # UI branding asset
├── baground.jpg          # Interface backdrop
├── requirements.txt      # Dependency specification
├── .gitignore            # Git exclusion rules
└── README.md             # Project documentation
```

---

## ‍ Author

**Yasmine Daoudi**  
*Computer Science & Software Engineering Student*  
Université Ibn Tofail — Kénitra, Morocco  
- LinkedIn: [linkedin.com/in/yasmine-daoudi-473260340](https://www.linkedin.com/in/yasmine-daoudi-473260340)  
- GitHub: [github.com/yasmenmad](https://github.com/yasmenmad)
