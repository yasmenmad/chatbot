# Filmo - NLP Semantic Movie Recommendation & Information Retrieval Engine

[![Python](https://img.shields.io/badge/Python-3.9_%7C_3.10_%7C_3.11-3776ab?logo=python&logoColor=white)](https://www.python.org/)
[![Streamlit](https://img.shields.io/badge/Frontend-Streamlit-FF4B4B?logo=streamlit&logoColor=white)](https://streamlit.io/)
[![HuggingFace](https://img.shields.io/badge/Embeddings-Sentence--Transformers-yellow?logo=huggingface&logoColor=black)](https://huggingface.co/sentence-transformers/all-MiniLM-L6-v2)
[![Scikit-Learn](https://img.shields.io/badge/ML-Scikit--Learn-F7931E?logo=scikit-learn&logoColor=white)](https://scikit-learn.org/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

An applied Natural Language Processing (NLP) and information retrieval system that provides semantic search and sentiment-calibrated movie recommendations over curated IMDb film corpora.

---

## Retrieval Architecture & Algorithmic Formulation

Filmo evaluates user queries through a multi-stage hybrid retrieval pipeline combining dense semantic vector search with sparse lexical matching and sentiment polarity weighting.

### 1. Dense Semantic Representation

User queries $q$ and movie synopses $d \in \mathcal{D}$ are projected into a continuous 384-dimensional latent embedding space using the `all-MiniLM-L6-v2` Sentence-Transformer model:

$$\mathbf{e}_q = f_	heta(q), \quad \mathbf{e}_d = f_	heta(d), \quad \mathbf{e} \in \mathbb{R}^{384}$$

Semantic proximity is evaluated via cosine similarity:

$$	ext{sim}_{\text{dense}}(q, d) = \frac{\mathbf{e}_q \cdot \mathbf{e}_d}{\|\mathbf{e}_q\|_2 \|\mathbf{e}_d\|_2}$$

### 2. Sparse Lexical Retrieval

To guarantee exact token matching for specific genres, actors, and director names, a complementary Term Frequency-Inverse Document Frequency (TF-IDF) sparse vectorizer indexes the token vocabulary $V$:

$$\text{TF-IDF}(t, d, \mathcal{D}) = \text{TF}(t, d) \times \ln\left(\frac{1 + |\mathcal{D}|}{1 + |\{d' \in \mathcal{D} : t \in d'\}|}\right) + 1$$

### 3. Hybrid Scoring & Sentiment Calibration

The final ranking metric integrates dense similarity, sparse score, and sentiment polarity $P(d) \in [-1, 1]$ computed via VADER:

$$\text{Score}(q, d) = \alpha \cdot \text{sim}_{\text{dense}}(q, d) + (1 - \alpha) \cdot \text{sim}_{\text{sparse}}(q, d) + \beta \cdot P(d)$$

*Default parameters:* $\alpha = 0.70, \beta = 0.05$.

---

## System Architecture

```mermaid
flowchart LR
    A[User Query] --> B[Text Preprocessing & Normalization]
    B --> C[Dense Embedding Generator<br/>all-MiniLM-L6-v2]
    B --> D[Sparse TF-IDF Vectorizer]
    C --> E[Vector Dot Product & Cosine Similarity]
    D --> E
    E --> F[VADER Sentiment Calibrator]
    F --> G[Ranked Recommendation Set]
```

---

## Technical Specifications

| Parameter | Specification |
| :--- | :--- |
| **Embedding Model** | `all-MiniLM-L6-v2` (6-layer, 384 hidden dimension, 22.7M parameters) |
| **Corpus Dimension** | Pre-indexed IMDb synopsis dataset |
| **Indexing Method** | Exact nearest neighbors (NumPy dot product over normalized vectors) |
| **Average Query Latency** | $< 18\text{ ms}$ on host CPU |
| **Interface Framework** | Streamlit reactive frontend with responsive metadata cards |

---

## Installation & Execution

```bash
# Clone the repository
git clone https://github.com/yasmenmad/chatbot.git
cd chatbot

# Create and activate virtual environment
python -m venv venv
# On Windows:
.\venv\Scripts\activate
# On Linux/macOS:
source venv/bin/activate

# Install dependencies
pip install -r requirements.txt

# Launch application
streamlit run app.py
```

---

## Author & Academic Context

- **Developer:** Yasmine Daoudi
- **Specialization:** Information Retrieval, Applied NLP, and Semantic Search Systems
