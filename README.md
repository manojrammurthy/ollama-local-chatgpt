```markdown
# 🚀 Local ChatGPT — Powered by Ollama + Python (Streaming + PDF RAG + Embeddings + FAISS)

![Python](https://img.shields.io/badge/Python-3.10+-blue)
![Flask](https://img.shields.io/badge/Flask-Web%20Framework-green)
![Ollama](https://img.shields.io/badge/Ollama-Local%20LLM-orange)
![FAISS](https://img.shields.io/badge/FAISS-Vector%20Search-purple)
![License](https://img.shields.io/badge/License-MIT-yellow)
![Stars](https://img.shields.io/github/stars/manojrammurthy/ollama-local-chatgpt?style=social)
🚀 Overview

This project is a local AI research and development platform that combines:

ChatGPT-like local LLM chat interface

Streamed responses

Model selector

PDF-aware RAG mode

Source citations

Session history

Embedding Explorer (Full Suite)

Generate embeddings

PCA 2D

PCA 3D

Cosine similarity / L2 / dot product

Difference heatmap

Batch embedding

FAISS-based persistent vector index

Similarity matrix

KMeans clustering visualization

PDF Intelligence (RAG)

Upload PDFs

Auto chunking

Embedding + FAISS indexing

Ask questions filtered by PDF

Page-aware source highlighting

Delete + rebuild index cleanly

This tool is ideal for:

Building RAG systems

Understanding embeddings

Debugging semantic similarity

Learning ML engineering

Explaining LLM internals

Academic demonstrations

🧰 Features
🧠 Local ChatGPT (Ollama UI)

✔ Stream chat responses
✔ Switch models instantly
✔ PDF mode toggle
✔ Multi-PDF selection
✔ Extracted page sources
✔ Typing animations
✔ Clean dark UI

🔍 Embedding Explorer — Advanced Tools
📌 1. Generate Embeddings

Instant vector preview

First 30 dims

PCA (2D scatter)

Dimension display

🔗 2. Compare Two Text Embeddings

Cosine similarity

L2 distance

Dot product

PCA 2D comparison

PCA 3D visualization (Plotly)

Dim-wise difference heatmap

🧬 3. Cluster Explorer

Batch embed any texts

Persist in FAISS

KMeans clustering

PCA 2D cluster plot

🧩 4. Similarity Matrix

Interactive cosine similarity grid

Values + color-coded cells

Scales based on semantic closeness

📚 PDF RAG Engine

✔ Upload any PDF
✔ Extract + chunk content
✔ Auto-embed with nomic-embed-text
✔ Build FAISS index
✔ Query with selected PDFs
✔ Return exact pages as sources
✔ Delete PDFs + clean index

🧱 Tech Stack

Backend: Python, Flask

LLM Runtime: Ollama (phi3, nomic-embed-text)

Vector Index: FAISS

Math/ML: NumPy, scikit-learn

Frontend: Tailwind CSS, Chart.js, Plotly

PDF: PyMuPDF (fitz)

Everything runs offline, local, and fast.

📦 Project Structure
ollama-local-chatgpt/
│── app.py
│── requirements.txt
│── uploaded_pdfs/
│── explorer_index.faiss
│── explorer_meta.json
│── templates/
│     ├── index.html
│     └── embedding_explorer.html
│── static/
└── README.md

▶️ Installation & Usage
1️⃣ Install dependencies
pip install -r requirements.txt

2️⃣ Start Ollama
ollama serve

3️⃣ Pull required models
ollama pull phi3
ollama pull nomic-embed-text

4️⃣ Run server
python app.py

5️⃣ Open in browser

➡ http://localhost:5000

🔥 Roadmap
Phase 1 — Embedding Visualizations

✔ Completed

Phase 2 — Financial Embedding Model

⏳ Next

Phase 3 — Fine-tuning embeddings

🎯 Coming soon

Phase 4 — Multi-user AI dashboard (Django + Postgres vector DB)

🔥 Future milestone

Phase 5 — Desktop version (Electron / Tauri)

🖥️ Planned
