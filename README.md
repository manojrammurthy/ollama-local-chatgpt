```markdown
# 🚀 Local ChatGPT — Powered by Ollama + Python (Streaming + PDF RAG + Embeddings + FAISS)

![Python](https://img.shields.io/badge/Python-3.10+-blue)
![Flask](https://img.shields.io/badge/Flask-Web%20Framework-green)
![Ollama](https://img.shields.io/badge/Ollama-Local%20LLM-orange)
![FAISS](https://img.shields.io/badge/FAISS-Vector%20Search-purple)
![License](https://img.shields.io/badge/License-MIT-yellow)
![Stars](https://img.shields.io/github/stars/manojrammurthy/ollama-local-chatgpt?style=social)
# 💬 Ollama Local ChatGPT + Embedding Explorer + PDF RAG  
**A fully local AI research & development platform built with Flask, FAISS, Plotly, Chart.js, Tailwind, and Ollama.**

🔗 **GitHub Repo:** https://github.com/manojrammurthy/ollama-local-chatgpt  
🧠 *Everything runs offline — no API keys or cloud required.*

---

## 🚀 Overview

This project is a **complete local AI workspace** that brings together:

### ✔ Local ChatGPT UI (Ollama)
- Streaming chat
- Model selector  
- PDF RAG mode  
- Source citations  
- Clean dark UI  

### ✔ Embedding Explorer (Full Interactive Toolkit)
- Generate embeddings  
- PCA (2D)  
- PCA (3D – Plotly)  
- Cosine similarity / L2 distance / Dot product  
- Dim-wise difference heatmap  
- Batch embedding  
- FAISS persistent vector index  
- Similarity matrix  
- KMeans clustering with PCA visual  

### ✔ PDF Intelligence (RAG)
- Upload PDFs  
- Auto chunking  
- Generate embeddings  
- Store in FAISS  
- Query by selected PDFs  
- Show page numbers as sources  
- Delete + auto-rebuild index  

Ideal for:
- RAG development  
- Embedding analysis  
- ML education  
- LLM debugging  
- Research & academic demos  

---

## 🧰 Features

### 🧠 Local ChatGPT UI
- Streamed messages  
- Model switching  
- PDF mode toggle  
- Multi-PDF selection  
- Page-level citations  
- Typing indicator  

---

## 🔍 Embedding Explorer Tools

### **1️⃣ Generate Embeddings**
- Vector preview  
- First 30 dims  
- PCA (2D scatter)  
- Dimension display  

### **2️⃣ Compare Two Texts**
- Cosine similarity  
- L2 distance  
- Dot product  
- PCA 2D comparison  
- PCA 3D visualization  
- Difference heatmap  

### **3️⃣ Cluster Explorer**
- Batch embedding  
- KMeans clustering  
- PCA 2D visualization  
- FAISS-powered  
- Persistent index  

### **4️⃣ Cosine Similarity Matrix**
- Full NxN grid  
- Color-coded similarity  
- Label display  

---

## 📚 PDF RAG Engine
✔ Upload any PDF  
✔ Auto chunk text  
✔ Embed with `nomic-embed-text`  
✔ FAISS vector index  
✔ Ask questions using selected PDFs  
✔ Show exact source pages  
✔ Delete PDFs & rebuild index  

---

## 🧱 Tech Stack

| Layer | Technology |
|-------|------------|
| Backend | **Python, Flask** |
| LLM Runtime | **Ollama (phi3, nomic-embed-text)** |
| Vector Index | **FAISS** |
| Math | **NumPy, scikit-learn** |
| Frontend | **Tailwind CSS, Chart.js, Plotly** |
| PDF Parsing | **PyMuPDF (fitz)** |

Everything runs **locally**, **offline**, **fast**.

---

# 📦 Project Structure



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

---

## ▶️ Installation


---

## ▶️ Installation

### 1️⃣ Install Python dependencies
```sh
pip install -r requirements.txt
2️⃣ Start Ollama
sh
Copy code
ollama serve
3️⃣ Pull required models
sh
Copy code
ollama pull phi3
ollama pull nomic-embed-text
4️⃣ Run the Flask app
sh
Copy code
python app.py
5️⃣ Open in your browser
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
