```markdown
# 🚀 Local ChatGPT — Powered by Ollama + Python (Streaming + PDF RAG + Embeddings + FAISS)

![Python](https://img.shields.io/badge/Python-3.10+-blue)
![Flask](https://img.shields.io/badge/Flask-Web%20Framework-green)
![Ollama](https://img.shields.io/badge/Ollama-Local%20LLM-orange)
![FAISS](https://img.shields.io/badge/FAISS-Vector%20Search-purple)
![License](https://img.shields.io/badge/License-MIT-yellow)
![Stars](https://img.shields.io/github/stars/manojrammurthy/ollama-local-chatgpt?style=social)

A **fully local ChatGPT alternative** with:

✔ ChatGPT-style UI  
✔ Local LLMs via Ollama (Phi-3, LLaMA-3, Mistral, etc.)  
✔ PDF Upload + RAG  
✔ Real-time streaming responses  
✔ Embeddings using `nomic-embed-text`  
✔ FAISS vector search  
✔ No cloud, no API keys — **100% offline**

Built using **Flask + TailwindCSS + Ollama + FAISS + PyMuPDF**.

---

---

# ✨ Features

| Feature           | Description                                              |
| ----------------- | -------------------------------------------------------- |
| 🧠 Local LLM Chat | Uses any Ollama model (Phi-3, LLaMA-3, Mistral, Gemma… ) |
| 📄 PDF Upload     | Load documents and ask questions from them               |
| 🔍 RAG            | FAISS-powered retrieval from PDF chunks                  |
| 🧬 Embeddings     | Uses `nomic-embed-text` for dense vector embeddings      |
| 🚀 Live Streaming | Real-time token-by-token streaming like ChatGPT          |
| 🎨 Modern UI      | TailwindCSS dark-mode chat interface                     |
| 🔁 Model Selector | Switch Ollama models dynamically                         |
| 💾 Chat Memory    | Auto-saved in browser localStorage                       |
| 🔐 100% Offline   | No external API calls — everything runs on your laptop   |

---

# 🧠 Architecture Overview

```
Frontend (HTML + TailwindCSS)
        ↓
Flask API (Python)
        ↓
Ollama Chat Models (phi3, llama3, mistral…)
        ↓
Ollama Embedding Model (nomic-embed-text)
        ↓
FAISS Vector Search
        ↓
PDF Question Answering (RAG)
```

---

# 📁 Project Structure

```
ollama_web_chat/
│
├── app.py
├── requirements.txt
├── README.md
├── .gitignore
│
├── templates/
│   └── index.html
│
└── static/   (optional for CSS/JS assets)
```

---

# ⚙️ Installation

## 1️⃣ Install Ollama

Linux/macOS:

```bash
curl -fsSL https://ollama.com/install.sh | sh
```

Windows:
Download from [https://ollama.com/download](https://ollama.com/download)

---

## 2️⃣ Pull required Ollama models

```bash
ollama pull phi3
ollama pull llama3
ollama pull mistral
ollama pull nomic-embed-text
```

---

## 3️⃣ Install Python dependencies

```bash
pip install -r requirements.txt
```

---

## 4️⃣ Run the Flask app

```bash
python app.py
```

Then open:

```
http://localhost:5000
```

---

# 🧬 Embeddings Explained

This project uses:

```
nomic-embed-text
```

Each chunk of your PDF is converted into a vector (embedding).
These vectors are stored in a **FAISS index**.

During a question:

1. User question → embedded
2. Vector compared against FAISS index
3. Top-k relevant chunks returned
4. Passed to LLM as RAG prompt
5. LLM answers using context

This gives extremely accurate answers for:

* Annual reports
* Research papers
* Legal documents
* Technical PDFs
* Policies

---

# 📄 PDF RAG Flow

```
PDF → Text Extraction → Chunking → Embedding → FAISS Index
                                 ↑
                           User Question
                                 ↓
                            Vector Search
                                 ↓
                          Relevant Chunks
                                 ↓
                         LLM generates answer
```

---

# 🔁 Switching Models

The UI pulls all locally installed models:

```
/models → from Ollama tags API
```

You can switch between:

* phi3
* mistral
* llama3
* codellama
* gemma
* your custom models

Every change clears chat history automatically.

---

# 🚀 Roadmap

### ✔ v1.0 (Current)

* Local chat
* PDF embedding + FAISS
* Model selection
* Streaming
* Modern UI

### 🔜 v1.1

* Sidebar multi-threaded chat
* Voice input (Whisper)
* Export conversation
* Multi-PDF knowledgebase
* Dark/light toggle

### 🔮 v2.0

* Custom embedding model fine-tuning
* Workspace mode (knowledge graphs)
* Browser extension version
* Desktop app (Electron or PyInstaller)

---

# 🤝 Contributing

Pull requests welcome!
If you want to add a feature, open an issue first.

---

# ⭐ Support the Project

If you find this useful, please give this repo a ⭐ on GitHub.
It helps more people discover offline LLM tools.

---

# 📄 License

This project is licensed under the **MIT License**.

````

