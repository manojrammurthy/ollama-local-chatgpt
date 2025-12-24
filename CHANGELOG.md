# 🧾 CHANGELOG

All notable changes to **Ollama Local ChatGPT + Embedding Explorer** will be documented in this file.

---

## [1.0.1] — 2025-02-XX  
### 🎉 Added  
- Complete Chat UI (Ollama-based)  
- PDF Upload + Chunking engine  
- FAISS vector index for PDFs  
- PDF RAG question-answer system  
- Source citations via `X-RAG-Sources` header  
- Delete PDF endpoint with FAISS rebuild  
- Embedding Explorer main UI  
- Generate Embedding tool  
- Compare Embeddings tool  
- Cosine, L2, Dot product calculator  
- PCA 2D scatter plot  
- PCA 3D (Plotly) visualization  
- Difference heatmap (matrix rows)  
- Batch embedding endpoint  
- Persistent FAISS index (`explorer_index.faiss`)  
- Metadata store (`explorer_meta.json`)  
- Similarity matrix endpoint  
- Cosine similarity grid visualizer  
- KMeans clustering endpoint  
- Cluster visualization using PCA  
- Clear Explorer Index endpoint  

### 🛠 Improved  
- Refactored FAISS code for zero-copy rebuild speed  
- Stabilized heatmap rendering using deferred rendering plugin  
- Fixed multiple Chart.js canvas reuse issues  
- Stronger validation on /compare_embeddings  
- Smoother UX for panel switching  

### 🐛 Fixed  
- `cluster_embeddings` returned JSON errors (NumPy int32 issue)  
- Heatmap crashed due to missing chartArea width  
- PCA 3D endpoint generated 500 errors  
- PDF deletion left stale vectors in index  
- Chart.js memory leak when switching tools  
- Duplicate model-loading race conditions  

---

## [0.1.0] — 2025-01  
### Initial prototype  
- Basic chat UI  
- PDF upload (non-indexed)  
- Initial embedding API tests  

---

## Format Guide  
- **Added** – new features  
- **Changed** – updates to existing functionality  
- **Fixed** – bug fixes  
- **Removed** – deleted features  
- **Security** – vulnerability patches  
