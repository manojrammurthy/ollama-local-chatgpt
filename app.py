from flask import Flask, request, Response, jsonify, render_template, send_file
from flask_cors import CORS
import requests
import json
import fitz
import numpy as np
import faiss
import os
import io

app = Flask(__name__)
CORS(app)

messages = []
MODEL = "phi3"
EMBED_MODEL = "nomic-embed-text"

documents = []       # stores PDF metadata + chunks
faiss_index = None   # global FAISS index
faiss_meta = []      # parallel metadata list for each vector


# -------------------------------
# HOME
# -------------------------------
@app.route("/")
def home():
    return render_template("index.html")


# -------------------------------
# CLEAR CONVERSATION
# -------------------------------
@app.route("/clear", methods=["POST"])
def clear():
    global messages
    messages = []
    return jsonify({"status": "cleared"})


# -------------------------------
# OLLAMA MODELS
# -------------------------------
@app.route("/models", methods=["GET"])
def models():
    try:
        response = requests.get("http://localhost:11434/api/tags")
        data = response.json()
        names = [m["name"] for m in data.get("models", [])]
        return jsonify({"models": names})
    except:
        return jsonify({"models": []})


# -------------------------------
# SET MODEL
# -------------------------------
@app.route("/set_model", methods=["POST"])
def set_model():
    global MODEL, messages
    MODEL = request.json.get("model", MODEL)
    messages = []
    return jsonify({"status": "ok", "model": MODEL})


# -------------------------------
# PDF THUMBNAIL (first page)
# -------------------------------
@app.route("/pdf_thumbnail/<path:doc_id>")
def pdf_thumbnail(doc_id):
    try:
        for doc in documents:
            if doc["doc_id"] == doc_id:
                pdf_path = doc["path"]
                break
        else:
            return "Not found", 404

        pdf = fitz.open(pdf_path)
        page = pdf.load_page(0)
        pix = page.get_pixmap(matrix=fitz.Matrix(0.3, 0.3))

        return send_file(
            io.BytesIO(pix.tobytes("png")),
            mimetype="image/png"
        )

    except Exception as e:
        print("Thumbnail error:", e)
        return "Error", 500


# -------------------------------
# NORMAL CHAT
# -------------------------------
@app.route("/chat", methods=["POST"])
def chat():
    user_msg = request.json.get("message", "")
    messages.append({"role": "user", "content": user_msg})

    payload = {
        "model": MODEL,
        "messages": messages,
        "stream": True
    }

    def generate():
        assistant_reply = ""
        response = requests.post(
            "http://localhost:11434/api/chat",
            json=payload,
            stream=True
        )

        for line in response.iter_lines():
            if line:
                chunk = json.loads(line.decode())
                delta = chunk.get("message", {}).get("content", "")
                assistant_reply += delta
                yield delta

        messages.append({"role": "assistant", "content": assistant_reply})

    return Response(generate(), content_type="text/plain")


# -------------------------------
# PDF UPLOAD + RAG CHUNK INDEXING
# -------------------------------
@app.route("/upload_pdf", methods=["POST"])
def upload_pdf():
    global faiss_index, faiss_meta, documents

    UPLOAD_DIR = "uploaded_pdfs"
    os.makedirs(UPLOAD_DIR, exist_ok=True)

    if "file" not in request.files:
        return jsonify({"error": "No file provided"}), 400

    file = request.files["file"]
    filename = file.filename
    file_path = os.path.join(UPLOAD_DIR, filename)

    # Read once → save → parse
    file_bytes = file.read()
    with open(file_path, "wb") as f:
        f.write(file_bytes)

    try:
        pdf = fitz.open(stream=file_bytes, filetype="pdf")
    except:
        return jsonify({"error": "Invalid PDF"}), 400

    chunks = []
    pages = []

    for page_num, page in enumerate(pdf):
        text = page.get_text()
        words = text.split()
        CHUNK = 200

        for i in range(0, len(words), CHUNK):
            chunk_text = " ".join(words[i:i+CHUNK])
            chunks.append(chunk_text)
            pages.append(page_num + 1)

    # Save metadata
    doc_obj = {
        "doc_id": filename,
        "path": file_path,
        "chunks": chunks,
        "pages": pages
    }
    documents.append(doc_obj)

    # Generate embeddings
    vectors = []
    for idx, chunk in enumerate(chunks):

        emb = requests.post(
            "http://localhost:11434/api/embed",
            json={"model": EMBED_MODEL, "input": chunk}
        ).json()

        if "embeddings" not in emb:
            print("Embedding failed for chunk", idx)
            continue

        vector = emb["embeddings"][0]
        vectors.append(vector)

        faiss_meta.append({
            "doc_id": filename,
            "page": pages[idx],
            "chunk": chunk
        })

    vectors = np.array(vectors, dtype="float32")

    if faiss_index is None:
        faiss_index = faiss.IndexFlatL2(vectors.shape[1])

    faiss_index.add(vectors)

    return jsonify({
        "file": filename,
        "chunks": len(chunks),
        "status": "indexed"
    })


# -------------------------------
# ASK QUESTIONS OVER PDF (RAG)
# -------------------------------
@app.route("/ask_pdf", methods=["POST"])
def ask_pdf():
    global faiss_index, faiss_meta, MODEL

    selected_docs = request.json.get("selected_docs", [])
    question = request.json.get("message", "")

    if faiss_index is None:
        return jsonify({"error": "No PDFs uploaded"}), 400

    qemb = requests.post(
        "http://localhost:11434/api/embed",
        json={"model": EMBED_MODEL, "input": question}
    ).json()

    if "embeddings" not in qemb:
        return jsonify({"error": "Embedding failed"}), 400

    qvec = np.array(qemb["embeddings"][0], dtype="float32").reshape(1, -1)

    # Search top-5 chunks
    distances, indices = faiss_index.search(qvec, 5)

    context = ""

    for idx in indices[0]:
        meta = faiss_meta[idx]

        if selected_docs and meta["doc_id"] not in selected_docs:
            continue

        context += f"\n[PDF: {meta['doc_id']} | Page {meta['page']}]\n{meta['chunk']}\n"

    prompt = f"""
Use ONLY the context below to answer the question.

Context:
{context}

Question: {question}

If the answer is NOT found in the context, reply:
"The documents do not contain this information."

Answer:
"""

    messages.append({"role": "user", "content": prompt})

    payload = {"model": MODEL, "messages": messages, "stream": True}

    def generate():
        assistant_reply = ""
        resp = requests.post(
            "http://localhost:11434/api/chat",
            json=payload,
            stream=True
        )

        for line in resp.iter_lines():
            if line:
                chunk = json.loads(line.decode())
                delta = chunk.get("message", {}).get("content", "")
                assistant_reply += delta
                yield delta

        messages.append({"role": "assistant", "content": assistant_reply})

    return Response(generate(), content_type="text/plain")


# -------------------------------
# EMBEDDING EXPLORER ENDPOINTS
# -------------------------------
@app.route("/embedding_explorer")
def embedding_explorer():
    return render_template("embedding_explorer.html")


@app.route("/embed_text", methods=["POST"])
def embed_text():
    text = request.json.get("text", "").strip()
    if not text:
        return jsonify({"error": "empty"}), 400

    emb = requests.post(
        "http://localhost:11434/api/embed",
        json={"model": EMBED_MODEL, "input": text}
    ).json()

    if "embeddings" not in emb:
        return jsonify({"error": "embedding_failed"}), 400

    vec = emb["embeddings"][0]
    arr = np.array(vec).reshape(1, -1)

    from sklearn.decomposition import PCA
    arr2 = np.vstack([arr, arr + np.random.normal(0, 1e-6, arr.shape)])
    pca = PCA(n_components=2)
    reduced = pca.fit_transform(arr2)[0].tolist()

    return jsonify({
        "vector": vec,
        "vector_preview": vec[:30],
        "dimension": len(vec),
        "pca": reduced
    })

@app.route("/delete_pdf", methods=["POST"])
def delete_pdf():
    global documents, faiss_index, faiss_meta

    doc_id = request.json.get("doc_id")
    if not doc_id:
        return jsonify({"error": "No doc_id provided"}), 400

    # --- 1️⃣ Remove file from disk ---
    for doc in documents:
        if doc["doc_id"] == doc_id:
            try:
                os.remove(doc["path"])
            except:
                pass
            break

    # --- 2️⃣ Remove from documents list ---
    documents = [doc for doc in documents if doc["doc_id"] != doc_id]

    # --- 3️⃣ Remove its vectors from FAISS metadata ---
    faiss_meta = [m for m in faiss_meta if m["doc_id"] != doc_id]

    # --- 4️⃣ Rebuild FAISS index ---
    if len(faiss_meta) == 0:
        faiss_index = None
        return jsonify({"status": "deleted", "remaining": 0})

    # Extract all remaining vectors again
    all_vectors = []

    for meta in faiss_meta:
        emb = requests.post(
            "http://localhost:11434/api/embed",
            json={"model": EMBED_MODEL, "input": meta["chunk"]}
        ).json()

        if "embeddings" in emb:
            all_vectors.append(emb["embeddings"][0])

    if len(all_vectors) == 0:
        faiss_index = None
        return jsonify({"status": "deleted", "remaining": 0})

    # Rebuild FAISS
    all_vectors = np.array(all_vectors, dtype="float32")
    dim = all_vectors.shape[1]

    faiss_index = faiss.IndexFlatL2(dim)
    faiss_index.add(all_vectors)

    return jsonify({
        "status": "deleted",
        "remaining": len(all_vectors)
    })


@app.route("/compare_embeddings", methods=["POST"])
def compare_embeddings():
    t1 = request.json.get("text1", "")
    t2 = request.json.get("text2", "")

    if not t1 or not t2:
        return jsonify({"error": "missing"}), 400

    e1 = requests.post(
        "http://localhost:11434/api/embed",
        json={"model": EMBED_MODEL, "input": t1}
    ).json()

    e2 = requests.post(
        "http://localhost:11434/api/embed",
        json={"model": EMBED_MODEL, "input": t2}
    ).json()

    v1 = np.array(e1["embeddings"][0])
    v2 = np.array(e2["embeddings"][0])

    cos = float(np.dot(v1, v2) / (np.linalg.norm(v1) * np.linalg.norm(v2)))
    l2 = float(np.linalg.norm(v1 - v2))
    dot = float(np.dot(v1, v2))

    from sklearn.decomposition import PCA
    pts = PCA(n_components=2).fit_transform(np.vstack([v1, v2])).tolist()

    return jsonify({
        "cosine_similarity": cos,
        "l2_distance": l2,
        "dot_product": dot,
        "pca_points": pts,
        "diff_vector": (v1 - v2).tolist()
    })


if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000, debug=True)
