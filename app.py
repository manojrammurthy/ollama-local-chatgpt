from flask import Flask, request, Response, jsonify, render_template, send_file
from flask_cors import CORS
import requests
import json
import fitz
import numpy as np
import faiss
import os
import io
import json


app = Flask(__name__)
CORS(app)

messages = []
MODEL = "phi3"
EMBED_MODEL = "nomic-embed-text"

documents = []       # stores PDF metadata + chunks
faiss_index = None   # global FAISS index
faiss_meta = []      # parallel metadata list for each vector

# Explorer FAISS index (persistent)
EXPLORER_FAISS_PATH = "explorer_index.faiss"
EXPLORER_META_PATH = "explorer_meta.json"

explorer_index = None
explorer_meta = []  # [{id, text, vector}]


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

    # Always read ONCE
    file_bytes = file.read()

    file_path = os.path.join(UPLOAD_DIR, filename)
    with open(file_path, "wb") as f:
        f.write(file_bytes)

    try:
        pdf = fitz.open(stream=file_bytes, filetype="pdf")
    except Exception as e:
        return jsonify({"error": "Invalid PDF", "details": str(e)}), 400

    chunks = []
    pages = []

    CHUNK = 200

    for page_num, page in enumerate(pdf):
        words = page.get_text().split()

        for i in range(0, len(words), CHUNK):
            chunk = " ".join(words[i:i + CHUNK])
            chunks.append(chunk)
            pages.append(page_num + 1)

    # Save PDF metadata
    doc_obj = {
        "doc_id": filename,
        "path": file_path,
        "chunks": chunks,
        "pages": pages
    }
    documents.append(doc_obj)

    # Generate embeddings in batch
    vectors = []
    for idx, chunk in enumerate(chunks):
        emb = requests.post(
            "http://localhost:11434/api/embed",
            json={"model": EMBED_MODEL, "input": chunk}
        ).json()

        if "embeddings" not in emb:
            continue

        vec = emb["embeddings"][0]
        vectors.append(vec)

        # Store metadata
        faiss_meta.append({
            "doc_id": filename,
            "page": pages[idx],
            "chunk": chunk,
            "vec": vec  # store vector directly (important for fast delete)
        })

    # Build/update FAISS index
    vectors = np.array(vectors, dtype="float32")

    if faiss_index is None:
        faiss_index = faiss.IndexFlatL2(vectors.shape[1])

    faiss_index.add(vectors)

    return jsonify({"file": filename, "chunks": len(chunks)})


# -------------------------------
# ASK QUESTIONS OVER PDF (RAG)
# -------------------------------
from flask import make_response

# -------------------------------
# ASK QUESTIONS OVER PDF (RAG)
# -------------------------------
from flask import make_response

@app.route("/ask_pdf", methods=["POST"])
def ask_pdf():
    global faiss_index, faiss_meta, MODEL

    selected_docs = request.json.get("selected_docs", [])
    question = request.json.get("message", "")

    if faiss_index is None:
        return jsonify({"error": "No PDFs uploaded"}), 400

    # Embed query
    qemb = requests.post(
        "http://localhost:11434/api/embed",
        json={"model": EMBED_MODEL, "input": question}
    ).json()

    if "embeddings" not in qemb:
        return jsonify({"error": "Embedding failed"}), 400

    qvec = np.array(qemb["embeddings"][0], dtype="float32").reshape(1, -1)

    # --------------------
    # FAISS SEARCH
    # --------------------
    k = 5
    distances, indices = faiss_index.search(qvec, k)

    context = ""
    source_info = []   # <-- used for frontend "Sources" panel

    for idx in indices[0]:
        meta = faiss_meta[idx]

        # Respect selected PDF filters
        if selected_docs and meta["doc_id"] not in selected_docs:
            continue

        # Add to RAG Context
        context += f"\n[PDF: {meta['doc_id']} | Page {meta['page']}]\n{meta['chunk']}\n"

        # Add to Sources box
        source_info.append({
            "doc_id": meta["doc_id"],
            "page": meta["page"]
        })

    # --------------------
    # Build RAG Prompt
    # --------------------
    prompt = f"""
Use ONLY the context below to answer the question.

Context:
{context}

Question: {question}

If the answer is NOT found in the context, say:
"The documents do not contain this information."

Answer:
"""

    messages.append({"role": "user", "content": prompt})
    payload = {"model": MODEL, "messages": messages, "stream": True}

    # --------------------
    # Streaming LLM Response
    # --------------------
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

    # --------------------
    # Attach Sources Header
    # --------------------
    response = Response(generate(), content_type="text/plain")
    response.headers["X-RAG-Sources"] = json.dumps(source_info)

    return response

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

    # 1️⃣ Remove file from disk
    for doc in documents:
        if doc["doc_id"] == doc_id:
            try:
                os.remove(doc["path"])
            except:
                pass
            break

    # 2️⃣ Remove documents metadata
    documents = [doc for doc in documents if doc["doc_id"] != doc_id]

    # 3️⃣ Keep only metadata from other PDFs
    faiss_meta = [m for m in faiss_meta if m["doc_id"] != doc_id]

    # 4️⃣ Rebuild FAISS index FAST using stored embeddings
    if not faiss_meta:
        faiss_index = None
        return jsonify({"status": "deleted", "remaining_vectors": 0})

    vectors = np.array([m["vec"] for m in faiss_meta], dtype="float32")

    faiss_index = faiss.IndexFlatL2(vectors.shape[1])
    faiss_index.add(vectors)

    return jsonify({
        "status": "deleted",
        "remaining_vectors": len(vectors)
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

@app.route("/pca_3d", methods=["POST"])
def pca_3d():
    text1 = request.json.get("text1", "")
    text2 = request.json.get("text2", "")

    if not text1 or not text2:
        return jsonify({"error": "Both text1 and text2 are required"}), 400

    # Embed both texts
    e1 = requests.post(
        "http://localhost:11434/api/embed",
        json={"model": EMBED_MODEL, "input": text1}
    ).json()

    e2 = requests.post(
        "http://localhost:11434/api/embed",
        json={"model": EMBED_MODEL, "input": text2}
    ).json()

    if "embeddings" not in e1 or "embeddings" not in e2:
        return jsonify({"error": "Embedding failed"}), 500

    v1 = np.array(e1["embeddings"][0])
    v2 = np.array(e2["embeddings"][0])

    # Make PCA stable by adding tiny noise duplicates
    M = np.vstack([
        v1,
        v2,
        v1 + np.random.normal(0, 1e-6, v1.shape),
        v2 + np.random.normal(0, 1e-6, v2.shape)
    ])

    from sklearn.decomposition import PCA
    pca = PCA(n_components=3)
    p3d = pca.fit_transform(M)

    # Only return the real points (first two rows)
    points_3d = p3d[:2].tolist()

    return jsonify({
        "points_3d": points_3d,
        "explained_variance": pca.explained_variance_ratio_.tolist()
    })


@app.route("/embed_batch", methods=["POST"])
def embed_batch():
    global explorer_index, explorer_meta

    texts = request.json.get("texts", [])
    if not texts:
        return jsonify({"error": "No texts provided"}), 400

    vectors = []

    for txt in texts:
        res = requests.post(
            "http://localhost:11434/api/embed",
            json={"model": EMBED_MODEL, "input": txt}
        ).json()

        if "embeddings" not in res:
            return jsonify({"error": "embedding_failed", "text": txt}), 400

        vectors.append(res["embeddings"][0])

    vectors_np = np.array(vectors, dtype="float32")

    # Create FAISS index if first time
    if explorer_index is None:
        dim = vectors_np.shape[1]
        explorer_index = faiss.IndexFlatL2(dim)

    # Add vectors
    explorer_index.add(vectors_np)

    # Save metadata
    for i, txt in enumerate(texts):
        explorer_meta.append({
            "id": len(explorer_meta),
            "text": txt
        })

    # Persist index + metadata
    faiss.write_index(explorer_index, EXPLORER_FAISS_PATH)
    with open(EXPLORER_META_PATH, "w") as f:
        json.dump(explorer_meta, f, indent=2)

    return jsonify({
        "count": len(texts),
        "total": explorer_index.ntotal
    })

@app.route("/similarity_matrix", methods=["GET"])
def similarity_matrix():
    global explorer_index, explorer_meta

    if explorer_index is None or explorer_index.ntotal == 0:
        return jsonify({"error": "No vectors indexed"}), 400

    xb = explorer_index.reconstruct_n(0, explorer_index.ntotal)
    xb = np.array(xb)

    # Normalise rows
    normed = xb / np.linalg.norm(xb, axis=1, keepdims=True)

    sim_matrix = np.matmul(normed, normed.T)

    return jsonify({
        "matrix": sim_matrix.tolist(),
        "labels": [m["text"][:50] for m in explorer_meta]
    })


from sklearn.cluster import KMeans
from sklearn.decomposition import PCA

@app.route("/cluster_embeddings", methods=["POST"])
def cluster_embeddings():
    global explorer_index, explorer_meta

    try:
        if explorer_index is None or explorer_index.ntotal == 0:
            return jsonify({"error": "No vectors indexed"}), 400

        k = int(request.json.get("k", 3))

        # Safe reconstruction
        xb = np.vstack([
            explorer_index.reconstruct(i)
            for i in range(explorer_index.ntotal)
        ])

        n = xb.shape[0]

        if k < 2:
            return jsonify({"error": "k must be >= 2"}), 400
        if k >= n:
            return jsonify({"error": f"k must be < {n}"}), 400

        # KMeans (compatible with sklearn 1.6+)
        model = KMeans(n_clusters=k, n_init=10, random_state=42)
        labels = model.fit_predict(xb)

        # Convert labels (NumPy int32 → Python int)
        labels_list = [int(l) for l in labels]

        # PCA
        pts2D = PCA(n_components=2).fit_transform(xb).tolist()

        # Group clusters: keys must be STRINGS
        clusters = {}
        for i, lbl in enumerate(labels_list):
            key = str(lbl)          # important!
            clusters.setdefault(key, []).append({
                "id": int(i),
                "text": explorer_meta[i]["text"]
            })

        return jsonify({
            "k": k,
            "labels": labels_list,
            "clusters": clusters,
            "pca2d": pts2D
        })

    except Exception as e:
        print("🔥 CLUSTER ERROR:", str(e))
        return jsonify({"error": str(e)}), 500




@app.route("/clear_explorer_index", methods=["POST"])
def clear_explorer_index():
    global explorer_index, explorer_meta

    explorer_index = None
    explorer_meta = []

    if os.path.exists(EXPLORER_FAISS_PATH):
        os.remove(EXPLORER_FAISS_PATH)
    if os.path.exists(EXPLORER_META_PATH):
        os.remove(EXPLORER_META_PATH)

    return jsonify({"status": "cleared"})

@app.route("/heatmap", methods=["POST"])
def heatmap():
    texts = request.json.get("texts", [])

    if not texts:
        return jsonify({"error": "no texts"}), 400

    embeddings = []
    for t in texts:
        e = requests.post(
            "http://localhost:11434/api/embed",
            json={"model": EMBED_MODEL, "input": t}
        ).json()
        embeddings.append(np.array(e["embeddings"][0]))

    n = len(embeddings)
    matrix = [[0]*n for _ in range(n)]

    for i in range(n):
        for j in range(n):
            v1, v2 = embeddings[i], embeddings[j]
            sim = float(np.dot(v1, v2) / (np.linalg.norm(v1) * np.linalg.norm(v2)))
            matrix[i][j] = sim

    return jsonify({"texts": texts, "matrix": matrix})

@app.route("/faiss_index_info", methods=["GET"])
def faiss_index_info():
    """
    Returns FAISS metadata including:
    - total vectors indexed
    - embedding dimension
    - list of all PDF chunks with their doc_id + page numbers
    - breakdown per PDF
    """
    global faiss_index, faiss_meta, documents

    if faiss_index is None:
        return jsonify({
            "status": "empty",
            "total_vectors": 0,
            "dimension": 0,
            "vectors_per_pdf": {},
            "meta": []
        })

    # Count vectors per PDF
    breakdown = {}
    for m in faiss_meta:
        doc = m["doc_id"]
        breakdown[doc] = breakdown.get(doc, 0) + 1

    # FAISS dimension (safer than faiss_index.d)
    dim = faiss_index.d if hasattr(faiss_index, "d") else len(faiss_meta[0]["chunk"])

    return jsonify({
        "status": "ok",
        "total_vectors": len(faiss_meta),
        "dimension": dim,
        "vectors_per_pdf": breakdown,
        "meta": faiss_meta
    })


def load_explorer_index():
    global explorer_index, explorer_meta

    # Load metadata
    if os.path.exists(EXPLORER_META_PATH):
        import json
        with open(EXPLORER_META_PATH, "r") as f:
            explorer_meta = json.load(f)

    # Load FAISS
    if os.path.exists(EXPLORER_FAISS_PATH):
        try:
            explorer_index = faiss.read_index(EXPLORER_FAISS_PATH)
            print("📌 Explorer FAISS loaded with", explorer_index.ntotal, "vectors")
        except Exception as e:
            print("⚠️ Failed to load FAISS:", e)
            explorer_index = None


load_explorer_index()


if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000, debug=True)
