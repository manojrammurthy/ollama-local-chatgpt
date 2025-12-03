from flask import Flask, request, Response, jsonify, render_template
from flask_cors import CORS
import requests
import json
import fitz
import numpy as np
import faiss
   # Compute PCA (2D reduction)
import numpy as np
from sklearn.decomposition import PCA

app = Flask(__name__)
CORS(app)

messages = []
MODEL = "phi3"
EMBED_MODEL = "nomic-embed-text"

vector_index = None
documents = []

@app.route("/")
def home():
    return render_template("index.html")

@app.route("/clear", methods=["POST"])
def clear():
    global messages
    messages = []
    return jsonify({"status": "cleared"})

@app.route("/models", methods=["GET"])
def models():
    try:
        response = requests.get("http://localhost:11434/api/tags")
        data = response.json()
        names = [m["name"] for m in data.get("models", [])]
        return jsonify({"models": names})
    except:
        return jsonify({"models": []})

@app.route("/set_model", methods=["POST"])
def set_model():
    global MODEL, messages
    MODEL = request.json.get("model", MODEL)
    messages = []
    return jsonify({"status": "ok", "model": MODEL})

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
        response = requests.post("http://localhost:11434/api/chat", json=payload, stream=True)

        for line in response.iter_lines():
            if line:
                data = json.loads(line.decode())
                delta = data.get("message", {}).get("content", "")
                assistant_reply += delta
                yield delta

        messages.append({"role": "assistant", "content": assistant_reply})

    return Response(generate(), content_type="text/plain")

@app.route("/upload_pdf", methods=["POST"])
def upload_pdf():
    global vector_index, documents

    print("FILES RECEIVED:", request.files)

    if 'file' not in request.files:
        return jsonify({"error": "No file part"}), 400

    file = request.files['file']
    if file.filename == "":
        return jsonify({"error": "No file selected"}), 400

    try:
        pdf = fitz.open(stream=file.read(), filetype="pdf")
    except:
        return jsonify({"error": "Invalid PDF"}), 400

    text = ""
    for page in pdf:
        text += page.get_text()

    # Chunk
    words = text.split()
    chunks = [" ".join(words[i:i+200]) for i in range(0, len(words), 200)]

    documents = chunks

    vectors = []
    for chunk in chunks:
        emb = requests.post(
            "http://localhost:11434/api/embed",
            json={"model": EMBED_MODEL, "input": chunk}
        ).json()

        # Properly check inside loop
        if "embeddings" not in emb:
            print("Embedding error:", emb)  # Log error
            continue  # Skip this chunk safely

        vectors.append(emb["embeddings"][0])



    vectors = np.array(vectors, dtype="float32")

    dim = vectors.shape[1]
    vector_index = faiss.IndexFlatL2(dim)
    vector_index.add(vectors)

    return jsonify({"chunks": len(chunks)})

@app.route("/ask_pdf", methods=["POST"])
def ask_pdf():
    global vector_index, documents, MODEL

    if vector_index is None:
        return jsonify({"error": "No PDF uploaded"}), 400

    question = request.json.get("message")

    qvec = requests.post(
        "http://localhost:11434/api/embed",
        json={"model": EMBED_MODEL, "input": question}
    ).json()
    
    if "embeddings" not in emb_q:
     return jsonify({"error": "Embedding failed", "details": emb_q}), 400

    qvec = emb_q["embeddings"][0]

    qvec = np.array(qvec, dtype="float32").reshape(1, -1)

    distances, indices = vector_index.search(qvec, 3)

    context = "\n".join([documents[i] for i in indices[0]])

    rag_prompt = f"""
Use the below context to answer the question.

Context:
{context}

Question: {question}

Answer concisely:
"""

    messages.append({"role": "user", "content": rag_prompt})

    payload = {"model": MODEL, "messages": messages, "stream": True}

    def generate():
        assistant_reply = ""
        response = requests.post("http://localhost:11434/api/chat", json=payload, stream=True)

        for line in response.iter_lines():
            if line:
                data = json.loads(line.decode())
                delta = data.get("message", {}).get("content", "")
                assistant_reply += delta
                yield delta

        messages.append({"role": "assistant", "content": assistant_reply})

    return Response(generate(), content_type="text/plain")

# @app.route("/embed_text", methods=["POST"])
# def embed_text():
#     text = request.json.get("text", "").strip()
#     if not text:
#         return jsonify({"error": "empty text"}), 400

#     # Call Ollama embedding endpoint
#     response = requests.post(
#         "http://localhost:11434/api/embed",
#         json={"model": EMBED_MODEL, "input": text}
#     ).json()

#     if "embeddings" not in response:
#         return jsonify({"error": "embedding_failed", "details": response}), 400

#     vector = response["embeddings"][0]

 
#     arr = np.array(vector).reshape(1, -1)

#     # # PCA requires at least 2 samples — workaround: duplicate vector
#     # if arr.shape[0] == 1:
#     #     arr_for_pca = np.vstack([arr, arr])  # duplicate
#     # else:
#     #     arr_for_pca = arr

#     # pca = PCA(n_components=2)
#     # reduced = pca.fit_transform(arr_for_pca)[0].tolist()

#     if arr.shape[0] == 1:
#         jitter = np.random.normal(0, 0.0001, size=arr.shape)
#         arr_for_pca = np.vstack([arr, arr + jitter])
#     else:
#         arr_for_pca = arr

    # return jsonify({
    #     "vector": vector,
    #     "vector_preview": vector[:30],  # show only first 30 dims
    #     "dimension": len(vector),
    #     "pca": reduced
    # })

@app.route("/embed_text", methods=["POST"])
def embed_text():
    text = request.json.get("text", "").strip()
    if not text:
        return jsonify({"error": "empty text"}), 400

    # Call Ollama embedding API
    response = requests.post(
        "http://localhost:11434/api/embed",
        json={"model": EMBED_MODEL, "input": text}
    ).json()

    if "embeddings" not in response:
        return jsonify({"error": "embedding_failed", "details": response}), 400

    vector = response["embeddings"][0]

    # Convert to numpy
    import numpy as np
    arr = np.array(vector).reshape(1, -1)

    # PCA reduction (handle 1-sample case)
    from sklearn.decomposition import PCA

    if arr.shape[0] == 1:
        # Duplicate vector to satisfy PCA requirement
        arr_for_pca = np.vstack([arr, arr + np.random.normal(0, 1e-6, arr.shape)])
    else:
        arr_for_pca = arr

    pca = PCA(n_components=2)
    pca_result = pca.fit_transform(arr_for_pca)

    # Use only the first point (actual vector)
    reduced = pca_result[0].tolist()

    return jsonify({
        "vector": vector,
        "vector_preview": vector[:30],
        "dimension": len(vector),
        "pca": reduced
    })
@app.route("/compare_embeddings", methods=["POST"])
def compare_embeddings():
    data = request.json
    text1 = data.get("text1", "").strip()
    text2 = data.get("text2", "").strip()

    if not text1 or not text2:
        return jsonify({"error": "Both text fields are required"}), 400

    # Call embedding API for both
    emb1 = requests.post(
        "http://localhost:11434/api/embed",
        json={"model": EMBED_MODEL, "input": text1}
    ).json()

    emb2 = requests.post(
        "http://localhost:11434/api/embed",
        json={"model": EMBED_MODEL, "input": text2}
    ).json()

    if "embeddings" not in emb1 or "embeddings" not in emb2:
        return jsonify({"error": "embedding_failed"}), 400

    v1 = np.array(emb1["embeddings"][0])
    v2 = np.array(emb2["embeddings"][0])

    # Cosine similarity
    cos_sim = float(np.dot(v1, v2) / (np.linalg.norm(v1) * np.linalg.norm(v2)))

    # L2 distance
    l2_dist = float(np.linalg.norm(v1 - v2))

    # Dot product
    dot_prod = float(np.dot(v1, v2))

    # PCA Visualisation
    matrix = np.vstack([v1, v2])

    from sklearn.decomposition import PCA
    pca = PCA(n_components=2)
    pca_points = pca.fit_transform(matrix).tolist()  # [[x1,y1], [x2,y2]]
    diff = (v1 - v2).tolist()

    return jsonify({
        "cosine_similarity": cos_sim,
        "l2_distance": l2_dist,
        "dot_product": dot_prod,
        "pca_points": pca_points,
        "diff_vector": diff
    })


@app.route("/embedding_explorer")
def embedding_explorer():
    return render_template("embedding_explorer.html")


if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000, debug=True)
