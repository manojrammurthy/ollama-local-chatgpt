from flask import Flask, request, Response, jsonify, render_template
from flask_cors import CORS
import requests
import json
import fitz
import numpy as np
import faiss

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

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000, debug=True)
