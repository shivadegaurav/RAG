from flask import Flask, request, jsonify
import requests
import numpy as np
import faiss
import os
import uuid
import json
from werkzeug.utils import secure_filename
import time

app = Flask(__name__)

# Constants
OLLAMA_BASE_URL = "http://localhost:11434/api"
EMBEDDING_MODEL = "nomic-embed-text"  # or "bge-base-en"
LLM_MODEL = "llama3.2"
EMBEDDING_DIM = 768  # Embedding dimension (adjust based on model)
CHUNK_SIZE = 512  # Token chunk size; reduce if memory issues persist
CHUNK_OVERLAP = 100  # Chunk overlap size
MAX_CONTEXT_CHUNKS = 5  # Maximum chunks to include in context

# Storage paths
DATA_DIR = "data"
FAISS_INDICES_DIR = os.path.join(DATA_DIR, "indices")
DOCUMENT_INFO_DIR = os.path.join(DATA_DIR, "document_info")

# Create directories if they don't exist
os.makedirs(FAISS_INDICES_DIR, exist_ok=True)
os.makedirs(DOCUMENT_INFO_DIR, exist_ok=True)

# In-memory document database for simplicity
documents = {}


def get_embedding(text):
    """Get text embedding from Ollama."""
    response = requests.post(
        f"{OLLAMA_BASE_URL}/embeddings",
        json={"model": EMBEDDING_MODEL, "prompt": text}
    )
    if response.status_code != 200:
        raise Exception(f"Failed to get embedding: {response.text}")
    return response.json()["embedding"]


def generate_text(prompt):
    """Generate text using Ollama LLM."""
    response = requests.post(
        f"{OLLAMA_BASE_URL}/generate",
        json={"model": LLM_MODEL, "prompt": prompt, "stream": False}
    )
    if response.status_code != 200:
        raise Exception(f"Failed to generate text: {response.text}")
    return response.json()["response"]


def chunk_text(text, chunk_size=CHUNK_SIZE, overlap=CHUNK_OVERLAP):
    """Split text into overlapping chunks based on whitespace tokens."""
    tokens = text.split()
    chunks = []
    for i in range(0, len(tokens), chunk_size - overlap):
        chunk = " ".join(tokens[i:i + chunk_size])
        chunks.append(chunk)
        if i + chunk_size >= len(tokens):
            break
    return chunks


def save_document_info(doc_id, document_info):
    """Save document metadata to disk."""
    with open(os.path.join(DOCUMENT_INFO_DIR, f"{doc_id}.json"), "w") as f:
        json.dump(document_info, f)


def load_document_info(doc_id):
    """Load document metadata from disk."""
    try:
        with open(os.path.join(DOCUMENT_INFO_DIR, f"{doc_id}.json"), "r") as f:
            return json.load(f)
    except FileNotFoundError:
        return None


def load_all_documents():
    """Load all document info from disk."""
    global documents
    documents = {}
    for filename in os.listdir(DOCUMENT_INFO_DIR):
        if filename.endswith(".json"):
            doc_id = filename.replace(".json", "")
            doc_info = load_document_info(doc_id)
            if doc_info:
                documents[doc_id] = doc_info


@app.route('/process', methods=['POST'])
def process_document():
    """Process document text, generate embeddings, and save to FAISS index."""
    data = request.json
    text = data.get('text')
    filename = data.get('filename', 'unknown.pdf')

    if not text:
        return jsonify({"error": "No text provided"}), 400

    # Generate unique document ID
    doc_id = str(uuid.uuid4())

    # Chunk the document
    chunks = chunk_text(text)

    # Prepare document metadata
    document_info = {
        "id": doc_id,
        "filename": filename,
        "chunk_count": 0,  # will update after successful embeddings
        "created_at": time.time(),
        "chunks": []
    }

    # Create FAISS index
    index = faiss.IndexFlatIP(EMBEDDING_DIM)  # Using inner product similarity

    successful_chunks = []

    for i, chunk in enumerate(chunks):
        try:
            embedding = get_embedding(chunk)
            embedding_array = np.array([embedding], dtype='float32')
            index.add(embedding_array)
            successful_chunks.append({
                "id": i,
                "text": chunk,
                "length": len(chunk)
            })
        except Exception as e:
            print(f"Error embedding chunk {i}: {e}")

    document_info["chunk_count"] = len(successful_chunks)
    document_info["chunks"] = successful_chunks

    # Save the FAISS index
    faiss.write_index(index, os.path.join(FAISS_INDICES_DIR, f"{doc_id}.index"))

    # Save document info
    save_document_info(doc_id, document_info)

    # Update in-memory document database
    documents[doc_id] = document_info

    return jsonify({
        "success": True,
        "document_id": doc_id,
        "chunk_count": len(successful_chunks)
    })


@app.route('/retrieve', methods=['POST'])
def retrieve_context():
    """Retrieve relevant document chunks based on query."""
    data = request.json
    query = data.get('query')
    doc_id = data.get('document_id')

    if not query:
        return jsonify({"error": "No query provided"}), 400

    if not doc_id:
        return jsonify({"error": "No document ID provided"}), 400

    # Load document info
    document_info = load_document_info(doc_id)
    if not document_info:
        return jsonify({"error": f"Document {doc_id} not found"}), 404

    # Load FAISS index
    index_path = os.path.join(FAISS_INDICES_DIR, f"{doc_id}.index")
    if not os.path.exists(index_path):
        return jsonify({"error": f"Index for document {doc_id} not found"}), 404

    index = faiss.read_index(index_path)

    # Get query embedding
    try:
        query_embedding = get_embedding(query)
    except Exception as e:
        return jsonify({"error": f"Failed to get embedding for query: {str(e)}"}), 500

    query_embedding_array = np.array([query_embedding], dtype='float32')

    # Determine how many context chunks to retrieve
    k = min(MAX_CONTEXT_CHUNKS, len(document_info["chunks"]))
    scores, indices = index.search(query_embedding_array, k)

    # Prepare context chunks
    contexts = []
    for i, idx in enumerate(indices[0]):
        if idx < len(document_info["chunks"]):
            chunk = document_info["chunks"][int(idx)]
            contexts.append({
                "text": chunk["text"],
                "score": float(scores[0][i]),
                "chunk_id": int(idx)
            })

    return jsonify({
        "query": query,
        "document_id": doc_id,
        "contexts": contexts
    })


@app.route('/generate', methods=['POST'])
def generate_response():
    """Generate response based on query and context."""
    data = request.json
    query = data.get('query')
    contexts = data.get('contexts', [])

    if not query:
        return jsonify({"error": "No query provided"}), 400

    # Construct prompt with context
    context_text = "\n\n".join([f"Context {i + 1}:\n{ctx['text']}" for i, ctx in enumerate(contexts)])
    prompt = f"""You are an assistant answering questions based on the provided document contexts. 
Use only the information in the contexts to answer the question.
If the answer cannot be found in the contexts, say so clearly.

CONTEXTS:
{context_text}

QUESTION: {query}

ANSWER:"""

    try:
        response = generate_text(prompt)
        return jsonify({"response": response})
    except Exception as e:
        return jsonify({"error": f"Failed to generate response: {str(e)}"}), 500


@app.route('/documents', methods=['GET'])
def list_documents():
    """List all available documents."""
    load_all_documents()

    doc_list = []
    for doc_id, info in documents.items():
        doc_list.append({
            "id": doc_id,
            "filename": info.get("filename", "unknown.pdf"),
            "chunk_count": info.get("chunk_count", 0),
            "created_at": info.get("created_at", 0)
        })

    return jsonify({"documents": doc_list})


@app.route('/documents/<doc_id>', methods=['DELETE'])
def delete_document(doc_id):
    """Delete a document and its index."""
    doc_info_path = os.path.join(DOCUMENT_INFO_DIR, f"{doc_id}.json")
    index_path = os.path.join(FAISS_INDICES_DIR, f"{doc_id}.index")

    if not os.path.exists(doc_info_path):
        return jsonify({"error": f"Document {doc_id} not found"}), 404

    try:
        if os.path.exists(doc_info_path):
            os.remove(doc_info_path)
        if os.path.exists(index_path):
            os.remove(index_path)

        if doc_id in documents:
            del documents[doc_id]

        return jsonify({"success": True, "message": f"Document {doc_id} deleted"})
    except Exception as e:
        return jsonify({"error": f"Failed to delete document: {str(e)}"}), 500


if __name__ == '__main__':
    load_all_documents()
    app.run(host='0.0.0.0', port=5000)
