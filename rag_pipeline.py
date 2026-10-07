import faiss
import pickle
import ollama

from sentence_transformers import SentenceTransformer


INDEX_FILE = "vector_db/index.faiss"
CHUNKS_FILE = "vector_db/chunks.pkl"


# Load embedding model
embedding_model = SentenceTransformer("all-MiniLM-L6-v2")


# Load FAISS index
index = faiss.read_index(INDEX_FILE)


# Load document chunks
with open(CHUNKS_FILE, "rb") as f:
    chunks = pickle.load(f)


def retrieve_documents(query, top_k=3):

    query_embedding = embedding_model.encode(
        [query],
        convert_to_numpy=True
    )

    distances, indices = index.search(
        query_embedding,
        top_k
    )

    results = []

    for i in indices[0]:

        if i < len(chunks):
            results.append(chunks[i])

    return results


def generate_answer(question, context):

    prompt = f"""
You are a helpful AI assistant.

Answer the user's question using ONLY the information
provided in the context below.

If the answer cannot be found in the context,
say: "I couldn't find this information in the provided document."

Do not make up information.

Context:
{context}

User Question:
{question}

Answer:
"""

    response = ollama.chat(
        model="qwen2.5-coder:3b",
        messages=[
            {
                "role": "user",
                "content": prompt
            }
        ]
    )

    return response["message"]["content"]


if __name__ == "__main__":

    question = input("Ask a question: ")

    # Retrieve relevant chunks
    results = retrieve_documents(question)

    # Combine chunks
    context = "\n\n".join(results)

    # Generate answer
    answer = generate_answer(question, context)

    print("\nAI Answer:")
    print(answer)