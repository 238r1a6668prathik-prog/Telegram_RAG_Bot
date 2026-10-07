from pathlib import Path
import pickle

import faiss
from sentence_transformers import SentenceTransformer

from document_loader import load_pdf, split_text


PDF_FOLDER = "documents"
VECTOR_DB_FOLDER = "vector_db"

INDEX_FILE = f"{VECTOR_DB_FOLDER}/index.faiss"
CHUNKS_FILE = f"{VECTOR_DB_FOLDER}/chunks.pkl"


# Load embedding model
embedding_model = SentenceTransformer("all-MiniLM-L6-v2")


def create_vector_database():

    pdf_files = list(Path(PDF_FOLDER).glob("*.pdf"))

    if not pdf_files:
        print("No PDF found in documents folder.")
        return

    all_chunks = []

    # Read all PDFs
    for pdf_file in pdf_files:

        print(f"Reading: {pdf_file.name}")

        text = load_pdf(pdf_file)

        chunks = split_text(text)

        all_chunks.extend(chunks)

    print(f"Total chunks: {len(all_chunks)}")

    # Create embeddings
    print("Creating embeddings...")

    embeddings = embedding_model.encode(
        all_chunks,
        convert_to_numpy=True
    )

    # Create FAISS index
    dimension = embeddings.shape[1]

    index = faiss.IndexFlatL2(dimension)

    index.add(embeddings)

    # Create vector_db folder
    Path(VECTOR_DB_FOLDER).mkdir(exist_ok=True)

    # Save FAISS index
    faiss.write_index(index, INDEX_FILE)

    # Save chunks
    with open(CHUNKS_FILE, "wb") as f:
        pickle.dump(all_chunks, f)

    print("\nVector database created successfully!")
    print(f"Vectors stored: {index.ntotal}")


if __name__ == "__main__":
    create_vector_database()