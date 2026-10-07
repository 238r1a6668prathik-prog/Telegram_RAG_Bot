from pathlib import Path
from pypdf import PdfReader


def load_pdf(file_path):
    reader = PdfReader(file_path)

    text = ""

    for page in reader.pages:
        page_text = page.extract_text()

        if page_text:
            text += page_text + "\n"

    return text


def split_text(text, chunk_size=500, overlap=50):
    chunks = []

    start = 0

    while start < len(text):
        end = start + chunk_size

        chunk = text[start:end]
        chunks.append(chunk)

        start += chunk_size - overlap

    return chunks


if __name__ == "__main__":

    pdf_files = list(Path("documents").glob("*.pdf"))

    if not pdf_files:
        print("No PDF found in documents folder.")
    else:
        pdf_path = pdf_files[0]

        text = load_pdf(pdf_path)

        chunks = split_text(text)

        print("PDF:", pdf_path.name)
        print("Characters:", len(text))
        print("Number of chunks:", len(chunks))

        print("\nFirst chunk:\n")
        print(chunks[0])