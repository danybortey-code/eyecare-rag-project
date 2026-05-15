def chunk_text(text, chunk_size=800, overlap=100):
    """
    Split text into overlapping chunks.
    """
    chunks = []
    start = 0

    while start < len(text):
        end = start + chunk_size
        chunk = text[start:end]
        chunks.append(chunk)
        start += chunk_size - overlap

    return chunks


def create_chunks(documents, chunk_size=800, overlap=100):
    """
    Convert full documents into smaller chunks for retrieval.
    """
    chunked_documents = []

    for doc in documents:
        chunks = chunk_text(doc["text"], chunk_size, overlap)

        for i, chunk in enumerate(chunks):
            chunked_documents.append({
                "disease": doc["disease"],
                "source_file": doc["source_file"],
                "chunk_id": i,
                "text": chunk
            })

    return chunked_documents