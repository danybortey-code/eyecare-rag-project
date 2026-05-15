import chromadb


def get_collection(db_path="chroma_db", collection_name="eye_care_rag"):
    """
    Connect to persistent ChromaDB collection.
    """
    chroma_client = chromadb.PersistentClient(path=db_path)
    collection = chroma_client.get_or_create_collection(name=collection_name)
    return collection


def build_vector_store(client, chunked_documents, collection):
    """
    Create embeddings and store them in ChromaDB.
    """
    for i, doc in enumerate(chunked_documents):
        response = client.embeddings.create(
            model="text-embedding-3-small",
            input=doc["text"]
        )

        embedding = response.data[0].embedding

        collection.upsert(
            ids=[str(i)],
            embeddings=[embedding],
            documents=[doc["text"]],
            metadatas=[{
                "disease": doc["disease"],
                "source_file": doc["source_file"],
                "chunk_id": doc["chunk_id"]
            }]
        )

    return collection