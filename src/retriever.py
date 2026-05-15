def retrieve_context(client, collection, query, n_results=3):
    """
    Retrieve the most relevant chunks for a user query.
    """
    query_response = client.embeddings.create(
        model="text-embedding-3-small",
        input=query
    )

    query_embedding = query_response.data[0].embedding

    results = collection.query(
        query_embeddings=[query_embedding],
        n_results=n_results
    )

    return results