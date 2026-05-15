def evaluate_retrieval(client, collection, eval_questions):
    """
    Evaluate whether the correct disease appears in retrieved results.
    """
    correct = 0

    for item in eval_questions:
        question = item["question"]
        expected = item["expected"]

        query_response = client.embeddings.create(
            model="text-embedding-3-small",
            input=question
        )
        query_embedding = query_response.data[0].embedding

        results = collection.query(
            query_embeddings=[query_embedding],
            n_results=3
        )

        retrieved_diseases = [
            meta["disease"] for meta in results["metadatas"][0]
        ]

        if expected in retrieved_diseases:
            correct += 1

        print(f"Q: {question}")
        print(f"Expected: {expected}")
        print(f"Retrieved: {retrieved_diseases}")
        print("-" * 40)

    accuracy = correct / len(eval_questions)
    return accuracy