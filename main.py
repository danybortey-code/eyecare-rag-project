from openai import OpenAI
import chromadb

client = OpenAI()

chroma_client = chromadb.PersistentClient(path="chroma_db")
collection = chroma_client.get_collection(name="eye_care_rag")

query = "What are the early symptoms of glaucoma?"

query_response = client.embeddings.create(
    model="text-embedding-3-small",
    input=query
)

query_embedding = query_response.data[0].embedding

results = collection.query(
    query_embeddings=[query_embedding],
    n_results=3
)

retrieved_context = "\n\n".join(results["documents"][0])

prompt = f"""
You are a helpful clinical decision support assistant.
Answer the user's question using only the context below.

Question:
{query}

Context:
{retrieved_context}

Instructions:
- Give a clear and short answer
- Use only the provided context
- If the answer is not in the context, say so
"""

response = client.chat.completions.create(
    model="gpt-4.1-mini",
    messages=[
        {"role": "user", "content": prompt}
    ]
)

print(response.choices[0].message.content)
eval_questions = [
    {"question": "What are the early symptoms of glaucoma?", "expected": "glaucoma"},
    {"question": "How is cataract treated?", "expected": "cataract"},
    {"question": "What is AMD?", "expected": "amd"},
    {"question": "What causes dry eye?", "expected": "dry_eye"}
]

print("Eval set ready:", len(eval_questions))
correct = 0

for item in eval_questions:
    question = item["question"]
    expected = item["expected"]

    # Embed question
    query_response = client.embeddings.create(
        model="text-embedding-3-small",
        input=question
    )
    query_embedding = query_response.data[0].embedding

    # Retrieve
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
print(f"Retrieval Accuracy: {accuracy:.2f}")