def generate_answer(client, question, retrieved_results):
    """
    Generate a grounded answer using retrieved context.
    """
    retrieved_context = "\n\n".join(retrieved_results["documents"][0])

    prompt = f"""
You are a helpful clinical decision support assistant.
Answer the user's question using only the context below.

Question:
{question}

Context:
{retrieved_context}

Instructions:
- Give a clear and short answer
- Use only the provided context
- If the answer is not in the context, say so
- Do not provide a diagnosis
"""

    response = client.chat.completions.create(
        model="gpt-4.1-mini",
        messages=[
            {"role": "user", "content": prompt}
        ]
    )

    return response.choices[0].message.content