import subprocess


def generate_answer(question, retrieved_results):
    """
    Generate a grounded answer using Ollama (llama3.2).
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
- Give a clear and concise answer.
- Use only the provided context.
- If the answer is not in the context, say so.
- Do not provide a diagnosis.
"""

    result = subprocess.run(
        ["ollama", "run", "llama3.2", prompt],
        capture_output=True,
        text=True,
        encoding="utf-8"
    )

    return result.stdout.strip()