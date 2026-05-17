from src.data_loader import load_documents
from src.chunker import create_chunks

# Optional imports for the full pipeline
# from src.vector_store import get_collection, build_vector_store
# from src.retriever import retrieve_context
# from src.generator import generate_answer
# from src.evaluator import evaluate_retrieval


def main():
    # Load source documents
    documents = load_documents()

    # Create chunks
    chunked_documents = create_chunks(documents)

    print("EyeCareRAG Modular Pipeline")
    print("-" * 40)
    print(f"Total documents loaded: {len(documents)}")
    print(f"Total chunks created: {len(chunked_documents)}")

    print("\nModules included:")
    print("- data_loader.py: loads medical documents")
    print("- chunker.py: splits documents into chunks")
    print("- vector_store.py: stores embeddings in ChromaDB")
    print("- retriever.py: retrieves relevant chunks")
    print("- generator.py: generates answers with Ollama (llama3.2)")
    print("- evaluator.py: evaluates retrieval accuracy")

    print("\nLLM Backend:")
    print("Ollama (llama3.2)")

    print("\nStatus:")
    print("Project refactored into reusable modules.")
    print("No OpenAI API key required.")
    print("Ready for local RAG execution with Ollama.")

    # Example full pipeline (uncomment when you want to run it)
    #
    # collection = get_collection()
    #
    # if collection.count() == 0:
    #     build_vector_store(chunked_documents, collection)
    #
    # question = "What are the early symptoms of glaucoma?"
    # results = retrieve_context(collection, question)
    # answer = generate_answer(question, results)
    #
    # print("\nQuestion:", question)
    # print("\nAnswer:", answer)


if __name__ == "__main__":
    main()