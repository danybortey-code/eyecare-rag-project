from src.data_loader import load_documents
from src.chunker import create_chunks


def main():
    documents = load_documents()
    chunked_documents = create_chunks(documents)

    print("EyeCareRAG Modular Pipeline")
    print("-" * 40)
    print(f"Total documents loaded: {len(documents)}")
    print(f"Total chunks created: {len(chunked_documents)}")

    print("\nModules included:")
    print("- data_loader.py: loads medical documents")
    print("- chunker.py: splits documents into chunks")
    print("- vector_store.py: handles ChromaDB vector storage")
    print("- retriever.py: retrieves relevant chunks")
    print("- generator.py: generates grounded answers")
    print("- evaluator.py: evaluates retrieval accuracy")

    print("\nStatus:")
    print("Code refactored into reusable modules.")
    print("OpenAI calls are implemented in modules but not executed in this demo script.")


if __name__ == "__main__":
    main()