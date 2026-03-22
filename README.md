# EyeCareRAG: Clinical Decision Support Chatbot for Eye Diseases

## Project Overview
EyeCareRAG is a Retrieval-Augmented Generation (RAG) chatbot built to answer questions about common eye diseases using trusted medical sources. The system focuses on four eye conditions:

- Glaucoma
- Cataract
- Age-related macular degeneration (AMD)
- Dry eye

Instead of answering from memory alone, the chatbot retrieves relevant information from medical source documents and then generates a grounded response.

## Motivation
Many AI chatbots can produce health-related answers without clearly showing reliable evidence. This project was built to create a more trustworthy clinical decision support assistant by combining retrieval, source grounding, and evaluation.

## Data Sources
The knowledge base was built from trusted public health and medical resources:

- National Eye Institute (NEI)
- MedlinePlus
- PubMed abstracts

For each disease, the project collects:
- `nei.txt`
- `medline.txt`
- `pubmed.txt`

## System Architecture
The project follows a RAG pipeline:

1. Collect disease-specific documents from trusted health sources
2. Split documents into smaller text chunks
3. Convert chunks into embeddings using OpenAI embeddings
4. Store embeddings in a persistent ChromaDB vector database
5. Embed a user query
6. Retrieve the most relevant chunks
7. Use the retrieved context to generate a grounded answer

## Technologies Used
- Python
- OpenAI API
- ChromaDB
- Requests
- BeautifulSoup
- Git / GitHub
- VS Code

## Project Structure
```text
eyecare-rag-project/
│
├── data/
│   ├── glaucoma/
│   ├── cataract/
│   ├── amd/
│   └── dry_eye/
│
├── chroma_db/        # local persistent vector database (ignored in Git)
├── main.py
├── README.md
└── .gitignore