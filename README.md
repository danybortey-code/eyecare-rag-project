# 👁️ EyeCareRAG

### A Retrieval-Augmented Generation (RAG) Clinical Assistant for Eye Disease Education and Triage

> A healthcare-focused AI system that retrieves information from trusted medical sources, grounds responses in evidence-based context, and generates safe, educational answers to common eye disease questions using a modular RAG pipeline and local LLM support through Ollama.

---

## Table of Contents

1. Overview
2. Clinical Motivation
3. Problem Statement
4. System Architecture
5. How It Works
6. Data Sources
7. Key Features
8. Tech Stack
9. Project Structure
10. Module Overview
11. Evaluation
12. Example Questions
13. Results
14. Installation
15. Environment Setup
16. Running the Project
17. Limitations
18. Future Work
19. Learning Outcomes
20. Author
21. Disclaimer

---

## Overview

EyeCareRAG is a Retrieval-Augmented Generation (RAG) system designed to answer educational questions about common eye diseases using trusted medical sources.

The system focuses on four major eye conditions:

- Glaucoma
- Cataract
- Age-related macular degeneration (AMD)
- Dry eye disease

Instead of relying solely on a language model's internal knowledge, EyeCareRAG retrieves relevant text from curated medical documents and uses that context to generate grounded responses.

The project was refactored from a single script into a modular architecture, separating data loading, chunking, vector storage, retrieval, answer generation, and evaluation into dedicated modules.

---

## Clinical Motivation

Patients frequently search online for information about eye diseases, but generic AI tools may generate responses that are inaccurate, outdated, or unsupported by medical evidence.

This project demonstrates how Retrieval-Augmented Generation can improve reliability by ensuring responses are grounded in trusted sources such as:

- National Eye Institute (NEI)
- MedlinePlus
- PubMed abstracts

---

## Problem Statement

Develop an AI system that can:

1. Answer questions about common eye diseases.
2. Retrieve information from trusted medical references.
3. Generate grounded educational responses.
4. Reduce unsupported hallucinations.
5. Measure retrieval performance using evaluation metrics.
6. Use modular software design for maintainability and scalability.

---

## System Architecture

```text
User Question
      ↓
Query Embedding
      ↓
ChromaDB Vector Database
      ↓
Top Relevant Chunks Retrieved
      ↓
Prompt Construction
      ↓
Ollama Local LLM (llama3.2)
      ↓
Grounded Educational Answer
      ↓
Evaluation Metrics
```

---

## How It Works

### Step 1 — Data Collection

Medical content is collected from NEI, MedlinePlus, and PubMed for each disease.

### Step 2 — Document Loading

All text files are loaded into Python using the `data_loader.py` module.

### Step 3 — Chunking

Documents are split into overlapping chunks using the `chunker.py` module.

### Step 4 — Embedding Generation

Each chunk is converted into a vector representation for similarity search.

### Step 5 — Vector Storage

Embeddings are stored in a persistent ChromaDB vector database.

### Step 6 — Retrieval

A user question is embedded and matched to the most relevant chunks.

### Step 7 — Answer Generation

The retrieved context is passed to Ollama (`llama3.2`) to generate a grounded educational answer.

### Step 8 — Evaluation

Retrieval accuracy is measured using a custom evaluation set.

---

## Data Sources

For each disease, the following source files are collected:

- `nei.txt`
- `medline.txt`
- `pubmed.txt`

### Trusted Medical Sources

- National Eye Institute (NEI)
- MedlinePlus
- PubMed

---

## Key Features

- Retrieval-Augmented Generation (RAG)
- Curated eye disease medical corpus
- Trusted medical knowledge base
- ChromaDB vector search
- Local LLM inference using Ollama (`llama3.2`)
- Grounded educational responses
- Custom evaluation framework
- Modular code architecture
- No OpenAI API key required for local answer generation
- Professional GitHub documentation

---

## Tech Stack

| Component | Technology |
|----------|----------|
| Language Model | Ollama (`llama3.2`) |
| Vector Database | ChromaDB |
| Retrieval Method | Embedding-based similarity search |
| Data Collection | Requests + BeautifulSoup |
| Evaluation | Custom retrieval accuracy |
| Language | Python |
| IDE | VS Code |
| Version Control | Git & GitHub |

---

## Project Structure

```text
eyecare-rag-project/
├── data/
│   ├── glaucoma/
│   ├── cataract/
│   ├── amd/
│   └── dry_eye/
│
├── src/
│   ├── data_loader.py
│   ├── chunker.py
│   ├── vector_store.py
│   ├── retriever.py
│   ├── generator.py
│   └── evaluator.py
│
├── chroma_db/          # Local vector database (not tracked in Git)
├── main.py             # Pipeline orchestration
├── README.md
├── requirements.txt
└── .gitignore
```

---

## Module Overview

| Module | Purpose |
|------|------|
| `data_loader.py` | Loads disease-specific text files |
| `chunker.py` | Splits documents into overlapping chunks |
| `vector_store.py` | Handles ChromaDB vector storage |
| `retriever.py` | Retrieves relevant chunks for a user query |
| `generator.py` | Generates grounded answers using Ollama |
| `evaluator.py` | Evaluates retrieval accuracy |

---

## Evaluation

### Custom Evaluation Questions

1. What are the early symptoms of glaucoma?
2. How is cataract treated?
3. What is AMD?
4. What causes dry eye?

### Evaluation Metric

**Retrieval Accuracy** measures whether the correct disease appears in the retrieved results.

This metric was chosen because retrieval is the most critical component of a RAG system. If the wrong context is retrieved, the final answer may also be incorrect.

---

## Example Questions

- What are the early symptoms of glaucoma?
- How is cataract treated?
- What causes dry eye?
- What is age-related macular degeneration?

---

## Results

| Metric | Value |
|------|------:|
| Retrieval Accuracy | **1.00** |

The system successfully retrieved the correct disease context for all evaluation questions.

---

## Example Output

**Question:** What are the early symptoms of glaucoma?

**Answer:** Early symptoms of glaucoma are often absent, meaning many patients notice no symptoms until peripheral vision loss begins.

---

## Installation

```bash
git clone https://github.com/danybortey-code/eyecare-rag-project.git
cd eyecare-rag-project
python -m venv venv
venv\Scripts\activate
pip install -r requirements.txt
```

---

## Environment Setup

### Install Ollama

Download and install Ollama from:

```text
https://ollama.com
```

### Download the Model

```bash
ollama pull llama3.2
```

### Optional API Key

If you choose to use cloud-based embeddings, create a `.env` file:

```text
OPENAI_API_KEY=your_api_key_here
```

---

## Running the Project

```bash
python main.py
```

The current `main.py` demonstrates the modular pipeline by loading documents, creating chunks, and confirming that all reusable RAG modules are available.

---

## Limitations

- Limited to four eye diseases
- Small evaluation dataset
- Intended for educational use only
- Requires Ollama to be installed locally
- No graphical user interface in the current version
- Retrieval accuracy was tested on a small custom evaluation set

---

## Future Work

- Streamlit web application
- Additional eye diseases
- Larger and more diverse evaluation dataset
- Source citations in final responses
- Fully local embeddings
- Multimodal support for OCT, visual fields, and PDFs
- Clinical-grade security and authentication
- Advanced evaluation metrics such as faithfulness and hallucination scoring

---

## Learning Outcomes

This project provided hands-on experience with:

- Retrieval-Augmented Generation (RAG)
- Local LLM deployment with Ollama
- ChromaDB vector databases
- Medical data curation
- Evaluation methodology
- Modular software design
- Git and GitHub workflows
- Building healthcare-focused AI systems safely

---

## Author

**Daniel Bortey, OD**  
MS in Data Science Candidate, University of Connecticut

- GitHub: https://github.com/danybortey-code
- Project Repository: https://github.com/danybortey-code/eyecare-rag-project

---

## Disclaimer

This project is intended solely for educational purposes and is not a substitute for professional medical advice, diagnosis, or treatment.