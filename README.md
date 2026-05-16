# 👁️ EyeCareRAG

### A Retrieval-Augmented Generation (RAG) Clinical Assistant for Eye Disease Education and Triage

> A healthcare-focused AI system that retrieves information from trusted medical sources, grounds responses in evidence-based context, and generates safe, educational answers to common eye disease questions.

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
10. Evaluation
11. Example Questions
12. Results
13. Installation
14. Environment Setup
15. Running the Project
16. Limitations
17. Future Work
18. Learning Outcomes
19. Author
20. Disclaimer

---

## Overview

EyeCareRAG is a Retrieval-Augmented Generation (RAG) system designed to answer educational questions about common eye diseases using trusted medical sources.

The system focuses on four major eye conditions:

* Glaucoma
* Cataract
* Age-related macular degeneration (AMD)
* Dry eye disease

Instead of relying solely on a language model's internal knowledge, EyeCareRAG retrieves relevant text from curated medical documents and uses that context to generate grounded responses.

---

## Clinical Motivation

Patients frequently search online for information about eye diseases, but generic AI tools may generate responses that are inaccurate, outdated, or unsupported by medical evidence.

This project demonstrates how Retrieval-Augmented Generation can improve reliability by ensuring responses are grounded in trusted sources such as:

* National Eye Institute (NEI)
* MedlinePlus
* PubMed abstracts

---

## Problem Statement

Develop an AI system that can:

1. Answer questions about common eye diseases.
2. Retrieve information from trusted medical references.
3. Generate grounded educational responses.
4. Reduce unsupported hallucinations.
5. Measure retrieval performance using evaluation metrics.

---

## System Architecture

```text
User Question
      ↓
Query Embedding (OpenAI Embeddings)
      ↓
ChromaDB Vector Database
      ↓
Top Relevant Chunks Retrieved
      ↓
Prompt Construction
      ↓
OpenAI LLM (GPT-4.1-mini)
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

All text files are loaded into Python.

### Step 3 — Chunking

Documents are split into overlapping chunks.

### Step 4 — Embedding Generation

Each chunk is converted into a vector representation using OpenAI embeddings.

### Step 5 — Vector Storage

Embeddings are stored in ChromaDB.

### Step 6 — Retrieval

A user question is embedded and matched to the most relevant chunks.

### Step 7 — Answer Generation

The retrieved context is provided to an LLM to generate a grounded answer.

### Step 8 — Evaluation

Retrieval accuracy is measured using a custom evaluation set.

---

## Data Sources

For each disease, the following source files are collected:

* `nei.txt`
* `medline.txt`
* `pubmed.txt`

### Trusted Medical Sources

* National Eye Institute (NEI)
* MedlinePlus
* PubMed

---

## Key Features

* Retrieval-Augmented Generation (RAG)
* Trusted medical knowledge base
* OpenAI embeddings
* ChromaDB vector search
* LLM-generated grounded responses
* Custom evaluation framework
* Modular code architecture
* Professional GitHub documentation

---

## Tech Stack

| Component       | Technology                |
| --------------- | ------------------------- |
| Language Model  | OpenAI GPT-4.1-mini       |
| Embeddings      | text-embedding-3-small    |
| Vector Database | ChromaDB                  |
| Data Collection | Requests + BeautifulSoup  |
| Evaluation      | Custom retrieval accuracy |
| Language        | Python                    |
| IDE             | VS Code                   |
| Version Control | Git & GitHub              |

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

## Evaluation

### Custom Evaluation Questions

1. What are the early symptoms of glaucoma?
2. How is cataract treated?
3. What is AMD?
4. What causes dry eye?

### Evaluation Metric

**Retrieval Accuracy**: Measures whether the correct disease appears in the retrieved results.

---

## Results

| Metric             |    Value |
| ------------------ | -------: |
| Retrieval Accuracy | **1.00** |

The system successfully retrieved the correct disease context for all evaluation questions.

---

## Example Questions

* What are the early symptoms of glaucoma?
* How is cataract treated?
* What causes dry eye?
* What is age-related macular degeneration?

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

Create a `.env` file in the project root:

```text
OPENAI_API_KEY=your_api_key_here
```

---

## Running the Project

```bash
python main.py
```

---

## Limitations

* Limited to four eye diseases
* Small evaluation dataset
* Intended for educational use only
* Requires OpenAI API access for full pipeline execution
* No graphical user interface in the current version

---

## Future Work

* Streamlit web application
* Additional eye diseases
* Larger and more diverse evaluation dataset
* Source citations in final responses
* Multimodal support (OCT, visual fields, PDFs)
* Clinical-grade security and authentication

---

## Learning Outcomes

This project provided hands-on experience with:

* Retrieval-Augmented Generation (RAG)
* OpenAI embeddings and LLM APIs
* ChromaDB vector databases
* Medical data curation
* Evaluation methodology
* Modular software design
* Git and GitHub workflows

---


---

## Disclaimer

This project is intended solely for educational purposes and is not a substitute for professional medical advice, diagnosis, or treatment.
