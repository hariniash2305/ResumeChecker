# ResumeChecker
---

# AI-Powered Bulk Resume Screener

An intelligent resume screening system that ranks and evaluates multiple candidate resumes against a job description using semantic search and Large Language Models (LLMs).

---

## Features

* Bulk PDF resume upload
* Semantic embedding generation using Sentence Transformers (all-MiniLM-L6-v2)
* Fast vector similarity search using FAISS
* AI-based candidate evaluation using Google Gemini
* Similarity score ranking
* Interactive Streamlit dashboard

---

## Architecture Overview

1. Extract text from uploaded PDF resumes
2. Generate embeddings for each resume
3. Store embeddings in a FAISS vector index
4. Convert the job description into an embedding
5. Perform a similarity search to rank candidates
6. Use Gemini LLM to generate structured evaluations for top candidates

---

## Tech Stack

* Python
* Streamlit
* FAISS
* Sentence Transformers
* Google Gemini API
* NumPy
* PDFPlumber

---

## Installation

```bash
git clone https://github.com/your-username/your-repo-name.git
cd your-repo-name

python -m venv venv
venv\Scripts\activate   # Windows
# source venv/bin/activate  # Mac/Linux

pip install -r requirements.txt
```

---

## Environment Setup

Create a `.env` file in the root directory:

```
GEMINI_API_KEY=AIzaSyAePqac--HDXaqxv9Hph7w2S1HQ9_8PlOc
```

## Running the Application

```bash
streamlit run app.py
```
## Use Case
This project assists recruiters in automating bulk resume screening by:
* Reducing manual shortlisting time
* Improving candidate-role matching accuracy
* Providing AI-generated hiring insights

