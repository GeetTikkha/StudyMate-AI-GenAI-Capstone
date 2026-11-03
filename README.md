# StudyMate AI — GenAI Capstone Project

A student-focused GenAI application combining LLM chat, RAG, embeddings, structured quiz generation, function calling, an AI agent workflow and study task management.

## Features
- AI study chat using an LLM API.
- Chat with PDF, DOCX and TXT documents using RAG.
- Text chunking, embeddings and cosine-similarity retrieval.
- Multiple-choice quiz generation with answer keys and explanations.
- Function calling: restricted calculator, add task and list tasks.
- SQLite study task planner.
- Inspect retrieved passages and agent tool calls.

## Stack
Python · Streamlit · OpenAI Chat Completions API · OpenAI Embeddings · NumPy · SQLite · pypdf · python-docx

## Architecture
Student → Streamlit UI → (AI Chat / Document RAG / Quiz Generator / Task Planner)
Document RAG: extraction → chunking → embeddings → similarity retrieval → LLM context → answer
Agent: user request → LLM tool selection → calculator or SQLite task tool → tool result → final response

## Run locally
1. Install Python 3.10+.
2. `pip install -r requirements.txt`
3. Copy `.env.example` to `.env` and add your own API key.
4. `streamlit run app.py`

API usage may incur charges. Never commit `.env` or share your API key.

## Deploy on Streamlit Community Cloud
1. Push this repository to GitHub.
2. Sign in to Streamlit Community Cloud and create an app from this repository, selecting `app.py`.
3. In App Settings → Secrets, add:
   `OPENAI_API_KEY = "your_key_here"`
   `OPENAI_MODEL = "gpt-4o-mini"`
   `OPENAI_EMBEDDING_MODEL = "text-embedding-3-small"`
4. Deploy and test each feature. The generated Streamlit URL is your live website.

## Demo steps
1. Upload a small text-based PDF/DOCX/TXT and process it.
2. Ask a question answered by the uploaded document; inspect retrieved passages.
3. Generate a quiz for the same topic and check answers.
4. Ask the assistant to calculate `(125 * 4) + 30`.
5. Ask it to add a study task and view it in the Task Planner.

## Responsible AI and limitations
LLM answers and quiz keys can be wrong; verify them. RAG may retrieve irrelevant or incomplete passages. Similarity scores are not confidence probabilities. Scanned PDFs may need OCR. Uploaded text is sent to the configured embedding API. SQLite storage on hosted instances may be temporary across restarts. Add authentication, managed persistent storage, OCR and systematic evaluation before production use.
