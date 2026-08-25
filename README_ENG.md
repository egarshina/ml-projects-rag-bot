# ML Projects RAG Assistant

A RAG assistant that answers questions about my ML projects, skills, and contacts.

## Description
A web interface built with Streamlit that lets users ask questions about my portfolio. The assistant uses Retrieval-Augmented Generation: first, relevant documents are found using SentenceTransformer embeddings and a FAISS vector index, then the retrieved fragments are passed to GigaChat, which generates a context-based answer.

## Repository structure
- `app.py` — main application: document loading, index creation, GigaChat integration, and a Streamlit UI with rate limiting.
- `requirements.txt` — project dependencies.
- `README.md` — project description in Russian (as the main language).
- `README_ENG.md` — project description in English, since the rest of the files are in English. For consistency and fairness.

## Technologies used
- Python 3
- SentenceTransformers
- FAISS
- Streamlit
- GigaChat API

## How to run
1. Clone the repository.
2. Install dependencies: `pip install -r requirements.txt`.
3. Set the `GIGACHAT_CREDENTIALS` environment variable.
4. Run: `streamlit run app.py`.

Alternatively, deploy it on Streamlit Cloud and add the `GIGACHAT_CREDENTIALS` secret in the app settings.

## Results
- The bot lists all projects from the portfolio. (For now, the list is stored directly in `app.py`; later I plan to add automatic reading of all readme files from my profile, including new ones.)
- It answers questions about skills, experience, and contacts.
- Rate limiting is built in to prevent abuse.
- It shows the sources used to generate each answer. (For now, these are short descriptions in `app.py`; after the commit with automatic readme reading, it will link to the actual files.)

## Conclusions
The project demonstrates a practical use of RAG: combining vector search with a generative model to create an assistant for a personal portfolio. Adding new projects does not require retraining the model — it is enough to update the document list (until the automatic readme reading is implemented).
