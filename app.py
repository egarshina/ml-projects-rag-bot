import os
import time
from collections import defaultdict

import faiss
import numpy as np
import streamlit as st
from sentence_transformers import SentenceTransformer
from gigachat import GigaChat
from gigachat.models import Chat, Messages, MessagesRole

documents = [
    "Проект iris-clustering: кластеризация датасета Iris с помощью KMeans, иерархической кластеризации, DBSCAN и GMM. Использовались метрики ARI и Silhouette.",
    "Проект fashion-mnist-classification: классификация изображений одежды с помощью свёрточной нейросети на PyTorch. Test Accuracy 91.3%.",
    "Проект titanic-decision-tree: предсказание выживаемости пассажиров Титаника с использованием Decision Tree, Random Forest и Logistic Regression. Подбор гиперпараметров через GridSearchCV."
]

@st.cache_resource
def load_embedder_and_index():
    embedder = SentenceTransformer('paraphrase-multilingual-MiniLM-L12-v2')
    embeddings = embedder.encode(documents, show_progress_bar=True)
    dimension = embeddings.shape[1]
    index = faiss.IndexFlatL2(dimension)
    index.add(embeddings.astype('float32'))
    return embedder, index

embedder, index = load_embedder_and_index()

def search(query, top_k=2):
    query_vec = embedder.encode([query])
    distances, indices = index.search(query_vec.astype('float32'), top_k)
    return [documents[i] for i in indices[0]]

GIGACHAT_CREDENTIALS = os.environ.get("GIGACHAT_CREDENTIALS")
if not GIGACHAT_CREDENTIALS:
    st.error("Не задан секрет GIGACHAT_CREDENTIALS")
    st.stop()

client = GigaChat(
    base_url="https://api.giga.chat/v1",
    credentials=GIGACHAT_CREDENTIALS,
    scope="GIGACHAT_API_PERS",
    verify_ssl_certs=False,
)

def ask_gigachat(prompt):
    chat = Chat(
        model="GigaChat-3-Ultra",
        messages=[Messages(role=MessagesRole.USER, content=prompt)],
        temperature=0.3,
        max_tokens=200,
    )
    resp = client.chat(chat)
    return resp.choices[0].message.content

request_log = defaultdict(list)
MAX_REQUESTS = 5
TIME_WINDOW = 60

def is_allowed(ip):
    now = time.time()
    request_log[ip] = [t for t in request_log[ip] if now - t < TIME_WINDOW]
    if len(request_log[ip]) >= MAX_REQUESTS:
        return False
    request_log[ip].append(now)
    return True

st.title("ML Project RAG Assistant")
st.write("Задайте вопрос о моих ML-проектах.")

question = st.text_input("Ваш вопрос:")

if question:
    import streamlit.runtime.scriptrunner as scriptrunner
    ctx = scriptrunner.get_script_run_ctx()
    ip = ctx.session_id  # замена IP на session_id для демо
    if not is_allowed(ip):
        st.warning("Слишком много запросов. Подождите минуту.")
    else:
        contexts = search(question, top_k=2)
        context_text = "\n".join(contexts)
        prompt = f"Используй только следующий контекст, чтобы ответить на вопрос.\nКонтекст:\n{context_text}\n\nВопрос: {question}\nОтвет:"
        answer = ask_gigachat(prompt)
        st.write("### Ответ:")
        st.write(answer)
        with st.expander("Показать источники"):
            for c in contexts:
                st.write("-", c)