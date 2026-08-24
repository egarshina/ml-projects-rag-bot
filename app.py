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
    "О себе: Robotics Engineer | ML/CV Engineer. Опыт 1.5 года CV engineer в НИИ робототехники, работала с нейросетями для анализа аэрофотоснимков, включая pix2pix и CycleGAN. Навыки: Python, PyTorch, scikit-learn, OpenCV, C++ (академические знания), ROS (академические знания), YOLO (академические знания). Английский B1, немецкий A2-B1, японский базовый. Контакты: egarshina21@gmail.com, Telegram @roboticsR",
    "Проект iris-clustering: кластеризация датасета Iris методами ML без учителя. Использованы KMeans, иерархическая кластеризация, DBSCAN, GMM. Метрики: ARI ≈ 0.73, Silhouette ≈ 0.50. Оптимальное число кластеров — 3, что совпадает с реальными видами ириса",
    "Проект fashion-mnist-classification: классификация изображений одежды (Fashion MNIST) с помощью свёрточной нейросети (CNN) на PyTorch. Архитектура: 2 свёрточных слоя, пулинг, полносвязные слои. Test Accuracy 91.3%",
    "Проект titanic-decision-tree: предсказание выживаемости пассажиров Титаника с использованием Decision Tree, Random Forest и Logistic Regression. Подбор гиперпараметров через GridSearchCV. Оценка точности и ROC-AUC",
    "Проект ml-projects-rag-bot: RAG-ассистент, который отвечает на вопросы о ML-проектах Екатерины. Использует эмбеддинги SentenceTransformer, векторный индекс FAISS и GigaChat для генерации ответов. Развёрнут как веб-интерфейс (Streamlit)"
]

# 2. Загрузка модели и индекса
@st.cache_resource
def load_embedder_and_index():
    embedder = SentenceTransformer('paraphrase-multilingual-MiniLM-L12-v2')
    embeddings = embedder.encode(documents, show_progress_bar=True)
    dimension = embeddings.shape[1]
    index = faiss.IndexFlatL2(dimension)
    index.add(embeddings.astype('float32'))
    return embedder, index

embedder, index = load_embedder_and_index()

def search(query, top_k=3):  
    query_vec = embedder.encode([query])
    distances, indices = index.search(query_vec.astype('float32'), top_k)
    return [documents[i] for i in indices[0]]

# 3. Подключение к GigaChat
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

# 4. Rate limiting
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

# 5. Интерфейс
st.title("ML Project RAG Assistant")
st.write("Задайте вопрос о моих ML-проектах.")

question = st.text_input("Ваш вопрос:")

if question:
    import streamlit.runtime.scriptrunner as scriptrunner
    ctx = scriptrunner.get_script_run_ctx()
    ip = ctx.session_id  # для демо используем session_id вместо IP

    if not is_allowed(ip):
        st.warning("Слишком много запросов. Подождите минуту.")
    else:
        # Поиск контекста
        contexts = search(question, top_k=len(documents))
        context_text = "\n".join(contexts)

        # Промпт для LLM
        prompt = f"""Ты — ассистент, отвечающий на вопросы о портфолио Екатерины.
Используй только предоставленный контекст, не добавляй ничего от себя.
Если вопрос касается проектов, перечисли все проекты из контекста.
Если вопрос о навыках, опыте, контактах — используй документ "О себе".
Отвечай в третьем лице (она/её), ведь ты говоришь о проектах Екатерины. Можно писать "В проекте", "В данном проекте" и похожим образом.
Контекст:
{context_text}

Вопрос: {question}
Ответ:"""

        # Запрос к GigaChat
        answer = ask_gigachat(prompt)

        # Вывод ответа и источников
        st.write("### Ответ:")
        st.write(answer)
        with st.expander("Показать источники"):
            for c in contexts:
                st.write("-", c)
