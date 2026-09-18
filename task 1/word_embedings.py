from gensim.models import Word2Vec
from sklearn.manifold import TSNE
from sklearn.metrics.pairwise import cosine_distances
import matplotlib.pyplot as plt
import random
import numpy as np
import re

def train_model():
    with open("src/lenin_pss_tomy_01-20.txt", "r", encoding="utf-8") as f:
        text = f.read()
    sentences_text = re.split(r"[.!?]+", text)
    sentences = []
    for sentence in sentences_text:
        sentence = re.sub(r"[^а-яё\s]", " ", sentence)
        sentence = re.sub(r"\s+", " ", sentence).strip()
        if sentence:
            sentences.append(sentence.split())
    model = Word2Vec(
        sentences=sentences,
        vector_size=100,
        window=5,
        min_count=5,
        workers=6,
        sg=1
    )
    model.save("lenin_word2vecUPD.model")

    return model


def read_model():
    return Word2Vec.load("lenin_word2vecUPD.model")


def trainTSNE(vectors):
    tsne = TSNE(
        n_components=2,
        random_state=42,
        perplexity=30
    )
    vectors_2d = tsne.fit_transform(vectors)
    np.save("tsne_vectors30.npy", vectors_2d)

    return vectors_2d


def readTSNE():
    return np.load("tsne_vectors30.npy")


# Обучение/чтение
model = read_model()
words = model.wv.index_to_key
vectors = model.wv.vectors
vectors_2d = readTSNE()
# vectors_2d = trainTSNE(vectors)
print(f"Всего слов: {len(words)}")

# Отображение точек
plt.figure(figsize=(12, 10))
plt.scatter(
    vectors_2d[:, 0],
    vectors_2d[:, 1],
    s=5
)

# Подпись случайных точек
SMPLS_CNT = 20
sampling_step = int(np.floor(len(words)/SMPLS_CNT))
sampled_words = words[::sampling_step]
samples_num = [i * sampling_step for i in range(len(sampled_words))]
for i, word in zip(samples_num, sampled_words):
    plt.annotate(
        word,
        (vectors_2d[i, 0], vectors_2d[i, 1]),
        fontsize=10
    )

# Выделяем класс: месяцы
months = [
    "январь",
    "февраль",
    "март",
    "апрель",
    "май",
    "июнь",
    "июль",
    "август",
    "сентябрь",
    "октябрь"
    "ноябрь",
    "декабрь",
]
months_in_model = [
    word for word in months
    if word in model.wv
]
print("Месяцы в модели:")
print(months_in_model)

# Сопоставляем месяцам индекс, вектор
word_to_index = {
    word: i
    for i, word in enumerate(words)
}
month_indices = [
    word_to_index[word]
    for word in months_in_model
]
month_vectors_2d = vectors_2d[month_indices]

# Отмечаем точки-месяцы
plt.scatter(
    month_vectors_2d[:, 0],
    month_vectors_2d[:, 1],
    s=30,
    label="Месяцы"
)
for word, index in zip(months_in_model, month_indices):
    plt.annotate(
        word,
        (vectors_2d[index, 0], vectors_2d[index, 1]),
        fontsize=10
    )
plt.grid(True)
plt.savefig("tsne_map30.png", dpi=300, bbox_inches="tight")
plt.show()

# Расстояние между месяцами
month_vectors = model.wv[months_in_model]
month_distances = cosine_distances(month_vectors)
month_pair_distances = month_distances[
    np.triu_indices(len(months_in_model), k=1)
]
month_class = month_pair_distances.mean()
print("Среднее расстояние между месяцами:", month_class)

# Для расчета расстояния между точками рассмотрим случайную выборку
np.random.seed(42)
sample_size = 15000      # ~40% всех слов
sample_indices = np.random.choice(
    len(words),
    size=sample_size,
    replace=False
)
sample_vectors = vectors[sample_indices]

# Расстояние между точками
all_distances = cosine_distances(sample_vectors)
all_pair_distances = all_distances[
    np.triu_indices(sample_size, k=1)
]
all_class = all_pair_distances.mean()
print("Среднее расстояние между словами корпуса:", all_class)

# Итоговое отношение
R = month_class / all_class
print("R =", R)
