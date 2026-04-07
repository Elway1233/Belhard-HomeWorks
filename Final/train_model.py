import numpy as np
import pickle
import nltk
import difflib
from nltk.stem.snowball import SnowballStemmer
import tensorflow as tf
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Embedding, LSTM, Dense, Dropout, GlobalAveragePooling1D

stemmer = SnowballStemmer("russian")

data_groups = {
    0: ["позвонить", "написать", "сделать", "записаться", "отправить", "убраться", "починить","задача ремонт", "сделать ремонт", "задача по дому",
        "выполнить задачу", "написать отчет", "позвонить маме",
        "забрать посылку", "задача проект", "ремонт квартиры задача",
        "задача", "срочная задача", "список задач"],
    1: ["купить", "заказать", "взять", "докупить", "приобрести", "семена", "продукты", "молоко","купить запчасти", "оплатить ремонт", "купить семена",
        "заказать еду", "купить молоко", "взять хлеб",
        "покупка материалов", "купить краску для ремонта",
        "чек за ремонт", "стоимость ремонта"],
    2: ["http", "https", "youtube", "сайт", "ссылка", "статья", "видео", "посмотреть","фильм","посмотреть видео про ремонт"],
    3: ["идея", "придумал", "мысль", "проект", "стартап", "концепт","дизайн ремонта идея","создать"]
}

raw_texts = []
labels = []
all_roots = []

for label_id, phrases in data_groups.items():
    for phrase in phrases:
        raw_texts.append(phrase)
        labels.append(label_id)
        for word in nltk.word_tokenize(phrase.lower()):
            all_roots.append(stemmer.stem(word))

all_roots = list(set(all_roots))

def get_clean_text(text):
    words = nltk.word_tokenize(text.lower())
    clean_words = []
    for w in words:
        root = stemmer.stem(w)
        matches = difflib.get_close_matches(root, all_roots, n=1, cutoff=0.8)
        if matches:
            clean_words.append(matches[0])
        else:
            clean_words.append(root)
    return " ".join(clean_words)

processed_texts = [get_clean_text(t) for t in raw_texts]

tokenizer = Tokenizer(num_words=1000, oov_token="<OOV>")
tokenizer.fit_on_texts(processed_texts)
sequences = tokenizer.texts_to_sequences(processed_texts)
padded = pad_sequences(sequences, maxlen=20, padding='post')

model = Sequential([
    Embedding(1000, 32),
    LSTM(64, return_sequences=True),
    GlobalAveragePooling1D(),
    Dense(32, activation='relu'),
    Dense(4, activation='softmax')
])

model.compile(loss='sparse_categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
model.fit(padded, np.array(labels), epochs=200, verbose=1)

model.save('gtd_model.h5')
with open('tokenizer.pickle', 'wb') as f:
    pickle.dump({'tokenizer': tokenizer, 'all_roots': all_roots}, f)

print("Модель обучена!")

