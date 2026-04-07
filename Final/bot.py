import telebot
from telebot import types
import pickle
import numpy as np
import sqlite3
import nltk
import difflib
import os
from dotenv import load_dotenv
from datetime import datetime
from nltk.stem.snowball import SnowballStemmer
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.sequence import pad_sequences

load_dotenv('telegram_token.env')
TOKEN = os.getenv('TOKEN')
bot = telebot.TeleBot(TOKEN)
stemmer = SnowballStemmer("russian")
max_length = 20
current_time = datetime.now().strftime('%Y-%m-%d %H:%M:%S')

def init_db():
    conn = sqlite3.connect('my_notes.db', check_same_thread=False)
    cursor = conn.cursor()
    cursor.execute('''
        CREATE TABLE IF NOT EXISTS notes (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            user_id INTEGER,
            text TEXT,
            tag TEXT,
            created_at DATETIME DEFAULT CURRENT_TIMESTAMP
        )
    ''')
    try:
        cursor.execute('ALTER TABLE notes ADD COLUMN created_at DATETIME DEFAULT CURRENT_TIMESTAMP')
    except:
        pass
    conn.commit()
    return conn, cursor


conn, cursor = init_db()

print("Загрузка нейросети")
model = load_model('gtd_model.h5')

with open('tokenizer.pickle', 'rb') as f:
    data = pickle.load(f)

    if isinstance(data, dict):
        tokenizer = data['tokenizer']
        all_roots = data['all_roots']
    else:

        tokenizer = data
        all_roots = []

tags_dict = {0: "#задачи", 1: "#покупки", 2: "#посмотреть_позже", 3: "#идеи"}


def preprocess_fuzzy(text):
    tokens = nltk.word_tokenize(text.lower())
    final_roots = []

    for word in tokens:
        root = stemmer.stem(word)

        if all_roots:
            matches = difflib.get_close_matches(root, all_roots, n=1, cutoff=0.8)
            if matches:
                final_roots.append(matches[0])
            else:
                final_roots.append(root)
        else:
            final_roots.append(root)

    return " ".join(final_roots)



def main_menu():
    markup = types.ReplyKeyboardMarkup(resize_keyboard=True)
    markup.row("Поиск по категориям", "Инструкция")
    return markup


def categories_menu():
    markup = types.InlineKeyboardMarkup()
    markup.row(types.InlineKeyboardButton("Задачи", callback_data="search_0"),
               types.InlineKeyboardButton("Покупки", callback_data="search_1"))
    markup.row(types.InlineKeyboardButton("Ссылки", callback_data="search_2"),
               types.InlineKeyboardButton("Идеи", callback_data="search_3"))
    return markup


@bot.message_handler(commands=['start'])
def start(message):
    bot.send_message(
        message.chat.id,
        "Привет! Я твой умный ассистент.\nПиши мне заметки, а я их рассортирую.\n",
        reply_markup=main_menu()
    )


@bot.message_handler(func=lambda m: m.text == "Поиск по категориям")
def search(message):
    bot.send_message(message.chat.id, "Выберите категорию для просмотра последних 10 записей:",
                     reply_markup=categories_menu())


@bot.message_handler(func=lambda m: m.text == "Инструкция")
def help(message):
    bot.send_message(message.chat.id,
                     "Для того,чтобы оставить заметку,нужно писать фразы по категориям: "
                     "1.Покупки, 2.Задачи, 3. Ссылки,4. Идеи. Примеры фраз: купить молоко, "
                     " посмотреть фильм 'Дюна', позвонить клиенту")


@bot.callback_query_handler(func=lambda call: call.data.startswith('search_'))
def callback_search(call):
    bot.answer_callback_query(call.id)
    class_idx = int(call.data.split('_')[1])
    tag = tags_dict[class_idx]

    cursor.execute('''
            SELECT text, strftime('%d.%m %H:%M', created_at, 'localtime') 
            FROM notes 
            WHERE user_id = ? AND tag = ? 
            ORDER BY id DESC LIMIT 10
        ''', (call.message.chat.id, tag))

    results = cursor.fetchall()

    if not results:
        bot.send_message(call.message.chat.id, f"В категории {tag} пока ничего нет.")
    else:
        results.reverse()

        text_resp = f"Последние записи в {tag}:\n" + "—" * 20 + "\n"
        for row in results:
            text_resp += f"[{row[1]}] {row[0]}\n"

        bot.send_message(call.message.chat.id, text_resp)


@bot.callback_query_handler(func=lambda call: call.data.startswith('search_'))
def callback_search(call):
    bot.answer_callback_query(call.id)
    class_idx = int(call.data.split('_')[1])
    tag = tags_dict[class_idx]

    conn = sqlite3.connect('my_notes.db', check_same_thread=False)
    cursor = conn.cursor()
    cursor.execute(
        "SELECT id, text, strftime('%d.%m %H:%M', created_at) FROM notes WHERE user_id=? AND tag=? ORDER BY id DESC LIMIT 10",
        (call.message.chat.id, tag))
    results = cursor.fetchall()
    conn.close()

    if not results:
        bot.send_message(call.message.chat.id, f"В категории {tag} пока ничего нет.")
    else:
        bot.send_message(call.message.chat.id, f"--- Записи в {tag} ---")
        for row in results:
            note_id = row[0]
            note_text = row[1]
            note_date = row[2]

            markup = types.InlineKeyboardMarkup()
            done_button = types.InlineKeyboardButton(text="✅ Выполнено / Удалить", callback_data=f"delete_{note_id}")
            markup.add(done_button)

            bot.send_message(call.message.chat.id, f"[{note_date}]\n{note_text}", reply_markup=markup)


@bot.callback_query_handler(func=lambda call: call.data.startswith('delete_'))
def callback_delete_note(call):
    note_id = int(call.data.split('_')[1])

    try:
        conn = sqlite3.connect('my_notes.db', check_same_thread=False)
        cursor = conn.cursor()
        cursor.execute("DELETE FROM notes WHERE id = ?", (note_id,))
        conn.commit()
        conn.close()

        bot.answer_callback_query(call.id, "Запись удалена!")
        bot.edit_message_text(chat_id=call.message.chat.id,
                              message_id=call.message.message_id,
                              text="✅ Выполнено и удалено из списка.")
    except Exception as e:
        bot.answer_callback_query(call.id, "Ошибка при удалении.")
        print(f"Error deleting: {e}")


@bot.message_handler(func=lambda m: True)
def handle_message(message):
    processed_text = preprocess_fuzzy(message.text)

    seq = tokenizer.texts_to_sequences([processed_text])
    padded = pad_sequences(seq, maxlen=max_length, padding='post')

    prediction = model.predict(padded)
    class_idx = np.argmax(prediction[0])
    confidence = prediction[0][class_idx] * 100
    tag = tags_dict[class_idx]

    cursor.execute(
        'INSERT INTO notes (user_id, text, tag, created_at) VALUES (?, ?, ?, ?)',
        (message.chat.id, message.text, tag, current_time)
    )
    conn.commit()

    bot.reply_to(
        message,
        f"Сохранено в {tag}\n"
        f"Уверенность: {confidence:.1f}%\n"
        f"Исправленный корень: {processed_text}"
    )


print("Бот запущен!")
bot.polling(none_stop=True)


