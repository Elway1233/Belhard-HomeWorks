import telebot
from telebot import types
import torch
import json
import random
import logging
import datetime

from model import NeuralNet, DeepNeuralNet
from nltk_utils import bag_of_words, tokenize

logger = logging.getLogger()
logger.setLevel(logging.INFO)
file_handler = logging.FileHandler('telegram_chat.log', encoding='utf-8')
formatter = logging.Formatter('%(asctime)s | %(message)s')
file_handler.setFormatter(formatter)
logger.addHandler(file_handler)



TOKEN = 'Your token'
bot = telebot.TeleBot(TOKEN)

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
with open('intents.json', 'r', encoding='utf-8') as json_data:
    intents = json.load(json_data)

data = torch.load("data.pth")
input_size = data["input_size"]
hidden_size = data["hidden_size"]
output_size = data["output_size"]
all_words = data['all_words']
tags = data['tags']
model_state = data["model_state"]
model_type = data.get("model_type", "basic")

if model_type == 'basic':
    model = NeuralNet(input_size, hidden_size, output_size).to(device)
elif model_type == 'deep':
    model = DeepNeuralNet(input_size, hidden_size, output_size).to(device)

model.load_state_dict(model_state)
model.eval()

user_data = {}


def get_user_data(chat_id):
    if chat_id not in user_data:
        user_data[chat_id] = {'state': None, 'cart': {}}
    return user_data[chat_id]


def get_main_menu():
    markup = types.ReplyKeyboardMarkup(resize_keyboard=True, row_width=2)
    markup.add("Ассортимент 💐", "Корзина 🛒", "Доставка 🚚", "Как оплатить? 💳", "Связаться с человеком 👨‍💻")
    return markup


def get_seasons_menu():
    markup = types.ReplyKeyboardMarkup(resize_keyboard=True, row_width=2)
    markup.add("Весна 🌷", "Лето 🌻", "Осень 🍁", "Зима ❄️", "Главное меню 🔙")
    return markup


def get_buy_keyboard(item_id, item_name):
    markup = types.InlineKeyboardMarkup()
    btn = types.InlineKeyboardButton(text=f"Добавить {item_name} в корзину ➕", callback_data=f"add_{item_id}")
    markup.add(btn)
    return markup


def get_cart_keyboard():
    markup = types.InlineKeyboardMarkup(row_width=1)
    btn_clear = types.InlineKeyboardButton(text="Очистить корзину 🗑", callback_data="clear_cart")
    btn_checkout = types.InlineKeyboardButton(text="Оформить заказ ✅", callback_data="checkout")
    markup.add(btn_checkout, btn_clear)
    return markup


def get_current_season():
    month = datetime.datetime.now().month
    if month in [12, 1, 2]:
        return 'зима'
    elif month in [3, 4, 5]:
        return 'весна'
    elif month in [6, 7, 8]:
        return 'лето'
    else:
        return 'осень'


@bot.callback_query_handler(func=lambda call: True)
def handle_callback(call):
    chat_id = call.message.chat.id
    u_data = get_user_data(chat_id)

    if call.data.startswith('add_'):
        item_code = call.data.split('_')[1]
        items_dict = {
            'rose': 'Розы', 'peony': 'Пионы', 'tulip': 'Тюльпаны',
            'chrys': 'Хризантемы', 'orchid': 'Орхидеи', 'alstr': 'Альстромерии',
            'daisy': 'Ромашки', 'sun': 'Подсолнухи', 'carn': 'Гвоздики',
            'aster': 'Астры', 'narc': 'Нарциссы/Гиацинты', 'field': 'Полевой микс',
            'fir': 'Букет с елью', 'custom': 'Авторский букет'
        }
        item_name = items_dict.get(item_code, 'Цветы')

        u_data['cart'][item_name] = u_data['cart'].get(item_name, 0) + 1

        bot.answer_callback_query(call.id, f"{item_name} добавлены в корзину!")
        bot.send_message(chat_id,
                         f"✅ {item_name} добавлены в корзину. Вы можете перейти в 'Корзину 🛒' для оформления заказа.")
        logging.info(f"Chat: {chat_id} | Added to cart: {item_name}")

    elif call.data == 'clear_cart':
        u_data['cart'] = {}
        bot.answer_callback_query(call.id, "Корзина очищена")
        bot.edit_message_text("Ваша корзина пуста 😔", chat_id, call.message.message_id)

    elif call.data == 'checkout':
        u_data['cart'] = {}
        bot.answer_callback_query(call.id, "Заказ оформлен!")
        bot.edit_message_text(
            "🎉 Спасибо за заказ! Наш менеджер скоро свяжется с вами для уточнения адреса доставки и времени.", chat_id,
            call.message.message_id)
        logging.info(f"Chat: {chat_id} | Checkout complete")


@bot.message_handler(commands=['start'])
def send_welcome(message):
    chat_id = message.chat.id
    get_user_data(chat_id)['state'] = None
    welcome_text = "Здравствуйте! Я виртуальный помощник цветочной лавки. Выберите интересующий вас раздел на клавиатуре ниже или напишите мне!"
    bot.send_message(chat_id, welcome_text, reply_markup=get_main_menu())


@bot.message_handler(func=lambda message: True)
def handle_message(message):
    chat_id = message.chat.id
    text = message.text
    u_data = get_user_data(chat_id)

    if text == "Главное меню 🔙":
        u_data['state'] = None
        bot.send_message(chat_id, "Вы вернулись в главное меню.", reply_markup=get_main_menu())
        return

    sentence_tokenized = tokenize(text)
    X = bag_of_words(sentence_tokenized, all_words)
    X = X.reshape(1, X.shape[0])
    X = torch.from_numpy(X).to(device)

    output = model(X)
    _, predicted = torch.max(output, dim=1)
    tag = tags[predicted.item()]

    probs = torch.softmax(output, dim=1)
    confidence = probs[0][predicted.item()].item()

    if confidence > 0.65:

        if tag == 'view_cart':
            cart = u_data['cart']
            if not cart:
                bot.send_message(chat_id, "Ваша корзина пуста 😔. Посмотрите наш ассортимент!")
            else:
                cart_text = "🛒 **Ваш заказ:**\n\n"
                for item, count in cart.items():
                    cart_text += f"🔸 {item}: {count} букет(ов)\n"
                bot.send_message(chat_id, cart_text, parse_mode="Markdown", reply_markup=get_cart_keyboard())
            u_data['state'] = None
            return

        for intent in intents['intents']:
            if tag == intent["tag"]:
                response = random.choice(intent['responses'])
                if tag == 'assortment':
                    u_data['state'] = 'waiting_for_season_answer'
                else:
                    u_data['state'] = None
                break

        if tag in ['season_spring', 'season_summer', 'season_autumn', 'season_winter']:
            bot.send_message(chat_id, response, reply_markup=get_main_menu())
        else:
            bot.send_message(chat_id, response)

    else:
        u_data['state'] = None
        bot.send_message(chat_id, "Извините, я не понимаю... Воспользуйтесь кнопками.", reply_markup=get_main_menu())

    logging.info(
        f"Chat: {chat_id} | "
        f"In: '{text}' | "
        f"Tag: {tag} | "
        f"Conf: {confidence:.2f} | "
        f"Out: '{response}'"
    )


if __name__ == '__main__':
    print("Бот запущен в Telegram! Нажмите Ctrl+C для остановки.")
    bot.polling(none_stop=True)
