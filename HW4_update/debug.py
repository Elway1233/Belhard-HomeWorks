import telebot
import torch
import json
import random
from model import NeuralNet, DeepNeuralNet
from nltk_utils import bag_of_words, tokenize

TOKEN = 'Your token'
bot = telebot.TeleBot(TOKEN)
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
with open('intents.json', 'r', encoding='utf-8') as json_data:
    intents = json.load(json_data)

# Загрузка модели
data = torch.load("data.pth", map_location=device)
input_size = data["input_size"]
hidden_size = data["hidden_size"]
output_size = data["output_size"]
all_words = data['all_words']
tags = data['tags']
model_state = data["model_state"]

model = DeepNeuralNet(input_size, hidden_size, output_size).to(device)
model.load_state_dict(model_state)
model.eval()


@bot.message_handler(commands=['start'])
def send_welcome(message):
    bot.send_message(message.chat.id, "Бот готов. Напиши что угодно, чтобы проверить классификацию.")


@bot.message_handler(func=lambda message: True)
def handle_message(message):
    sentence_tokenized = tokenize(message.text)
    X = bag_of_words(sentence_tokenized, all_words)
    X = X.reshape(1, X.shape[0])
    X = torch.from_numpy(X).to(device)

    output = model(X)
    probs = torch.softmax(output, dim=1)

    confidence, predicted = torch.max(probs, dim=1)
    tag = tags[predicted.item()]
    conf_value = confidence.item()

    selected_response = "Ответ не найден"
    for intent in intents['intents']:
        if intent["tag"] == tag:
            selected_response = random.choice(intent['responses'])
            break

    debug_text = (f"Предсказание: *{tag}*\n"
                  f"Уверенность: *{conf_value:.4f}*\n"
                  f"Ответ бота: _{selected_response}_\n"
                  f"Текст: {message.text}")

    bot.send_message(message.chat.id, debug_text, parse_mode="Markdown")

if __name__ == '__main__':
    bot.polling(none_stop=True)
