Задача: Вариационный автоэнкодер для генерации новых изображений лиц: Реализация вариационного автоэнкодера (VAE), способного генерировать реалистичные изображения лиц на основе обучающего набора данных.

## Инструкция по запуску:
1. Скачайте датасет ([https://www.kaggle.com/datasets/jessicali9530/lfw-dataset](https://www.kaggle.com/datasets/matheuseduardo/flickr-faces-dataset-resized?resource=download-directory&select=256x256](https://www.kaggle.com/datasets/matheuseduardo/flickr-faces-dataset-resized?resource=download-directory&select=64x64))
](https://www.kaggle.com/datasets/matheuseduardo/flickr-faces-dataset-resized?resource=download-directory&select=64x64))
2. Зайти на Google colab, создать новый блокнот,вставить весь код.
3. Загрузить датасет, и в строчке full_dataset = datasets.ImageFolder(root='/content/drive/MyDrive/dataset/', transform=train_transform), укажите свой путь(Обязательно после dataset/ должна быть еще подпапка в которой лежат сами фотографии,т.е полный путь /content/drive/MyDrive/dataset/faces/00001.png)
4. Запустить проект.

Результат:
Программа создат 32 несуществующих лица,на основе полученных фотографий,а так же сравнит некоторые результаты с оригиналом.
