# Анализ эмоционального окраса сообщения
![image](https://user-images.githubusercontent.com/47640060/133846881-23e266d7-f2e3-4304-8817-1758e59cc10b.png)

Сайт проекта: http://pochtabank-case.std-1370.ist.mospolytech.ru/

## Структура
### [Сверточная нейронная сеть](ru.py)
Сверточная нейронная сеть для обучения модели

### [Обученная модель](/ru_model)
Обученная модель, используемая для оценки сообщений

### [Фронтенд](https://github.com/TeamBolognese/PochtaBank-Beseda-FINOdays/tree/front)
Фронтенд часть сервера

### [Веб-сервер на bottle](bottle_app.py)
Веб-сервер на bottle, связывающий модель и фронтенд

## Описание

### Используемые датасеты:
1. [ImdbRu](https://disk.yandex.ru/d/nBbRyRfdX8S2eA) - Нашей командой машинно-переведенные на русский язык 50 тысяч отзывов Imdb
2. [Кинопоиск](https://disk.yandex.ru/d/EjANVCwooJyf6w) - 28 тысяч отзывов с кинотеатра Кинопоиск

## Инструкции по использованию

Для запуска бекенда необходимо:
```
sudo apt install python3.9
sudo apt install python3-pip
pip3 install bottle
pip3 install spacy==3.1.0
pip3 install pymorphy2
python3 bottle_app.py

Наличие файла bottle_app.py и папки ru_model в рабочем каталоге Python
```

Проект уже развернут и готова к использованию по адресу: http://pochtabank-case.std-1370.ist.mospolytech.ru/

Для обучения модели: 
```
python3 -m spacy download ru_core_news_sm
pip3 install spacy-lookups-data
```

Для переобучения модели: 
```
python3 -m spacy download ru_core_news_sm
python3 -m spacy download ru_core_news_lg
pip3 install spacy-lookups-data
```

Метрики модели:
```
Accuracy | Precision | Recall
0.854      0.839       0.870
```

**Accuracy** - среднее гармоническое точности и полноты

**Precision** - отношение истинно положительных результатов ко всем элементам, отмеченным моделью как положительные

**Recall** - отношение истинно положительных отзывов ко всем фактическим положительным отзывам
