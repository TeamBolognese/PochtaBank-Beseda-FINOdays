import spacy
import os
import random
from spacy.util import minibatch, compounding
from spacy.training.example import Example


path = "E:\Education\Hacks\Finodays-2021-2-stage\\test1"


# Загрузка данных для тестирования из файла
def load_training_data(
    data_direstory: str = "kinopoisk",
    split: float = 0.8,
    limit: int = 0
) -> tuple:
    # Загрузка данных из файла
    reviews = []
    for label in ["good", "bad"]:
        labeled_directory = f"{data_direstory}/{label}"
        for review in os.listdir(labeled_directory):
            if review.endswith(".txt"):
                p = f"{labeled_directory}/{review}"
                with open(p, encoding='utf-8') as f:
                    text = f.read()
                    text = text.replace("<br />", "\n\n")
                    if text.strip():
                        spacy_label = {
                            "cats": {
                                "good": "good" == label,
                                "bad": "bad" == label,
                            }
                        }
                        reviews.append((text, spacy_label))
    random.shuffle(reviews)

    if limit:
        reviews = reviews[:limit]
    split = int(len(reviews) * split)
    return reviews[:split], reviews[split:]


# Тренировка модели и тестирование модели
def train_model(
    training_data: list,
    test_data: list,
    iterations: int = 20,
) -> None:
    # Строим конвейер
    nlp = spacy.load('ru_core_news_lg')
    if "textcat" not in nlp.pipe_names:
        nlp.add_pipe("textcat", last=True)
    
    textcat = nlp.get_pipe("textcat")

    textcat.add_label("good")
    textcat.add_label("bad")

    # Обучаем только textcat
    training_excluded_pipes = [
        pipe for pipe in nlp.pipe_names if pipe != "textcat"
    ]

    with nlp.disable_pipes(training_excluded_pipes):
        optimizer = nlp.begin_training()

        print("i.\tLoss\t\tPrec.\tRec.\tF-score")
        batch_sizes = compounding(
            4.0, 32.0, 1.001
        ) # Генератор бесконечно последовательности входных чисел

        for i in range(iterations):
            losses = {}
            random.shuffle(training_data)
            batches = minibatch(training_data, size=batch_sizes)

            for batch in batches:
                for text, labels in batch:
                    doc = nlp.make_doc(text)
                    example = Example.from_dict(doc, labels)
                    nlp.update(
                        [example],
                        sgd=optimizer,
                        drop=0.2,
                        losses=losses
                    )

            with textcat.model.use_params(optimizer.averages):
                score = evaluate_model(
                    tokenizer=nlp.tokenizer,
                    textcat=textcat,
                    test_data=test_data
                )
                print(f"{i+1}.\t{losses['textcat']:9.6f}\t{score['precision']:.3f}\t{score['recall']:.3f}\t{score['f-score']:.3f}")
    
    # Сохраняем модель
    with nlp.use_params(optimizer.averages):
        nlp.to_disk(f"{path}\\ru_model")
            

# Оценка тонального окраса сообщения
def evaluate_model(tokenizer, textcat, test_data: list) -> dict:
    reviews, labels = zip(*test_data)
    reviews = (tokenizer(review) for review in reviews)
    # Указываем TP как малое число, 
    # чтобы в знаменателе не оказался 0
    TP, FP, TN, FN = 1e-8, 0, 0, 0
    for i, review in enumerate(textcat.pipe(reviews)):
        true_label = labels[i]['cats']
        scope_pos = review.cats['good']
        if true_label['good']:
            if scope_pos >= 0.5:
                TP += 1
            else:
                FN += 1
        elif true_label['bad']:
            if scope_pos >= 0.5:
                FP += 1
            else:
                TN += 1
    precision = TP / (TP + FP)
    recall = TP / (TP + FN)
    f_score = 2 * precision * recall / (precision + recall)
    return {"precision": precision, "recall": recall, "f-score": f_score}


train, test = load_training_data(limit=10)
train_model(train, test, 10)


# Загрузка модели из файла и проверка на примере
def test_model(input_data: str):
    loaded_model = spacy.load(f"{path}\\ru_model")
    parsed_text = loaded_model(input_data)

    if parsed_text.cats["good"] > parsed_text.cats["bad"]:
        prediction = "Положительный"
        scope = parsed_text.cats["good"]
    else:
        prediction = "Негативный"
        scope = parsed_text.cats["bad"]
    print(f"Текст обзора: {input_data}\nТональный окрас: {prediction}\nScore: {scope:.3f}")


good_str = """
Мне очень понравилось, фильм прекрасен, я в восторге!
"""

bad_str = """
Мне не понравилось, выглядело очень плохо.
"""

rep = str(input())

test_model(input_data=rep)