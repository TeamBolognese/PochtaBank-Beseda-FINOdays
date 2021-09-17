import spacy
import os
import random
from spacy.util import minibatch, compounding
from spacy.training import Example


model_path = "ru_model"

# Загрузка данных для тестирования из файла
def load_data(directory: str, limit: int = 0) -> list:
    reviews = []
    for label in ["pos", "neg"]:
        labeled_directory = f"{directory}\{label}"
        for review in os.listdir(labeled_directory):
            if review.endswith(".txt"):
                path = f"{labeled_directory}\{review}"
                with open(path, encoding='utf-8') as file:
                    text = file.read()
                    text = text.replace("<br />", " ")
                    if text.strip():
                        spacy_label = {
                            "cats": {
                                "pos": "pos" == label,
                                "neg": "neg" == label,
                            }
                        }
                        reviews.append((text, spacy_label))
    random.shuffle(reviews)

    if limit:
        reviews = reviews[:limit]
    return reviews


# Тренировка модели
def train_model(training_data: list, testing_data: list, iterations: int = 20) -> None:
    nlp = spacy.load('ru_core_news_sm')
    
    if "textcat" not in nlp.pipe_names:
        nlp.add_pipe("textcat", last=True)
        
    textcat = nlp.get_pipe("textcat")

    textcat.add_label("pos")
    textcat.add_label("neg")

    with nlp.select_pipes(enable="textcat"):
        optimizer = nlp.begin_training()

        print("i.\tLoss\t\tPrec.\tRec.\tAccuracy")
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
                    testing_data=testing_data
                )
                print(f"{i+1}.\t{losses['textcat']:9.6f}\t{score['precision']:.3f}\t{score['recall']:.3f}\t{score['f-score']:.3f}")
    
    # Сохраняем модель
    with nlp.use_params(optimizer.averages):
        nlp.to_disk(model_path)


# Тренировка модели
def resume_train_model(training_data: list, testing_data: list, iterations: int = 20) -> None:
    nlp = spacy.load(model_path)
    
    if "textcat" not in nlp.pipe_names:
        nlp.add_pipe("textcat", last=True)
        
    textcat = nlp.get_pipe("textcat")

    textcat.add_label("pos")
    textcat.add_label("neg")

    with nlp.select_pipes(enable="textcat"):
        optimizer = nlp.resume_training()

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
                    testing_data=testing_data
                )
                print(f"{i+1}.\t{losses['textcat']:9.6f}\t{score['precision']:.3f}\t{score['recall']:.3f}\t{score['accuracy']:.3f}")
    
    # Сохраняем модель
    with nlp.use_params(optimizer.averages):
        nlp.to_disk(model_path)


# Оценка модели на тестовых данных
def evaluate_model(tokenizer, textcat, testing_data: list) -> dict:
    reviews, labels = zip(*testing_data)
    reviews = (tokenizer(review) for review in reviews)
    # Указываем TP как малое число, чтобы в знаменателе
    # не оказался 0
    TP, FP, TN, FN = 1e-8, 0, 0, 0
    for i, review in enumerate(textcat.pipe(reviews)):
        true_label = labels[i]['cats']
        score_pos = review.cats['pos'] 
        if true_label['pos']:
            if score_pos >= 0.5:
                TP += 1
            else:
                FN += 1
        else:
            if score_pos >= 0.5:
                FP += 1
            else:
                TN += 1    
    precision = TP / (TP + FP)
    recall = TP / (TP + FN)
    accuracy = 2 * precision * recall / (precision + recall)
    return {"precision": precision, "recall": recall, "accuracy": accuracy}


train = load_data(directory="")
split = int(len(train) * 0.8)
test = train[split:]
train = train[:split]

# print(f"test len -> {len(test)}")
# print(f"train len -> {len(train)}")

resume_train_model(train, test, 10)