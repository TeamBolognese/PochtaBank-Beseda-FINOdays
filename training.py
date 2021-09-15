from re import T
import spacy
import os
import random
from spacy.util import minibatch, compounding
from spacy.training.example import Example

path = "E:\Education\Hacks\Finodays-2021-2-stage\\test1"

def load_training_data(
    data_direstory: str = "aclImdb/train",
    split: float = 0.8,
    limit: int = 0
) -> tuple:
    # Загрузка данных из файла
    reviews = []
    for label in ["pos", "neg"]:
        labeled_directory = f"{data_direstory}/{label}"
        for review in os.listdir(labeled_directory):
            if review.endswith(".txt"):
                with open(f"{labeled_directory}/{review}", encoding='utf-8') as f:
                    # print(f"{labeled_directory}/{review}")
                    text = f.read()
                    text = text.replace("<br />", "\n\n")
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
    split = int(len(reviews) * split)
    return reviews[:split], reviews[split:]

def train_model(
    training_data: list,
    test_data: list,
    iterations: int = 20,
) -> None:
    # Строим конвейер
    nlp = spacy.load("en_core_web_sm")
    if "textcat" not in nlp.pipe_names:
        nlp.add_pipe("textcat", last=True)
    
    textcat = nlp.get_pipe("textcat")

    textcat.add_label("pos")
    textcat.add_label("neg")

    # Обучаем только textcat
    training_excluded_pipes = [
        pipe for pipe in nlp.pipe_names if pipe != "textcat"
    ]
    with nlp.disable_pipes(training_excluded_pipes):
        optimizer = nlp.begin_training()
        # Итерации обучения
        print("Начинаем обучение")
        print("Loss\t\tPrec.\tRec.\tF-score")
        batch_sizes = compounding(
            4.0, 32.0, 1.001
        ) # Генератор бесконечно последовательности входных чисел
        for i in range(iterations):
            loss = {}
            random.shuffle(training_data)
            batches = minibatch(training_data, size=batch_sizes)
            for batch in batches:
                for text, annotations in batch:
                    doc = nlp.make_doc(text)
                    example = Example.from_dict(doc, annotations)
                    nlp.update(
                        [example], 
                        sgd=optimizer,
                        drop=0.2, 
                        losses=loss
                    )
            with textcat.model.use_params(optimizer.averages):
                evaluation_results = evaluate_model(
                    tokenizer=nlp.tokenizer,
                    textcat=textcat,
                    test_data=test_data
                )
                print(f"{loss['textcat']:9.6f}\t{evaluation_results['precision']:.3f}\t{evaluation_results['recall']:.3f}\t{evaluation_results['f-score']:.3f}")
    
    # Сохраняем модель
    with nlp.use_params(optimizer.averages):
        nlp.to_disk(f"{path}\\model_artifacts")

def evaluate_model(tokenizer, textcat, test_data: list) -> dict:
    reviews, labels = zip(*test_data)
    reviews = (tokenizer(review) for review in reviews)
    # Указываем TP как малое число, 
    # чтобы в знаменателе не оказался 0
    TP, FP, TN, FN = 1e-8, 0, 0, 0
    for i, review in enumerate(textcat.pipe(reviews)):
        true_label = labels[i]['cats']
        scope_pos = review.cats['pos']
        if true_label['pos']:
            if scope_pos >= 0.5:
                TP += 1
            else:
                FN += 1
        else:
            if scope_pos >= 0.5:
                FP += 1
            else:
                TN += 1
    precision = TP / (TP + FP)
    recall = TP / (TP + FN)
    f_score = 2 * precision * recall / (precision + recall)
    return {"precision": precision, "recall": recall, "f-score": f_score}

# train, test = load_training_data(limit=5000)
# train_model(train, test)

def test_model(input_data: str):
    loaded_model = spacy.load(f"{path}\\model_artifacts")
    parsed_text = loaded_model(input_data)

    if parsed_text.cats["pos"] > parsed_text.cats["neg"]:
        prediction = "Положительный ответ"
        scope = parsed_text.cats["pos"]
    else:
        prediction = "Негативный ответ"
        scope = parsed_text.cats["neg"]
    print(f"Текст обзора: {input_data}\nПредсказание: {prediction}\nScore: {scope:.3f}")

TEST_REVIEW = """
good
"""

test_model(input_data=TEST_REVIEW)