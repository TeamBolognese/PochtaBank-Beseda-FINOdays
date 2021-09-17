import spacy
import os
import random
from spacy.util import minibatch, compounding
from spacy.training.example import Example

model_path = "en_model"

def load_training_data(
    data_direstory: str,
    split: float = 0.8,
    limit: int = 0
) -> tuple:
    reviews = []
    for label in ["pos", "neg"]:
        labeled_directory = f"{data_direstory}/{label}"
        for review in os.listdir(labeled_directory):
            if review.endswith(".txt"):
                with open(f"{labeled_directory}/{review}", encoding='utf-8') as f:
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
    nlp = spacy.load("en_core_web_sm")
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
        )
        for i in range(iterations):
            losses = {}
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
                        losses=losses
                    )
            with textcat.model.use_params(optimizer.averages):
                score = evaluate_model(
                    tokenizer=nlp.tokenizer,
                    textcat=textcat,
                    test_data=test_data
                )
                print(f"{i+1}.\t{losses['textcat']:9.6f}\t{score['precision']:.3f}\t{score['recall']:.3f}\t{score['accuracy']:.3f}")
    
    with nlp.use_params(optimizer.averages):
        nlp.to_disk(model_path)

def resume_train_model(
    training_data: list,
    test_data: list,
    iterations: int = 20,
) -> None:
    nlp = spacy.load(model_path)
    if "textcat" not in nlp.pipe_names:
        nlp.add_pipe("textcat", last=True)
    
    textcat = nlp.get_pipe("textcat")

    textcat.add_label("pos")
    textcat.add_label("neg")

    with nlp.select_pipes(enable="textcat"):
        optimizer = nlp.resume_training()
        
        print("i.\tLoss\t\tPrec.\tRec.\tAccuracy")
        batch_sizes = compounding(
            4.0, 32.0, 1.001
        ) 
        for i in range(iterations):
            losses = {}
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
                        losses=losses
                    )
            with textcat.model.use_params(optimizer.averages):
                score = evaluate_model(
                    tokenizer=nlp.tokenizer,
                    textcat=textcat,
                    test_data=test_data
                )
                print(f"{i+1}.\t{losses['textcat']:9.6f}\t{score['precision']:.3f}\t{score['recall']:.3f}\t{score['accuracy']:.3f}")
    
    with nlp.use_params(optimizer.averages):
        nlp.to_disk(model_path)

def evaluate_model(tokenizer, textcat, test_data: list) -> dict:
    reviews, labels = zip(*test_data)
    reviews = (tokenizer(review) for review in reviews)

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
    accuracy = 2 * precision * recall / (precision + recall)
    return {"precision": precision, "recall": recall, "accuracy": accuracy}

# train, test = load_training_data(limit=5000)
# train_model(train, test)