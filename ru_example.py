import spacy

nlp = spacy.load('ru_core_news_sm')

text = "Привет, бро! Смотрели вчера футбол? Классно Месси забил, не так ли?"

doc = nlp(text)
token_list = [token for token in doc]

print("Токенизация")
print(token_list)

filtered_tokens = [token for token in doc if not token.is_stop]

print("Удаление стоп-слов")
print(filtered_tokens)

lemmas = [
    f"Token: {token}, lemma: {token.lemma_}"
    for token in filtered_tokens
]

print("Лемматизация")
print(lemmas)
