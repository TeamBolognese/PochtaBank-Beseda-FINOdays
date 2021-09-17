import spacy


model_path = "ru_model"

# Загрузка модели из файла и проверка на примере
def test_model(input_data: str):
    loaded_model = spacy.load(model_path)
    parsed_text = loaded_model(input_data)

    if parsed_text.cats["pos"] > parsed_text.cats["neg"]:
        prediction = "Положительный"
        scope = parsed_text.cats["pos"]
    else:
        prediction = "Негативный"
        scope = parsed_text.cats["neg"]
    print(f"Сообщение: {input_data}\nТональный окрас: {prediction}\nScore: {scope:.3f}")


print("Введите сообщение")
message = str(input())
test_model(input_data=message)