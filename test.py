import spacy

model_path = "en_model"

def test_model(input_data: str):
    loaded_model = spacy.load(model_path)
    parsed_text = loaded_model(input_data)

    if parsed_text.cats["pos"] > parsed_text.cats["neg"]:
        prediction = "Positive"
        scope = parsed_text.cats["pos"]
    else:
        prediction = "Negative"
        scope = parsed_text.cats["neg"]
    print(f"Message: {input_data}\nTonal: {prediction}\nScore: {scope:.3f}")



print("Write your message")
message = str(input())
test_model(input_data=message)