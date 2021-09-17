from bottle import default_app, route, response, run
import spacy
from urllib.parse import unquote

model_name = "ru_model"
model_path = f"{model_name}/"

def test_model(input_data: str):
    loaded_model = spacy.load(model_path)
    parsed_text = loaded_model(input_data)

    if parsed_text.cats["pos"] > parsed_text.cats["neg"]:
        prediction = "'pos'"
        scope = parsed_text.cats["pos"]
    else:
        prediction = "'neg'"
        scope = parsed_text.cats["neg"]
    return (f"'tonal': {prediction}, 'score': {scope:.3f}")

@route('/')
def hello_world():
    return 'Api is working<br><br>Example: /api/<text>'

@route('/api/<text>')
def userinfo(text):
    response.add_header('Access-Control-Allow-Origin', '*')
    text = unquote(text).replace('+',' ') # URL Decode
    result = test_model(input_data=text) # Analysis
    return '{' + result + '}'

run(host='localhost', port=80, debug=True)
