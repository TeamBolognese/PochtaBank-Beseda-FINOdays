import spacy

path = "E:\Education\Hacks\Finodays-2021-2-stage\\test1\\"

nlp = spacy.load("en_core_web_sm").from_disk(f"{path}\\model1")