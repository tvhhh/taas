import gensim
import numpy as np
import re
import torch
import sys

from flask import Flask, jsonify, request
from flask_api import status
from flask_cors import cross_origin
from gensim.corpora import Dictionary
from models.modeling_sus import SusForConditionalGeneration
from nltk.stem import WordNetLemmatizer
from transformers.models.pegasus.tokenization_pegasus import PegasusTokenizer

app = Flask(__name__)

taas, taas_tokenizer = None, None
pegasus, pegasus_tokenizer = None, None
dictionary = None
lemmatizer = None
stop_words = None

device = "cuda" if torch.cuda.is_available() else "cpu"


def init():
    print("Initalizing models...")

    global lemmatizer
    lemmatizer = WordNetLemmatizer()

    global stop_words
    with open("../data/stopwords/stopwords_english.txt", "r") as reader:
        raw_text = reader.read()
        stop_words = set(word.strip() for word in raw_text.split("\n"))

    global dictionary
    dictionary = Dictionary.load_from_text("../data/corpus/dict.txt")

    global taas
    global taas_tokenizer
    taas = SusForConditionalGeneration.from_pretrained("../checkpoints/taas-batm")
    taas_tokenizer = PegasusTokenizer.from_pretrained("../checkpoints/taas-batm")
    taas.eval()
    
    global pegasus
    global pegasus_tokenizer
    pegasus = SusForConditionalGeneration.from_pretrained("../checkpoints/pegasus")
    pegasus_tokenizer = PegasusTokenizer.from_pretrained("../checkpoints/pegasus")
    pegasus.eval()

    print("Successfully initialized models")


def get_corpus(text):
    text = re.sub(r"[^\x00-\x7F]+", "", text)
    tokens = gensim.utils.tokenize(text)
    tokens = [lemmatizer.lemmatize(token) for token in tokens]
    tokens = re.sub(r"[^a-zA-Z ]", "", " ".join(tokens)).split()
    tokens = [token.lower() for token in tokens if token.lower() not in stop_words]
    return tokens


@app.route("/api/summarize", methods=["POST"])
@cross_origin()
def summarize():
    body = request.json
    content, model_name = body["text"], body["model"]

    model, model_tokenizer = None, None
    if model_name == "taas":
        model, model_tokenizer = taas, taas_tokenizer
    elif model_name == "pegasus":
        model, model_tokenizer = pegasus, pegasus_tokenizer
    else:
        return f"Unknown model {model_name}", status.HTTP_400_BAD_REQUEST
    
    batch = model_tokenizer([content], truncation=True, padding="longest", return_tensors="pt")
    if model_name == "taas":
        corpus = get_corpus(content)
        gensim_bows = dictionary.doc2bow(corpus)
        bows = np.zeros((1, len(dictionary.token2id)))
        if len(gensim_bows) > 0:
            ids, freqs = zip(*gensim_bows)
            bows[0][list(ids)] = list(freqs)
        batch["bag_of_words"] = torch.from_numpy(bows).float()

    batch = batch.to(device)
    summary = model.generate(**batch)
    tgt_text = model_tokenizer.batch_decode(summary, skip_special_tokens=True)[0]

    return jsonify({"summary": tgt_text}), status.HTTP_200_OK


if __name__ == "__main__":
    init()
    port = int(sys.argv[1])
    app.run(host="0.0.0.0", port=port)
