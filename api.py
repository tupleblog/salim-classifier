from flask import Flask, render_template, request, jsonify
from flask_cors import CORS, cross_origin

import os

os.environ["CUDA_VISIBLE_DEVICES"] = "-1"

from transformers import AutoTokenizer, AutoModelForSequenceClassification, pipeline

from torch.nn.functional import softmax

import timeit


# download model from hub
TOKENIZER = AutoTokenizer.from_pretrained("tupleblog/salim-classifier")
MODEL = AutoModelForSequenceClassification.from_pretrained("tupleblog/salim-classifier")

app = Flask(__name__)
cors = CORS(app)


def predict(model, tokenizer, text):

    device = "cpu"

    _inputs = tokenizer(text, return_tensors="pt").to(device)

    outputs = model(**_inputs)

    result = softmax(outputs[0], dim=1).cpu().data.numpy().round(6).tolist()

    result = result[0]

    format_result = [
        {"label": label, "score": float(result[index])}
        for index, label in model.config.id2label.items()
    ]

    return format_result


@app.route("/", methods=["POST"])
def index():

    text = request.form.get("text", "")

    print(text)

    start_time = timeit.default_timer()

    result = predict(MODEL, TOKENIZER, text)

    usage_time = round(timeit.default_timer() - start_time, 3)

    return jsonify({"result": result, "usage_time": usage_time})


if __name__ == "__main__":
    app.run(debug=True, host="0.0.0.0", port=5000)
