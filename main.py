from flask import Flask, request, render_template, jsonify
import numpy as np
import pickle

app = Flask(__name__)
model = None


@app.route('/', methods=['POST', 'GET'])
def index():
    #  Если просто зайти на главную в браузере — получим html.
    if request.method == 'GET':
        return render_template('index.html')
    data = request.json
    pixels = np.array(data['pixels'])
    pixels = pixels.T.reshape(28 * 28)
    value = model.predict([pixels, ])[0]
    probs = model.predict_proba([pixels, ])[0]
    probs = list(zip(range(0, 9), probs))
    probs = sorted(probs, key=lambda x: x[1], reverse=True)
    response = {'predicted': value, 'probs': probs[:5]}

    return jsonify(response)


if __name__ == '__main__':
    with open('model_digits.pickle', 'rb') as f:
        model = pickle.load(f)
    app.run()
