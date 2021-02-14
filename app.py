import numpy as np
import tensorflow as tf
from flask import Flask, request, jsonify
from flask_cors import CORS


app = Flask(__name__)
CORS(app)


# api request to prevent heroku to sleep
@app.route('/mnist/api/v1.0/status', methods=['POST'])
def status():
    return jsonify({'status': 'ok'})


def load_mnist_model():
    return tf.keras.models.load_model('saved_model/mnist_model')


def convert_json_to_numpy_array(json):
    image = np.array(json)
    image = image.reshape(-1, 28, 28, 1)
    return image


def predict_image(image, model):
    predict = model.predict(image)
    return np.argmax(predict)


# returns the prediction of the drawn digit
@app.route('/mnist/api/v1.0/model', methods=['POST'])
def evaluate_model():
    json = request.get_json()
    image_json = json.get('image')
    if image_json:
        image = convert_json_to_numpy_array(image_json)
        model = load_mnist_model()
        prediction = predict_image(image, model)
        return jsonify({'prediction': int(prediction)})
    else:
        return jsonify({'error': 'error'})


if __name__ == '__main__':
    app.run()
