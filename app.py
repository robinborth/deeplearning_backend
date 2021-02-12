from flask import Flask
from flask_cors import CORS
import tensorflow as tf
import numpy as np

app = Flask(__name__)
CORS(app)


def load_mnist_model():
    return tf.keras.models.load_model('saved_model/mnist_model')

def convert_image_to_numpy_array(image):
    #image = np.zeros((28,28))
    #image[:, 15] = np.full((1,), 1)
    #image[:, 14] = np.full((1,), 1)
    #image[:, 13] = np.full((1,), 1)
    #image = image.reshape(-1, 28,28,1)
    #TODO: Convert the image o
    return image

def predict_image(image, model):
    predict = model.predict(image)
    return np.argmax(predict)

@app.route('/mnist/api/v1.0/model', methods=['POST'])
def evaluate_model():
    request = request.get_json()
    image = ''
    print(request)
    model = load_mnist_model()
    image = convert_image_to_numpy_array(image)
    prediction = predict_image(image, model)


if __name__ == '__main__':
    app.run(debug=True)