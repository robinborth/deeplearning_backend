# TensorFlow and tf.keras
import tensorflow as tf
from tensorflow.keras import layers, models, losses
# helper libraries
import numpy as np

# loading data
(train_images, train_labels), (test_images, test_labels) = tf.keras.datasets.mnist.load_data()

# map the images values between 0 and 1
train_images, test_images = train_images / 255.0, test_images / 255.0

# reshape the images to match the input_shape in Conv2D: from (60000, 28, 28) -> to (60000, 28, 28, 1)
train_images, test_images = train_images.reshape(-1, 28,28,1), test_images.reshape(-1, 28,28,1)

# split train_images in 80 % train_images and 20 % validation_images
split_value = int(0.8 * len(train_images))
validation_images, validation_labels = train_images[split_value:], train_labels[split_value:]
train_images, train_labels = train_images[:split_value], train_labels[:split_value]

# dataset with 10 different classes
class_names = ['zero', 'one', 'two', 'three', 'four', 'five', 'six', 'seven', 'eight', 'nine']

# keras API for creating the CNN architecture
model = models.Sequential([
    layers.Conv2D(28, (3, 3), activation='relu', input_shape=(28, 28, 1)),
    layers.MaxPooling2D((2,2)),
    layers.Conv2D(64,(3,3), activation='relu'),
    layers.MaxPooling2D((2,2)),
    layers.Conv2D(64,(3,3), activation='relu'),
    layers.Flatten(),
    layers.Dense(64, activation='relu'),
    layers.Dense(10),
])

# compile and train the model
model.compile(optimizer='adam',
              loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
              metrics=['accuracy'])

history = model.fit(train_images, train_labels, epochs=10,
                    validation_data=(validation_images, validation_labels))

# valuating the loss and the accuracy against the test_images
loss, acc = model.evaluate(test_images,  test_labels, verbose=2)
# saving the results in 'model_results.txt'
with open('model_results.txt', 'w') as file:
    file.write(f"loss: {loss} \naccuracy: {acc}")

# saving the model
model = models.Sequential([model, layers.Softmax()])
model.save('saved_model/mnist_model')
